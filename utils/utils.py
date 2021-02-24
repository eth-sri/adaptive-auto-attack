from foolbox.distances import LpDistance
import numpy as np
import eagerpy as ep
from math import ceil
import pandas as pd
from attack.criteria import TargetedMisclassification
import torch

# Here defines the maximum allowed batch size:
MAX_BATCH = 128

def collect_attack_search_statistics(trials):
    typect = {}
    for trial in trials:
        att_type = type(trial[0]['attack'])
        if att_type not in typect:
            typect[att_type] = 1
        else:
            typect[att_type] += 1
    return typect


def collect_scores(trials):
    score_dict = {}
    for trial in trials:
        att_type = type(trial[0]['attack'])
        if att_type not in score_dict:
            score_dict[att_type] = [trial[2]]
        else:
            score_dict[att_type].append(trial[2])
    return score_dict


def collect_scores_df(trials):
    attack_list = []
    score_list = []
    for trial in trials:
        att_type = type(trial[0]['attack'])
        attack_list.append(att_type.name())
        score_list.append(trial[2])
    return pd.DataFrame({"attack": attack_list, "score": score_list})


def collect_param_scores_df(trials):
    attack_list = []
    score_list = []
    param_list = []
    for trial in trials:
        att_type = type(trial[0]['attack'])
        trial[0].pop('attack')
        attack_list.append(att_type.name())
        score_list.append(trial[2])
        param_list.append(trial[0])
    return pd.DataFrame({"attack": attack_list, "score": score_list, 'param': param_list})


def get_progression(trials):
    individual_list = []
    progress_list = []
    curr_max = -100
    for trial in trials:
        individual_list.append(trial[2])
        curr_max = max(curr_max, trial[2])
        progress_list.append(curr_max)
    return individual_list, progress_list


def misclassification_criterion(instance, adv_imgs, labels, detector=False):
    """
    Misclassification Criterion function
    """
    adv_pred, adv_det = instance.predict(torch.clamp(adv_imgs, 0, 1))
    robust_acc_bool = torch.argmax(adv_pred, dim=1).eq(labels.to(instance.device))
    is_adv_det_bool = ~robust_acc_bool & adv_det
    if detector:
        return is_adv_det_bool, adv_det
    else:
        return is_adv_det_bool


def misclassification_criterion_acc_check(instance, adv_imgs, labels, imgs):
    """
    This method is for detectors to remove false-negative samples
    if no detector, it is the same as misclassification_criterion
    Check if the adv_imgs are misclassified adversarial
    """
    pred, det = instance.predict(imgs)
    acc_bool = torch.argmax(pred, dim=1).eq(labels.to(instance.device))
    acc_det_bool = acc_bool & det
    return ~acc_det_bool | misclassification_criterion(instance, adv_imgs, labels)


def misclassification_criterion_rerr(instance, adv_imgs, labels, imgs):
    """
    This method is for detectors to remove false-negative samples
    if no detector, it is the same as misclassification_criterion
    Check if the adv_imgs are misclassified adversarial
    """
    pred, det = instance.predict(imgs)
    acc_err_bool = ~torch.argmax(pred, dim=1).eq(labels.to(instance.device)) & det
    adv_pred, adv_det = instance.predict(torch.clamp(adv_imgs, 0, 1))
    robust_err_bool = ~torch.argmax(adv_pred, dim=1).eq(labels.to(instance.device)) & adv_det
    mask = det | adv_det
    err = acc_err_bool | robust_err_bool
    return err, mask


def get_acc_robust_disturbance(instance, imgs, adv_imgs, labels, norm=ep.inf, verbose=False):
    """
    Return a list: [accuracy, rerr, disturbance, robustness, loss]
    """
    adv_imgs = adv_imgs.clone()
    # deal with nan in the image.
    if (torch.isnan(adv_imgs).any()):
        print("There is nan is the adversarial image, will fill nan pixels with the original img")
        mask = torch.isnan(adv_imgs)
        adv_imgs[mask] = 0
        adv_imgs = mask*imgs + adv_imgs
    assert(not torch.isnan(adv_imgs).any())

    pred, det = instance.predict(imgs)
    pred2, det = instance.predict(imgs)
    is_random = not pred.eq(pred2).all()
    acc_bool = torch.argmax(pred, dim=1).eq(labels.to(instance.device))
    det_bool = acc_bool & det
    acc_err_bool = ~torch.argmax(pred, dim=1).eq(labels.to(instance.device)) & det

    raw_accuracy = float(torch.sum(acc_bool.to(torch.float32)) / len(labels))
    det_accuracy = float(torch.sum(det_bool.to(torch.float32)) / len(labels))

    adv_pred, adv_det = instance.predict(torch.clamp(adv_imgs, 0, 1))
    predsoft = adv_pred.softmax(dim=1)
    ce_loss = -(predsoft[np.arange(labels.shape[0]), labels]+0.001).log()  # 0.001 to avoid extreme values

    if not is_random:
        robust_err_bool = ~torch.argmax(adv_pred, dim=1).eq(labels.to(instance.device)) & adv_det
        mask = det | adv_det
        err_bool = acc_err_bool | robust_err_bool
        num_adv = float(torch.sum(err_bool.to(torch.float32)))
    else:
        num_adv = 0
        mask = det | adv_det
        for _ in range(10):
            adv_pred, adv_det = instance.predict(torch.clamp(adv_imgs, 0, 1))
            robust_err_bool = ~torch.argmax(adv_pred, dim=1).eq(labels.to(instance.device)) & adv_det
            err_bool = acc_err_bool | robust_err_bool
            num_adv += float(torch.sum(err_bool.to(torch.float32)))/10

    remain_list = []
    for i in range(len(labels)):
        if det_bool[i]:
            remain_list.append(i)
    adv_imgs_r = adv_imgs[remain_list, :]
    imgs_r = imgs[remain_list, :]

    rerr = num_adv / float(torch.sum(mask))
    det_attack_accuracy = num_adv / len(labels)
    network_robustness = 1 - rerr
    ce_loss = float(torch.mean(ce_loss))

    assert(len(adv_imgs_r) == len(imgs_r))
    disturbance = get_disturbance(imgs_r, adv_imgs_r, norm)

    if verbose:
        print("Raw accuracy is: {:.2f}%".format(raw_accuracy*100))
        print("Detector accuracy is: {:.2f}%".format(det_accuracy*100))
        print("Robustness error rate is: {:.2f}%".format(rerr*100))
        print("Detector attack success rate is: {:.2f}%".format(det_attack_accuracy*100))
        print("Robustness of the network is: {:.2f}%".format(network_robustness*100))
        print("The average {:.1f} norm disturbance is: {:.4f}".format(norm, disturbance))
        print("Untargeted CE loss is {:.3f}".format(ce_loss))

    result = [det_accuracy*100, rerr*100, disturbance, network_robustness*100, ce_loss]
    return result


def get_disturbance(x, y, norm):
    linf = LpDistance(norm)
    batch_eps = linf(x, y)
    return float(torch.mean(batch_eps))


def print_eval_result(evals):
    print("The accuracy of the network is {:.3f}%".format(evals[0]))
    print("The robustness of the network is {:.3f}%".format(evals[1]))
    print("The ASR of the attack is {:.3f}%".format(evals[2]))
    print("Attack time used is {:.3f}s".format(evals[3]))
    print("Total time used is {:.3f}s".format(evals[4]))
    print()


def batch_is_adv(model, adv_imgs, labels):
    pred = batch_forward(model, adv_imgs)
    if isinstance(labels, TargetedMisclassification):
        is_adv = torch.argmax(pred, dim=1).eq(labels.target_classes.raw)
    else:
        is_adv = torch.argmax(pred, dim=1).ne(labels)
    return is_adv


def batch_forward(model, images, max_batch=MAX_BATCH):
    """
    Batched version of the forward function
    """
    N = images.shape[0]
    nbatchs = ceil(N / max_batch)
    pred_list = []

    with torch.no_grad():
        for i in range(nbatchs):
            pred_list.append(model(images[i * max_batch: (i + 1) * max_batch]))
    return torch.cat(pred_list, dim=0)
