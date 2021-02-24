from art.utils import is_probability
from utils.utils import misclassification_criterion, misclassification_criterion_acc_check, \
    misclassification_criterion_rerr, print_eval_result
from attack.attack_pool import *
from attack.attack_base import RepeatAttack, TargetedAttack, SeqAttack
import numpy as np
import time


def attack_from_config(config):
    att = config["attack"]
    int_param_list = ['max_iter', 'EOT', 'repeat', 'n_queries']
    for item in int_param_list:
        if item in config:
            config[item] = int(config[item])
    attparam = {}
    refparam = get_param_dict(att)
    for i in config:
        # using ref parameter to filter out the parameters, may change impl later
        if i in refparam:
            attparam[i] = config[i]
    attack = att.set_param(attparam)
    if "set_targeted" in config:
        if config["set_targeted"]:
            if "n_target_classes" in config:
                attack = TargetedAttack(attack, n_target_classes=config["n_target_classes"])
            else:
                attack = TargetedAttack(attack)
    attack = RepeatAttack(attack, config["repeat"])
    return attack


def attack_from_trial(trial):
    """
    Reconstruct the attack parameters from trial result
    In the trial config dictionary, the attack parameter is set to the trial if it exists; set to default otherwise
    """
    config = trial.config
    return attack_from_config(config)


def get_config_string(config):
    config_repr = ""
    for i in config:
        if i != 'data':
            config_repr = config_repr + str(i) + " : " + str(config[i]) + "\n"
    return config_repr


def get_ktop_config(scores, k, infolist):
    """
    If the length of the infolist is smaller than k, then return the sorted infolist.
    """
    def get_top_k_idx(scores, k):
        ind = list(np.argsort(scores, axis=0))
        ind.reverse()  # default is increaing, so reverse the sorting result
        for i in range(k):
            # to avoid getting timeout attacks
            if scores[ind[i]] <= -1:
                k = i
                break
        return ind[:k]

    ktop = get_top_k_idx(scores, k)
    best_k_configs = []
    best_k_scores = []
    for idx in ktop:
        best_k_configs.append(infolist[idx])
        best_k_scores.append(scores[idx])
    return ktop, best_k_configs, best_k_scores


class AttackUpdater:
    """
    Object to keep track of the samples in the search
    """
    def __init__(self, data, f_net, g_net_model, epsilon, detector, is_random):
        self.timer = 0
        self.raw_data = data[0].clone()
        self.adv_data = data[0].clone()
        self.labels = data[1].clone()
        self.remain_idx = np.arange(len(data[0]))
        self.epsilon = epsilon
        self.f_net = f_net
        self.g_net_model = g_net_model
        self.n_clean = -1
        self.detector = detector
        self.is_random = is_random
        self.acc_attack = []
        self.n_samples = len(data[0])

    def get_acc_img(self):
        start_time = time.time()
        images = self.adv_data[self.remain_idx]
        labels = self.labels[self.remain_idx]
        nimages = images.shape[0]
        relative_idx = np.arange(nimages)
        attack = NoAttack()
        adv_img = attack(self.g_net_model, images[relative_idx], labels[relative_idx], self.epsilon)
        is_adv, mask = misclassification_criterion(self.f_net, adv_img, labels, detector=True)
        is_adv = is_adv | ~mask
        remove_list = []
        for i in range(is_adv.shape[0]):
            if is_adv[i]:
                remove_list.append(i)
        self.adv_data[self.remain_idx[remove_list]] = adv_img[remove_list]
        if not self.is_random:
            # if random, also attack on the misclassified samples
            self.remain_idx = np.delete(self.remain_idx, remove_list, axis=0)
        relative_idx = np.delete(relative_idx, remove_list, axis=0)
        newdata = (images[relative_idx].clone(), labels[relative_idx].clone())
        self.timer += (time.time() - start_time)
        self.n_clean = len(newdata[0])
        return newdata

    def get_robust_img(self, attack_list):
        start_time = time.time()
        self.acc_attack += attack_list
        images = self.raw_data[self.remain_idx]
        labels = self.labels[self.remain_idx]
        nimages = images.shape[0]
        relative_idx = np.arange(nimages)
        for attack in attack_list:
            adv_img = attack(self.g_net_model, images[relative_idx], labels[relative_idx], self.epsilon)
            is_adv = misclassification_criterion(self.f_net, adv_img, labels)
            remove_list = []
            for i in range(is_adv.shape[0]):
                if is_adv[i]:
                    remove_list.append(i)
            self.adv_data[self.remain_idx] = adv_img
            self.remain_idx = np.delete(self.remain_idx, remove_list, axis=0)
            relative_idx = np.delete(relative_idx, remove_list, axis=0)
        newdata = (images[relative_idx].clone(), labels[relative_idx].clone())
        self.timer += (time.time() - start_time)
        return newdata

    def evaluation(self):
        if not self.is_random:
            is_adv, mask = misclassification_criterion_rerr(self.f_net, self.adv_data, self.labels, self.raw_data)
            num_adv = float(torch.sum(is_adv))
            num_not_adv = float(torch.sum(~is_adv))
        else:
            num_adv = 0
            num_not_adv = 0
            for i in range(10):
                is_adv, mask = misclassification_criterion_rerr(self.f_net, self.adv_data, self.labels, self.raw_data)
                num_adv += float(torch.sum(is_adv)) / 10
                num_not_adv += float(torch.sum(~is_adv)) / 10
        if self.detector:
            denominator = float(torch.sum(mask))
            numerator = denominator - num_adv
        else:
            denominator = self.n_clean
            numerator = num_not_adv
        attack_time = self.timer
        return self.n_clean, denominator, numerator, attack_time


def attackUpdateEval(attackUpdater, n_samples, start_time):
    n_clean, denominator, n_robust, attack_time = attackUpdater.evaluation()

    acc = n_clean / n_samples * 100
    robustness = n_robust / n_samples * 100
    ASR = (denominator - n_robust) / denominator * 100
    total_time = time.time() - start_time
    eval_result = [acc, robustness, ASR, attack_time, total_time]
    # print result
    print_eval_result(eval_result)
    return eval_result


def get_robust_img(attack_list, f_net, g_net_model, data, epsilon, norm):
    if not isinstance(attack_list, list):
        attack_list = [attack_list]
    images = data[0]
    labels = data[1]
    nimages = images.shape[0]
    remain_idx = np.arange(nimages)
    for attack in attack_list:
        adv_img, _ = attack(g_net_model, images[remain_idx], labels[remain_idx], epsilon)
        is_adv = misclassification_criterion(f_net, adv_img, labels)
        remove_list = []
        for i in range(is_adv.shape[0]):
            if is_adv[i]:
                remove_list.append(i)
        remain_idx = np.delete(remain_idx, remove_list, axis=0)
        if len(remain_idx) == 0:
            break
    newdata = (images[remain_idx], labels[remain_idx])
    return newdata


def get_data_samples(data, num, offset=0, random=True):
    length = data[0].shape[0]
    if random:
        indices = np.random.permutation(length)[:num]
        pdata = (data[0][indices], data[1][indices])
    else:
        if length < offset+num:
            print("WARNING: getting less data than desired due to the length of data! If offset non-zero, it is automatically adjusted")
            if length > num:
                offset = length - num
        pdata = (data[0][offset:offset+num], data[1][offset:offset+num])
    return pdata


def budget_awareness_prior(config):
    """
    To estimate how expensive the attack is, the estimation includes the number of forwarding/backward per sample.
    Such heuristic gives good run time estimation and it can be used to eliminate the candidate if runtime is obviously too long
    This module can be easily be replaced by a learned approach: a network to estimate the number of forwarding/ backwarding based on the parameters
    """
    att = config["attack"]
    def cond(config, string, default=1):
        if string in config:
            return config["string"]
        else:
            return default

    def targetcond(config, default=False, multiplier=9):
        targeted = cond(config, "targeted", default)  # hardcode 9 for now
        if targeted:
            target_cost = multiplier
        else:
            target_cost = 1
        return target_cost

    if isinstance(att, LinfFastGradientAttack) or isinstance(att, L2FastGradientAttack):
        EOT = cond(config, "EOT", 1)
        repeat = cond(config, "repeat", 1)
        target_cost = targetcond(config)
        return EOT * repeat * target_cost

    elif isinstance(att, LinfProjectedGradientDescentAttack) or isinstance(att, L2ProjectedGradientDescentAttack):
        steps = cond(config, "steps", 40)
        EOT = cond(config, "EOT", 1)
        repeat = cond(config, "repeat", 1)
        target_cost = targetcond(config)
        return EOT * repeat * target_cost * steps

    elif isinstance(att, LinfDeepFoolAttack) or isinstance(att, L2DeepFoolAttack):
        repeat = cond(config, "repeat", 1)
        target_cost = targetcond(config)
        return repeat * 400 * target_cost

    elif isinstance(att, LinfCarliniWagnerAttack):
        max_doubling = cond(config, "max_doubling", 10)
        max_halving = cond(config, "max_halving", 10)
        return 1000 * max_doubling * max_halving

    return None


def check_softmax(model, images):
    modelout = model(images).detach().cpu().numpy()
    is_softmax = True
    for i in range(modelout.shape[0]):
        is_softmax = is_probability(modelout[i, :]) & is_softmax
    return is_softmax


def explode(L, num):
    m = len(L)
    n = m ** num
    result = []
    for i in range(n):
        temp = i
        item = []
        for j in range(num):
            item.append(L[temp % m])
            temp = temp // m
        result.append(item)
    return result