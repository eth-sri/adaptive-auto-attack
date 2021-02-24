from typing_extensions import final
import numpy as np
import torch

from attack.criteria import TargetedMisclassification
from utils.utils import MAX_BATCH, batch_forward, batch_is_adv


class Attack:
    def __init__(self):
        return

    @final
    def __call__(self, model, images, labels, epsilon):
        return self.run(model, images, labels, epsilon)

    def run(self, model, images, labels, epsilon):
        raise NotImplementedError


class SeqAttack(Attack):
    def __init__(self, attack_list, detector=None):
        super(SeqAttack, self).__init__()
        assert(isinstance(attack_list, list)), "Argument attack_list has to be a list"
        self.attack_list = attack_list
        self.detector = detector

    def run(self, model, images, labels, epsilon):
        nimages = int(images.shape[0])
        remain_idx = np.arange(nimages)
        adv_total = images.clone()
        for attack in self.attack_list:
            advs = attack(model, images[remain_idx], labels[remain_idx], epsilon)
            is_adv = batch_is_adv(model, advs, labels[remain_idx])
            # check if the attack pass the detector
            if self.detector is not None:
                det = batch_forward(self.detector, advs)
                if det.dim() == 2:
                    det = torch.argmax(det, dim=1).to(torch.bool)
                else:
                    assert(det.dim() == 1), "The dimension of the detector output has to be 1 or 2"
                is_adv = is_adv & det.to(is_adv.device)
            remove_list = []
            adv_total[remain_idx] = advs
            for i in range(is_adv.shape[0]):
                if is_adv[i]:
                    remove_list.append(i)
            remain_idx = np.delete(remain_idx, remove_list, axis=0)
            if len(remain_idx) == 0:
                break
        return adv_total

    def append(self, attack):
        self.attack_list.append(attack)


class RepeatAttack(SeqAttack):
    def __init__(self, attack, times, detector=None):
        self.attack = attack
        self.times = times
        attack_list = [attack] * times
        super().__init__(attack_list, detector=detector)

    def run(self, model, images, labels, epsilon):
        self.attack_list = [self.attack] * self.times
        return super().run(model, images, labels, epsilon)


class TargetedAttack(Attack):
    def __init__(self, attack, n_target_classes=3, detector=None):
        super().__init__()
        self.attack = attack
        self.n_target_classes = n_target_classes
        self.detector = detector

    def run(self, model, images, labels, epsilon):
        """
        The targeted implementation is to pass in TargetedMisclassification label into the attack function
        The attack has to support the TargetedMisclassification.
        """
        N = int(images.shape[0])
        remain_idx = np.arange(N)
        adv_total = images.clone()

        output = batch_forward(model, images, MAX_BATCH)
        y_target = output.sort(dim=1)[1][:, - self.n_target_classes - 1: -1]  # get the top n_target_classes labels

        for target_class in range(self.n_target_classes-1, -1, -1):  # evaluate higher logits first
            criterion = TargetedMisclassification(y_target[remain_idx, target_class],)
            advs = self.attack(model, images[remain_idx], criterion, epsilon)
            is_adv = batch_is_adv(model, advs, labels[remain_idx])
            if self.detector is not None:
                det = batch_forward(self.detector, advs)
                if det.dim() == 2:
                    det = torch.argmax(det, dim=1).to(torch.bool)
                else:
                    assert(det.dim() == 1)
                is_adv = is_adv & det.to(is_adv.device)
            remove_list = []
            adv_total[remain_idx] = advs
            for i in range(is_adv.shape[0]):
                if is_adv[i]:
                    remove_list.append(i)
            remain_idx = np.delete(remain_idx, remove_list, axis=0)
            if len(remain_idx) == 0:
                break
        return adv_total
