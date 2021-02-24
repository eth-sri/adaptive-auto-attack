""" This file contains all the attacks

Any attack has signature: f, X, Y -> X',
where f is the model, X are the clean images, Y are the labels, X' are the adversarial images
Note1: Every attack returns the adversarial images X', and X' will be used to evaluate robustness
Note2: f in A^3 search is the searched surrogate network and it has no knowledge of the origin network
Note3: f does not have detector component under current formulation
"""

import torch
import torch.nn as nn
import foolbox_custom as fb
from art.classifiers import PyTorchClassifier
from foolbox_custom.distances import LpDistance
import eagerpy as ep
import numpy as np
from math import ceil
from attack.attack_base import Attack
from utils.utils import MAX_BATCH
from attack.criteria import TargetedMisclassification, Misclassification
from autoattack.autoattack import AutoAttack
from autoattack_custom.fab_pt import FABAttack
from autoattack_custom.square import SquareAttack
from autoattack_custom.autopgd_pt import APGDAttack
import attack.cw_attack as carlini
import time


def get_param_dict(attack):
    """
    This dictionary defines the default parameters for every attack
    Any parameters passed into the attack has to match the name in the dictionary, and will be ignored otherwise
    """
    if isinstance(attack, LinfFastGradientAttack) or isinstance(attack, L2FastGradientAttack):
        return {"random_start": True, "EOT": 1, "loss": 'ce', 'mod': None}

    elif isinstance(attack, LinfProjectedGradientDescentAttack) or isinstance(attack, L2ProjectedGradientDescentAttack):
        return {"rel_stepsize": 0.01 / 0.3, "abs_stepsize": None, "steps": 40, "random_start": True, "EOT": 1,
                "loss": 'ce', 'mod': None}

    elif isinstance(attack, LinfDeepFoolAttack) or isinstance(attack, L2DeepFoolAttack):
        return {"steps": 30, "candidates": 10, "overshoot": 0.02, "loss": 'l1', 'mod': None}

    elif isinstance(attack, LinfCarliniWagnerAttack):
        return {"confidence": 0, "targeted": False, "learning_rate": 0.005, "max_iter": 50, "max_halving": 10,
                "max_doubling": 10, "batch_size": MAX_BATCH}

    elif isinstance(attack, L2CarliniWagnerAttack):
        return {"binary_search_steps": 9, "steps": 100, "stepsize": 0.01, "confidence": 0.01, "initial_const": 0.001,
                "abort_early": True}

    elif isinstance(attack, LinfBrendelBethgeAttack) or isinstance(attack, L2BrendelBethgeAttack):
        return {"init_attack": fb.attacks.LinearSearchBlendedUniformNoiseAttack(), "overshoot": 1.1, "steps": 50,
                "lr": 0.01, "lr_decay": 0.5, "lr_num_decay": 20, "momentum": 0.8, "binary_search_steps": 7}

    elif isinstance(attack, L2DDNAttack):
        return {"init_epsilon": 1.0, "steps": 10, "gamma": 0.05}

    elif isinstance(attack, BoundaryAttack):
        return {"init_attack": None, "steps": 2500, "spherical_step": 0.01, "source_step": 0.01,
                "source_step_convergance": 1e-07, "step_adaptation": 1.5, "update_stats_every_k": 10}

    elif isinstance(attack, LinfRepeatedAdditiveUniformNoiseAttack) or isinstance(attack, L2RepeatedAdditiveUniformNoiseAttack):
        return {"repeats": 100, "check_trivial": True}

    elif isinstance(attack, LinfAutoAttack) or isinstance(attack, L2AutoAttack):
        return {"seed": time.time(), "verbose": True, "attacks_to_run": [], 'version': "standard", "is_tf_model": False, "device": 'cuda', "log_path": None}

    elif isinstance(attack, LinfFABAttack) or isinstance(attack, L2FABAttack):
        return {"n_restarts": 5, "n_iter": 100, "seed": time.time(), "eta": 1.05, "beta": 0.9, "verbose": False, "device": 'cuda', "targeted": False}

    elif isinstance(attack, LinfSquareAttack) or isinstance(attack, L2SquareAttack):
        return {"p_init": .8, "n_queries": 5000, "n_restarts": 1, "loss": 'ce', "seed": None, "verbose": False,
                "device": 'cuda', "resc_schedule": False, 'mod': None, }

    elif isinstance(attack, LinfAPGDAttack) or isinstance(attack, L2APGDAttack):
        return {"n_iter": 100, "n_restarts": 1, "seed": time.time(), "loss": 'ce', "eot_iter": 1, "rho": .75, "verbose": False,
                'mod': None, "device": 'cuda'}

    elif isinstance(attack, LinfNaturalEvolutionStrategyAttack) or isinstance(attack, L2NaturalEvolutionStrategyAttack):
        return {"rel_stepsize": 0.01 / 0.3, "abs_stepsize": None, "steps": 40, "random_start": True, "n_samples": 1000,
                "parallel": False, "mod": None}

    elif isinstance(attack, CustomLinfProjectedGradientDescentAttack):
        return {"rel_stepsize": 0.01 / 0.3, "param_dict": {"rel_slope": 1, "freq": 1}, "abs_stepsize": None,
                "steps": 40, "random_start": True, "EOT": 1, "loss": 'ce'}

    else:
        assert False, "Unrecognized attack"


def fbattack(fcn, model, images, criterion, epsilon, kwargs):
    """
    foolbox_custom ported attack implementation
    :param fcn: the foolbox_custom attack function
    :param model: network in DAGModule
    :param images:  test samples
    :param criterion: labels/criterion object
    :param epsilon: epsilon to test with
    :param kwargs: other arguments to the model
    :return: perturbed image, whether the attack is successful or not
    """
    maxbatch = MAX_BATCH  # MAX_BATCH defined in utils.utils
    nsamples = images.shape[0]
    fbmodel = fb.PyTorchModel(model.eval(), bounds=(0, 1))
    attack = fcn(**kwargs)
    adv_imgs_list = []

    nbatchs = ceil(nsamples / maxbatch)
    for i in range(nbatchs):
        if isinstance(criterion, (list, torch.Tensor, np.ndarray)):
            batch_criterion = criterion[i*maxbatch:(i+1)*maxbatch]
        elif isinstance(criterion, TargetedMisclassification):
            batch_criterion = TargetedMisclassification(criterion.target_classes[i*maxbatch:(i+1)*maxbatch])
        elif isinstance(criterion, Misclassification):
            batch_criterion = Misclassification(criterion.labels[i*maxbatch:(i+1)*maxbatch])
        else:
            assert False, "Unsupport criterion type"
        _, adv_imgs_b, _ = attack(fbmodel, images[i*maxbatch:(i+1)*maxbatch], batch_criterion, epsilons=epsilon)
        adv_imgs_list.append(adv_imgs_b)
    adv_imgs = torch.cat(adv_imgs_list, dim=0)
    return adv_imgs


def artattack(fcn, model, images, criterion, epsilon, kwargs):
    """
    adversarial robustness toolbox ported attack implementation
    :param fcn: the foolbox_custom attack function
    :param model: network in DAGModule
    :param images:  test samples
    :param criterion: labels/criterion object
    :param epsilon: epsilon to test with
    :param kwargs: other arguments to the model
    :return: perturbed image, whether the attack is successful or not
    """
    artmodel = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(0,),
        nb_classes=10
    )
    device = model.device
    attack = fcn(classifier=artmodel, **kwargs)
    adv_imgs = torch.as_tensor(attack.generate(x=images.clone().cpu())).to(device)
    linf = LpDistance(ep.inf)
    adv_imgs = linf.clip_perturbation(images, adv_imgs, epsilon)
    return adv_imgs


class AutoAttackBase(Attack):
    def __init__(self, norm, params=None):
        super().__init__()
        self.params = get_param_dict(self)
        self.norm = norm
        if params is not None:
            assert(isinstance(params, dict)), "Parameters has to be type dictionary"
            for key in params:
                if key in self.params:
                    self.params[key] = params[key]

    def run(self, model, images, labels, epsilon):
        raise NotImplementedError

    def batch_attack(self, attack, images, criterion, apgd_flag=False):
        maxbatch = MAX_BATCH  # MAX_BATCH defined in utils.utils
        N = images.shape[0]
        nbatchs = int(np.ceil(N / maxbatch))
        adv_img_list = []
        for i in range(nbatchs):
            if isinstance(criterion, (list, torch.Tensor, np.ndarray)):
                batch_criterion = criterion[i * maxbatch:(i + 1) * maxbatch]
            elif isinstance(criterion, TargetedMisclassification):
                batch_criterion = TargetedMisclassification(criterion.target_classes[i * maxbatch:(i + 1) * maxbatch])
            elif isinstance(criterion, Misclassification):
                batch_criterion = Misclassification(criterion.labels[i * maxbatch:(i + 1) * maxbatch])
            else:
                assert False, "Unsupport criterion type"

            if apgd_flag:
                adv_imgs_b = attack.perturb(images[i*maxbatch: (i+1)*maxbatch], batch_criterion)[1]
            else:
                adv_imgs_b = attack.perturb(images[i*maxbatch: (i+1)*maxbatch], batch_criterion)
            adv_img_list.append(adv_imgs_b.detach())
        return torch.cat(adv_img_list, dim=0)

    def set_param(self, params):
        assert (isinstance(params, dict)), "Parameters has to be type dictionary"
        for key in params:
            if key in self.params:
                self.params[key] = params[key]
        return self


class FoolboxAttack(Attack):
    def __init__(self, fcn, params=None):
        super().__init__()
        self.params = get_param_dict(self)
        if params is not None:
            assert(isinstance(params, dict)), "Parameters has to be type dictionary"
            for key in params:
                if key in self.params:
                    self.params[key] = params[key]
        self.fcn = fcn

    def run(self, model, images, labels, epsilon):
        return fbattack(self.fcn, model, images, labels, epsilon, self.params)

    def set_param(self, params):
        assert (isinstance(params, dict)), "Parameters has to be type dictionary"
        for key in params:
            if key in self.params:
                self.params[key] = params[key]
        return self


class NoAttack(Attack):
    """
    Same as the attack interface but just to get the accuracy and image-wise statistics.
    Note the gradient information is required
    The implementation is to use FGSM with epsilon 0.
    """
    def __init__(self):
        super().__init__()
        self.params = get_param_dict(LinfFastGradientAttack())

    def run(self, model, images, labels, epsilon):
        return fbattack(fb.attacks.LinfFastGradientAttack, model, images, labels, 0, self.params)


class LinfAutoAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="Linf", params=params)

    def run(self, model, images, labels, epsilon):
        attack = AutoAttack(model.eval(), norm=self.norm, eps=epsilon, **self.params)
        adv_imgs = attack.run_standard_evaluation(images, labels)
        return adv_imgs

    @staticmethod
    def name():
        return "AA"


class LinfFABAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="Linf", params=params)

    def run(self, model, images, labels, epsilon):
        attack = FABAttack(model.eval(), norm=self.norm, eps=epsilon, **self.params)
        # adv_imgs = attack.perturb(images, labels)
        adv_imgs = self.batch_attack(attack, images, labels)
        return adv_imgs

    @staticmethod
    def name():
        return "FAB"


class LinfSquareAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="Linf", params=params)

    def run(self, model, images, labels, epsilon):
        attack = SquareAttack(model.eval(), norm=self.norm, eps=epsilon, **self.params)
        # adv_imgs = attack.perturb(images, labels)
        adv_imgs = self.batch_attack(attack, images, labels)
        return adv_imgs

    @staticmethod
    def name():
        return "SQR"


class LinfAPGDAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="Linf", params=params)

    def run(self, model, images, labels, epsilon):
        attack = APGDAttack(model.eval(), norm=self.norm, eps=epsilon, **self.params)
        adv_imgs = self.batch_attack(attack, images, labels, apgd_flag=True)
        return adv_imgs

    @staticmethod
    def name():
        return "APGD"


class LinfRepeatedAdditiveUniformNoiseAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack, params)


class LinfProjectedGradientDescentAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfProjectedGradientDescentAttack, params)

    @staticmethod
    def name():
        return "PGD"


class LinfNaturalEvolutionStrategyAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfNaturalEvolutionStrategyAttack, params)

    @staticmethod
    def name():
        return "NES"


class CustomLinfProjectedGradientDescentAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.CustomLinfProjectedGradientDescentAttack, params)

    @staticmethod
    def name():
        return "CustomPGD"


class LinfFastGradientAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfFastGradientAttack, params)

    @staticmethod
    def name():
        return "FGSM"


class LinfDeepFoolAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfDeepFoolAttack, params)

    @staticmethod
    def name():
        return "DF"


class LinfCarliniWagnerAttack(Attack):
    def __init__(self, params=None):
        super().__init__()
        self.params = get_param_dict(self)
        if params is not None:
            assert (isinstance(params, dict)), "Parameters has to be type dictionary"
            for key in params:
                if key in self.params:
                    self.params[key] = params[key]

    def run(self, model, images, labels, epsilon):
        fcn = carlini.CarliniLInfMethod
        aux_model = fb.PyTorchModel(model.eval(), bounds=(0, 1))
        # fcn = artatt.evasion.carlini.CarliniLInfMethod
        attmodel = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),  # need to change this a bit
            optimizer=None,
            input_shape=(0,),
            nb_classes=10
        )
        if isinstance(labels, TargetedMisclassification):
            labels = labels.target_classes.raw
            self.params['targeted'] = True
        else:
            self.params['targeted'] = False

        # device = model.device
        device = 'cuda'
        attack = fcn(classifier=attmodel, aux_classifier=aux_model, eps=epsilon, **self.params)
        val = np.clip(images.clone().cpu().numpy(), 0, 1)
        adv_imgs = torch.clamp(torch.as_tensor(attack.generate(x=val, y=labels.cpu().reshape(-1))).to(device), 0, 1)
        return adv_imgs

    def set_param(self, params):
        assert (isinstance(params, dict)), "Parameters has to be type dictionary"
        for key in params:
            if key in self.params:
                self.params[key] = params[key]
        return self

    @staticmethod
    def name():
        return "C&W"


class LinfBrendelBethgeAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.LinfinityBrendelBethgeAttack, params)

    @staticmethod
    def name():
        return "BB"


# L2 attack
class L2AutoAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="L2", params=params)

    def run(self, model, images, labels, epsilon):
        attack = AutoAttack(model, norm=self.norm, eps=epsilon, **self.params)
        adv_imgs = attack.run_standard_evaluation(images, labels, bs=MAX_BATCH)
        return adv_imgs

    @staticmethod
    def name():
        return "AA"


class L2FABAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="L2", params=params)

    def run(self, model, images, labels, epsilon):
        attack = FABAttack(model, norm=self.norm, eps=epsilon, **self.params)
        # adv_imgs = attack.perturb(images, labels)
        adv_imgs = self.batch_attack(attack, images, labels)
        return adv_imgs

    @staticmethod
    def name():
        return "FAB"


class L2SquareAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="L2", params=params)

    def run(self, model, images, labels, epsilon):
        attack = SquareAttack(model, norm=self.norm, eps=epsilon, **self.params)
        # adv_imgs = attack.perturb(images, labels)
        adv_imgs = self.batch_attack(attack, images, labels)
        return adv_imgs

    @staticmethod
    def name():
        return "SQR"


class L2APGDAttack(AutoAttackBase):
    def __init__(self, params=None):
        super().__init__(norm="L2", params=params)

    def run(self, model, images, labels, epsilon):
        attack = APGDAttack(model, norm=self.norm, eps=epsilon, **self.params)
        # adv_imgs = attack.perturb(images, labels)
        adv_imgs = self.batch_attack(attack, images, labels, apgd_flag=True)
        return adv_imgs

    @staticmethod
    def name():
        return "APGD"


class L2RepeatedAdditiveUniformNoiseAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2RepeatedAdditiveUniformNoiseAttack, params)


class L2ProjectedGradientDescentAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2ProjectedGradientDescentAttack, params)

    @staticmethod
    def name():
        return "PGD"


class L2FastGradientAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2FastGradientAttack, params)


class L2NaturalEvolutionStrategyAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2NaturalEvolutionStrategyAttack, params)


class L2DeepFoolAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2DeepFoolAttack, params)


class L2CarliniWagnerAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2CarliniWagnerAttack, params)


class L2DDNAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.DDNAttack, params)


class L2BrendelBethgeAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.L2BrendelBethgeAttack, params)


class BoundaryAttack(FoolboxAttack):
    def __init__(self, params=None):
        super().__init__(fb.attacks.BoundaryAttack, params)


# def L2CarliniWagnerAttack(model, images, criterion, epsilon, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10, max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1):
#     fcn = artatt.CarliniL2Method
#     return artattack(fcn, model, images, criterion, epsilon, confidence=confidence, targeted=targeted, learning_rate=learning_rate,
#                      binary_search_steps=binary_search_steps, max_iter=max_iter, initial_const=initial_const, max_halving=max_halving,
#                      max_doubling=max_doubling, batch_size=batch_size)
