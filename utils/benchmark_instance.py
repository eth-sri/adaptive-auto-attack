from zoo.nets.net import *
from utils.utils import get_acc_robust_disturbance
from utils.defense_instance import DefenseInstance
import os.path as osp
import time
import eagerpy as ep
from typing_extensions import final


class BenchmarkInstance:
    """
    This class wraps around the DefenseInstance model, providing loading and evaluation
    """
    def __init__(self, name, load, device, path='saved_instances'):
        self.path = path
        self.name = name
        self.device = device
        self.instance = self.build()
        assert(isinstance(self.instance, DefenseInstance)), "Model has to be a defense instance"
        if load:
            # self.instance.model.load(osp.join(self.path, self.name, "classifier.model"))
            self.load()

    def load(self):
        self.instance.model.load_state_dict(torch.load(osp.join(self.path, self.name, "classifier.model")))
        if self.instance.detector is not None:
            self.instance.detector.load(osp.join(self.path, self.name, "detector.model"))
        self.instance.eval()

    def build(self):
        raise NotImplementedError

    @final
    def eval(self, data, eps, attack, bounds=(0, 1), norm=ep.inf, verbose=False, eval_instance=None):
        """
        This is the main method used to evaluate the benchmark instance
        eval_net is the DefenseInstance for evaluation, which can be different from the attack model
        """
        assert(self.instance is not None)
        x_test, y_test = data
        images = torch.as_tensor(x_test).to(self.device)
        labels = y_test.to(self.device)
        model = self.instance.get_attack_model(bounds=bounds)
        start = time.time()
        adv_img = attack(model, images, labels, eps)
        time_spend = time.time() - start
        if eval_instance is None:
            result = get_acc_robust_disturbance(self.instance, images, adv_img, y_test, norm, verbose)
        else:
            result = get_acc_robust_disturbance(eval_instance.instance, images, adv_img, y_test, norm, verbose)
        result.append(time_spend)
        if verbose:
            print("Test result for", self.name, ":")
            print("The time used for attack: {:.4f} s".format(time_spend))
            print()
        return result

    def _save(self):
        self.instance.model.save("classifier", osp.join(self.path, self.name))
        if self.instance.detector is not None:
            self.instance.detector.save("detector", osp.join(self.path, self.name))

    def exists(self):
        return osp.exists(osp.join(self.path, self.name, "classifier.model"))