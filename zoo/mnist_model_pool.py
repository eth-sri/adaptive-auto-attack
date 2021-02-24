import torch
import torch.nn as nn
import torch.optim as optim
import foolbox as fb
from attack import attack_pool
from nets.net import *
from utils.dag_module import DAGModule
from utils.benchmark_instance import BenchmarkInstance
from utils.utils import get_acc_robust_disturbance
from defenses.diversity_ensemble import DiversityEnsembleLoss
from defenses.ensemble_vote import LossCombine
from defenses import *
from utils.defense_instance import DefenseInstance
from utils.loader import get_data_tensor, get_detector_loader
import os


class MNISTInstance(BenchmarkInstance):
    def __init__(self, name, load=False, device='cuda'):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        super(MNISTInstance, self).__init__(name, load, device, path=os.path.join(dname, '../zoo/saved_instances/mnist'))


class Model1(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model1'
        super(Model1, self).__init__(name, load, device)

    def build(self):
        network1 = SimpleNet(in_ch=1, out_ch=10)
        classifier = DAGModule([network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model2(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model2'
        super(Model2, self).__init__(name, load, device)

    def build(self):
        network1 = SimpleNet(in_ch=1, out_ch=10)
        classifier = DAGModule([network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = attack_pool.LinfProjectedGradientDescentAttack()
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model3(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model3'
        super(Model3, self).__init__(name, load, device)

    def build(self):
        TEdefense = ThermometerEncoding((0, 1), num_space=10)
        network1 = SimpleNet(in_ch=10, out_ch=10)
        classifier = DAGModule([TEdefense, network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model4(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model4'
        super(Model4, self).__init__(name, load, device)

    def build(self):
        Jpegdefense = JpegCompression((0, 1), 30)
        RSdefense = ReverseSigmoid()
        network1 = SimpleNet(in_ch=1, out_ch=10)
        classifier = DAGModule([Jpegdefense, network1, RSdefense], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model5(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model5'
        super(Model5, self).__init__(name, load, device)

    def build(self):
        network1 = SimpleNet1(in_ch=1, out_ch=10)
        network2 = SimpleNet2(in_ch=1, out_ch=10)
        detector1 = SimpleDetector(network1)
        module1 = DAGModule([network1, network2], dependency=[[-1], [0]], output=[1], device=self.device)
        module2 = DAGModule([network1, detector1], dependency=[[-1], [0]], output=[1], device=self.device)
        instance = DefenseInstance(model=module1, detector=module2)
        return instance

    def train(self, train_loader, eps, save=True):
        # fit model
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)

        # fit detector
        detector_loader = get_detector_loader(self.instance, train_loader, epsilons=eps)
        self.instance.model.moduleList[0].freeze()
        optimizer_detector = optim.Adam(self.instance.detector.moduleList[1].parameters(), lr=0.01)
        self.instance.detector_fit(detector_loader, loss_fcn=train_loss, optimizer=optimizer_detector, nb_epochs=2)
        self.instance.eval()

        if save:
            self._save()
        return


class Model6(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model6'
        super(Model6, self).__init__(name, load, device)

    def build(self):
        ITdefense = InputTransformation(degrees=(-30, 30))
        TEdefense = ThermometerEncoding((0, 1), num_space=10)
        Jpegdefense = JpegCompression((0, 1), 30)
        RSdefense = ReverseSigmoid()
        network1 = SimpleNet(in_ch=1, out_ch=10)
        network2 = SimpleNet(in_ch=10, out_ch=10)
        network3 = SimpleNet(in_ch=1, out_ch=10)
        module1 = DAGModule([TEdefense, network2], device=self.device)
        module2 = DAGModule([Jpegdefense, network1, RSdefense], device=self.device)
        module3 = DAGModule([ITdefense, network3], device=self.device)
        losscomb = LossCombine()
        classifier = DAGModule([module1, module2, module3, losscomb], dependency=[[-1], [-1], [-1], [0, 1, 2]],
                             output=[3], device=self.device)
        instance = DefenseInstance(model=classifier)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = DiversityEnsembleLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model7(MNISTInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model7'
        super(Model7, self).__init__(name, load, device)

    def build(self):
        ITdefense = InputTransformation(degrees=(-30, 30))
        # Varmin = VarianceMinimization(prob=0.3, norm=2)
        # Jpegdefense = JpegCompression((0, 1), 30)
        # RSdefense = ReverseSigmoid()
        network1 = SimpleNet(in_ch=1, out_ch=10)
        classifier = DAGModule([ITdefense, network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.01)
        nb_epoches = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


