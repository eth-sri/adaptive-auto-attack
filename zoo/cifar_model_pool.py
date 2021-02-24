""" This file defines the benchmark models.

Any benchmark instance needs to inherit BasicInstance.
Method build() is to build a DefenseInstance (instance). Each instance has a model and a detector (default None)
Method load() is to load the model from the saved file.
Method train() can be used to train the model if implemented.

DAGModule is used to define the custom DAG graph for the network. Usually it is used if there are some input
processing stage, post processing stage or some ensemble components. Currently the functionality is very limited.
"""
import torch
import torch.optim as optim
from attack import attack_pool
from zoo.nets.net import *
from utils.dag_module import DAGModule, DistillationWrapper
from utils.benchmark_instance import BenchmarkInstance
import zoo.nets.kWTA as resnet
from defenses.ensemble_vote import LossCombine
from defenses import *
from utils.loader import get_data_tensor, TensorDataset
from utils.defense_instance import DefenseInstance
import os


class CIFARInstance(BenchmarkInstance):
    def __init__(self, name, load=False, device='cuda'):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        super(CIFARInstance, self).__init__(name, load, device, path=os.path.join(dname, '../zoo/saved_instances/cifar'))


class Model1(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model1'
        super(Model1, self).__init__(name, load, device)

    def build(self):
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        classifier = DAGModule([network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=5e-4)
        nb_epoches = 30
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model2(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model2'
        super(Model2, self).__init__(name, load, device)

    def build(self):
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        classifier = DAGModule([network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=5e-4)
        nb_epoches = 40
        attack = attack_pool.LinfProjectedGradientDescentAttack()
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model3(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model3'
        super(Model3, self).__init__(name, load, device)

    def build(self):
        TEdefense = ThermometerEncoding((0, 1), num_space=10)
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=2, linear_size=200, input_channel=30, with_normalization=False)
        classifier = DAGModule([TEdefense, network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 30
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model4(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model4'
        super(Model4, self).__init__(name, load, device)

    def build(self):
        Jpegdefense = JpegCompression((0, 1), 60)
        RSdefense = ReverseSigmoid()
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        classifier = DAGModule([Jpegdefense, network1, RSdefense], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 30
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model5(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model5'
        super(Model5, self).__init__(name, load, device)

    def build(self):
        network = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        network1 = ConvMedBig1(network)
        network2 = ConvMedBig2(network)
        detector1 = SimpleDetector(network1, 200)
        module1 = DAGModule([network1, network2], dependency=[[-1], [0]], output=[1], device=self.device)
        module2 = DAGModule([network1, detector1], dependency=[[-1], [0]], output=[1], device=self.device)
        classifier = module1
        instance = DefenseInstance(model=classifier, detector=module2)
        # return instance
        return instance

    def get_detector_loader(self, instance, loader, epsilons, device='cuda'):
        from attack.attack_pool import LinfFastGradientAttack
        test_attack = LinfFastGradientAttack()
        x_train, y_train = get_data_tensor(loader, 600000, device=device)  # memory can be an issue with large dataset
        nsamples = x_train.shape[0]
        y_train_detector = torch.ones([nsamples])
        fmodel = instance.get_attack_model(bounds=(0, 1))
        adv_img = test_attack(fmodel, x_train, y_train, epsilons)
        nsamples = adv_img.shape[0]
        x_train_detector = torch.cat([x_train, adv_img], dim=0).to(device)
        y_train_detector = torch.cat([y_train_detector, torch.zeros([nsamples])], dim=0).to(device).to(torch.long)
        detector_loader = torch.utils.data.DataLoader(TensorDataset(x_train_detector, y_train_detector),
                                                      batch_size=64, shuffle=True, num_workers=0, drop_last=False)
        return detector_loader

    def train(self, train_loader, eps, save=True):
        # fit model
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 30
        detector_epochs = 5
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()

        # fit detector
        detector_loader = get_detector_loader(self.instance, train_loader, epsilons=eps)
        self.instance.model.moduleList[0].freeze()  # HACK, assuming only one module is before the detector network
        optimizer_detector = optim.Adam(self.instance.detector.moduleList[1].parameters(), lr=0.001)
        self.instance.detector_fit(detector_loader, loss_fcn=train_loss, optimizer=optimizer_detector, nb_epochs=detector_epochs)
        self.instance.eval()

        if save:
            self._save()
        return


class Model6(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model6'
        super(Model6, self).__init__(name, load, device)

    def build(self):
        ITdefense = InputTransformation(degrees=(-30, 30))
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        classifier = DAGModule([ITdefense, network1], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 30
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model7(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model7'
        super(Model7, self).__init__(name, load, device)

    def build(self):
        ITdefense = InputTransformation(degrees=(-30, 30))
        TEdefense = ThermometerEncoding((0, 1), num_space=10)
        Jpegdefense = JpegCompression((0, 1), 30)
        RSdefense = ReverseSigmoid()
        network1 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        network2 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=2, linear_size=200, input_channel=30, with_normalization=False)
        network3 = ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200, input_channel=3, with_normalization=True)
        module1 = DAGModule([TEdefense, network2], device=self.device)
        module2 = DAGModule([Jpegdefense, network1, RSdefense], device=self.device)
        module3 = DAGModule([ITdefense, network3], device=self.device)
        losscomb = LossCombine()
        classifier = DAGModule([module1, module2, module3, losscomb], dependency=[[-1], [-1], [-1], [0, 1, 2]],
                               output=[3], device=self.device)
        instance = DefenseInstance(model=classifier)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 30
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model8(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model8'
        self.target_instance = None
        super(Model8, self).__init__(name, load, device)

    def build(self):
        temp = 100
        network1 = DistillationWrapper(resnet.ResNet18().to(self.device), temp)
        self.target_instance = DefenseInstance(model=network1)
        network2 = DistillationWrapper(ConvMedBig(device=self.device, dataset='cifar10', width1=4, width2=4, width3=4,
                                                  linear_size=200, input_channel=3, with_normalization=True), temp)
        instance = DefenseInstance(model=network2, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.target_instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 25
        attack = None
        kwargs = {}
        self.target_instance.train()
        self.target_instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.target_instance.eval()

        nb_epoches = 25
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        defensive_distillation = DefensiveDistillation(self.target_instance)
        defensive_distillation.fit(self.instance, loader=train_loader, optimizer=optimizer, nb_epochs=nb_epoches, temp=5)

        if save:
            self._save()
        return


class Model9(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model9'
        super(Model9, self).__init__(name, load, device)

    def build(self):
        network = resnet.SparseResNet18(sparsities=[0.1, 0.1, 0.1, 0.1], sparse_func='reg').to(self.device)
        classifier = DAGModule([network], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 40
        attack = None
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.6, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return


class Model10(CIFARInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'model10'
        super(Model10, self).__init__(name, load, device)

    def build(self):
        network = resnet.SparseResNet18(sparsities=[0.1, 0.1, 0.1, 0.1], sparse_func='reg').to(self.device)
        classifier = DAGModule([network], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def train(self, train_loader, eps, save=True):
        train_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.instance.model.train_model.parameters(), lr=0.001)
        nb_epoches = 40
        attack = attack_pool.LinfFastGradientAttack()
        kwargs = {}
        self.instance.train()
        self.instance.advfit(train_loader, loss_fcn=train_loss, optimizer=optimizer, epsilon=eps, attack=attack,
                             nb_epochs=nb_epoches, ratio=0.7, **kwargs)
        self.instance.eval()
        if save:
            self._save()
        return