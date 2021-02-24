""" This file defines the benchmark models.

Any benchmark instance needs to inherit BasicInstance.
Method build() is to build a DefenseInstance (instance). Each instance has a model and a detector (default None)
Method load() is to load the model from the saved file.
Method train() can be used to train the model if implemented.  
"""
import torch
import torch.nn as nn
from utils.dag_module import DAGModule
from utils.benchmark_instance import BenchmarkInstance
from utils.defense_instance import DefenseInstance
from defenses import *
import os
import os.path as osp
# from robustbench.utils import load_model

ROTATION_DEGREES = 30
COMPRESSION_QUALITY = 30


class CommonInstance(BenchmarkInstance):
    def __init__(self, name, load=False, extend='net', strict=True, device='cuda'):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        self.extend = extend
        self.strict = strict
        super(CommonInstance, self).__init__(name, load, device, path=os.path.join(dname, 'saved_instances'))

    def load(self):
        if self.extend:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"))[self.extend]
        else:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"))
        self.instance.model.load_state_dict(dic, strict=self.strict)
        self.instance.eval()

    def train(self, train_loader, eps, save=True):
        pass

    def build(self):
        raise NotImplementedError


class BenchmarkInstanceCustom(BenchmarkInstance):
    def __init__(self, name, load=False, extend='net', strict=True, device='cuda'):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        self.extend = extend
        self.strict = strict
        self.path = os.path.join(dname, 'saved_instances')
        self.name = name
        self.device = device
        self.network = self.network_build()
        if load:
            self.load()
        self.instance = self.build()
        self.instance.eval()  # important to set to evaluation mode
        assert (isinstance(self.instance, DefenseInstance)), "Model has to be a defense instance"

    def load(self):
        if self.extend:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"))[self.extend]
        else:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"))
        self.network.load_state_dict(dic, strict=self.strict)
        self.network.eval()

    def train(self, train_loader, eps, save=True):
        pass

    def build(self):
        raise NotImplementedError


class AWP_RST_wrn28_10(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'RST-AWP_cifar10_linf_wrn28-10'
        extend = 'state_dict'
        super(AWP_RST_wrn28_10, self).__init__(name, load, strict=False, extend=extend, device=device)

    def build(self):
        from zoo.wideresnet import WideResNet
        classifier = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
        classifier = nn.DataParallel(classifier).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class AWP_RST_wrn28_10_compression(BenchmarkInstanceCustom):
    def __init__(self, load=False, device='cuda'):
        name = 'RST-AWP_cifar10_linf_wrn28-10'
        extend = 'state_dict'
        super(AWP_RST_wrn28_10_compression, self).__init__(name, load, strict=False, extend=extend, device=device)

    def build(self):
        Jpegdefense = JpegCompression((0, 1), COMPRESSION_QUALITY)
        RSdefense = ReverseSigmoid()
        classifier = DAGModule([Jpegdefense, self.network, RSdefense], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def network_build(self):
        from zoo.wideresnet import WideResNet
        network = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
        network = nn.DataParallel(network).cuda()
        return network


class AWP_RST_wrn28_10_transformation(BenchmarkInstanceCustom):
    def __init__(self, load=False, device='cuda'):
        name = 'RST-AWP_cifar10_linf_wrn28-10'
        extend = 'state_dict'
        super(AWP_RST_wrn28_10_transformation, self).__init__(name, load, strict=False, extend=extend, device=device)

    def build(self):
        ITdefense = InputTransformation(degrees=(-ROTATION_DEGREES, ROTATION_DEGREES))
        classifier = DAGModule([ITdefense, self.network], device=self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def network_build(self):
        from zoo.wideresnet import WideResNet
        network = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
        network = nn.DataParallel(network).cuda()
        return network


class AWP_TRADES_wrn34_10(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'TRADES-AWP_cifar10_linf_wrn34-10'
        extend = ''
        super(AWP_TRADES_wrn34_10, self).__init__(name, load, extend=extend, device=device)

    def build(self):
        from zoo.wideresnet import WideResNet
        classifier = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
        classifier = nn.DataParallel(classifier).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class AWP_Retraining_wrn34_10(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'Retraining-AWP_cifar10_linf_wrn34-10'
        extend = ''
        super(AWP_Retraining_wrn34_10, self).__init__(name, load, strict=False, extend=extend, device=device)

    def build(self):
        from zoo.wideresnet import WideResNet
        classifier = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
        classifier = nn.DataParallel(classifier).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class FeatureScatter(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'FeatureScatter'
        extend = 'net'
        super(FeatureScatter, self).__init__(name, load, extend=extend, device=device)

    def build(self):
        from zoo.wideresnet import WideResNetNormalize
        from zoo.saved_instances.FeatureScatter.attack_methods import Attack_FeaScatter
        config_feature_scatter = {
            'train': True,
            'epsilon': 8.0 / 255 * 2,
            'num_steps': 1,
            'step_size': 8.0 / 255 * 2,
            'random_start': True,
            'ls_factor': 0.5,
        }
        basic_net = WideResNetNormalize(28, 10, widen_factor=10, dropRate=0.0)
        net = Attack_FeaScatter(basic_net, config_feature_scatter)
        net = torch.nn.DataParallel(net)
        instance = DefenseInstance(model=net, detector=None)
        return instance


class JEM(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'JEM'
        extend = "model_state_dict"
        super(JEM, self).__init__(name, load, strict=True, extend=extend, device=device)

    def build(self):
        from zoo.saved_instances.JEM.CCF_model import CCF
        # nsteps = 10
        classifier = CCF(28, 10, None, dropout_rate=0, n_classes=10).to(self.device)
        # classifier = WrapperModel(f, nsteps).to(self.device)
        # classifier = gradient_attack_wrapper(f)
        # classifier = f.to(self.device)
        # classifier = nn.DataParallel(classifier).to(self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class JEM_compression(BenchmarkInstanceCustom):
    def __init__(self, load=False, device='cuda'):
        name = 'JEM'
        extend = "model_state_dict"
        super(JEM_compression, self).__init__(name, load, strict=True, extend=extend, device=device)

    def build(self):
        from zoo.saved_instances.JEM.CCF_model import CCF
        Jpegdefense = JpegCompression((0, 1), COMPRESSION_QUALITY)
        RSdefense = ReverseSigmoid()
        classifier = DAGModule([Jpegdefense, self.network, RSdefense], device=self.device)
        # classifier = WrapperModel(f, nsteps).to(self.device)
        # classifier = gradient_attack_wrapper(f)
        # classifier = f.to(self.device)
        # classifier = nn.DataParallel(classifier).to(self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def network_build(self):
        from zoo.saved_instances.JEM.CCF_model import CCF
        network = CCF(28, 10, None, dropout_rate=0, n_classes=10).to(self.device)
        return network


class JEM_transformation(BenchmarkInstanceCustom):
    def __init__(self, load=False, device='cuda'):
        name = 'JEM'
        extend = "model_state_dict"
        super(JEM_transformation, self).__init__(name, load, strict=True, extend=extend, device=device)

    def build(self):
        from zoo.saved_instances.JEM.CCF_model import CCF
        # nsteps = 10
        ITdefense = InputTransformation(degrees=(-ROTATION_DEGREES, ROTATION_DEGREES))
        classifier = DAGModule([ITdefense, self.network], device=self.device)
        # classifier = WrapperModel(f, nsteps).to(self.device)
        # classifier = gradient_attack_wrapper(f)
        # classifier = f.to(self.device)
        # classifier = nn.DataParallel(classifier).to(self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance

    def network_build(self):
        from zoo.saved_instances.JEM.CCF_model import CCF
        network = CCF(28, 10, None, dropout_rate=0, n_classes=10).to(self.device)
        return network


class kWTA(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'kWTA_0p1_adv'
        extend = ''
        super(kWTA, self).__init__(name, load, strict=True, extend=extend, device=device)

    def build(self):
        from zoo.saved_instances.kWTA_0p1_adv import resnet
        classifier = resnet.SparseResNet18(sparsities=[0.1,0.1,0.1,0.1], sparse_func='vol').to(self.device)
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class EnResNet(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'EnResNet'
        extend = 'net'
        super(EnResNet, self).__init__(name, load, strict=True, extend=extend, device=device)

    def load(self):
        self.instance.model = torch.load(osp.join(self.path, self.name, "classifier.model"))[self.extend]
        self.instance.eval()

    def build(self):
        import zoo.saved_instances.EnResNet.enresnet_wideresnet as EWRes
        classifier = EWRes.WideResNet(noise_coef=0.0).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class MART(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'MART'
        extend = ''
        super(MART, self).__init__(name, load, strict=False, extend=extend, device=device)

    def build(self):
        from zoo.wideresnet import WideResNet
        classifier = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
        classifier = nn.DataParallel(classifier).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class Hydra(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'Hydra'
        extend = 'state_dict'
        super(Hydra, self).__init__(name, load, strict=False, extend=extend, device=device)

    def load(self):
        if self.extend:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"), map_location=self.device)[self.extend]
        else:
            dic = torch.load(osp.join(self.path, self.name, "classifier.model"), map_location=self.device)
        self.instance.model.load_state_dict(dic, strict=self.strict)
        if self.instance.detector is not None:
            self.instance.detector.load(osp.join(self.path, self.name, "detector.model"))
        self.instance.eval()

    def build(self):
        from zoo.wideresnet import WideResNet
        classifier = WideResNet(28, 10, widen_factor=10, dropRate=0.0)
        classifier = nn.DataParallel(classifier).cuda()
        instance = DefenseInstance(model=classifier, detector=None)
        return instance


class TurningWeakness(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'TurningWeakness'
        extend = 'state_dict'
        super(TurningWeakness, self).__init__(name, load, strict=True, extend=extend, device=device)

    def build(self):
        from zoo.saved_instances.TurningWeakness.vgg19 import vgg19, TurningWeaknessDetector
        classifier = vgg19()
        classifier.features = torch.nn.DataParallel(classifier.features)
        classifier.to(self.device)
        detector = TurningWeaknessDetector(classifier=classifier)
        instance = DefenseInstance(model=classifier, detector=detector)
        return instance


class OddsAreOdd(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'Odds'
        extend = ''
        super(OddsAreOdd, self).__init__(name, load=False, strict=False, extend=extend, device=device)

    def build(self):
        import zoo.saved_instances.Odds.cifar_model as cifar_model
        import zoo.saved_instances.Odds.robustify as robustify
        classifier = cifar_model.cifar10(128, pretrained=True, map_location=None, trained_adv=False).to(self.device)
        detector = robustify.OddsAreOdd(classifier=classifier)
        instance = DefenseInstance(model=classifier, detector=detector)
        return instance


class CCAT(CommonInstance):
    def __init__(self, load=False, device='cuda'):
        name = 'CCAT'
        extend = ''
        super(CCAT, self).__init__(name, load=False, strict=False, extend=extend, device=device)

    def build(self):
        import zoo.saved_instances.CCAT.common.state as state
        from zoo.saved_instances.CCAT.detector import CCATDetector
        import sys
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.insert(1, dir_path + '/saved_instances/CCAT')
        import zoo.saved_instances.CCAT.models as models
        model_file = osp.join(self.path, self.name, "classifier.model")
        state = state.State.load(model_file)
        classifier = state.model.to(self.device)
        detector = CCATDetector(classifier, 0.4)
        instance = DefenseInstance(model=classifier, detector=detector)  # , detector=detector)
        return instance


# class RobustBenchmark(BasicInstance):
#     def __init__(self, name, load=False, extend='net', strict=True, device='cuda'):
#         abspath = os.path.abspath(__file__)
#         dname = os.path.dirname(abspath)
#         self.extend = extend
#         self.strict = strict
#         super(RobustBenchmark, self).__init__(name, False, device, path=os.path.join(dname, 'saved_instances'))
#
#     def build(self):
#         model = load_model(model_name=self.name, norm='Linf')
#         instance = DefenseInstance(model=model.to(self.device), detector=None)
#         return instance
#

# class Pang2020Boosting(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Pang2020Boosting'
#         super(Pang2020Boosting, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Zhang2020Attacks(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Zhang2020Attacks'
#         super(Zhang2020Attacks, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Rice2020Overfitting(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Rice2020Overfitting'
#         super(Rice2020Overfitting, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Huang2020Self(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Huang2020Self'
#         super(Huang2020Self, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Carmon2019Unlabeled(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Carmon2019Unlabeled'
#         super(Carmon2019Unlabeled, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Chen2020Adversarial(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Chen2020Adversarial'
#         super(Chen2020Adversarial, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Engstrom2019Robustness(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Engstrom2019Robustness'
#         super(Engstrom2019Robustness, self).__init__(name, load=False, strict=False, device=device)
#
#
# class Ding2020MMA(RobustBenchmark):
#     def __init__(self, load=False, device='cuda'):
#         name = 'Ding2020MMA'
#         super(Ding2020MMA, self).__init__(name, load=False, strict=False, device=device)

