import eagerpy as ep
from attack.attack_pool import LinfAutoAttack
from attack.attack_search import attack_eval
from attack.attack_base import RepeatAttack, SeqAttack, TargetedAttack

"""
This script runs Autoattack on the networks. For defenses with detectors, run aa_detector_eval.py instead.  
Uncomment the the models to run. May need to adjust epsilon and rand version

Model code: model name
A1: CCAT  -> detector
A2: Model2
A3: Model3
A4: Model4
A5: Model5  -> detector
A6: Model6
A7: Model7
A8: Model8
A9: Model9
A10: Model10
B11: AWP_RST_wrn28_10
B12: AWP_TRADES_wrn34_10
B13: FeatureScatter
B14: JEM
B15: kWTA
B16: EnResNet
B17: MART
B18: Hydra
B19: AWP_RST_wrn28_10_compression
B20: JEM_compression
B21: AWP_RST_wrn28_10_transformation
B22: JEM_transformation
B23: TurningWeakness  -> detector
"""

device = 'cuda'
# epsilon = 4/255
epsilon = 8/255
norm = ep.inf
ntest = 10000

aa = LinfAutoAttack()
# aa = LinfAutoAttack({'version': 'rand'})

from zoo.cifar_model_pool import *
from zoo.benchmark import *
from zoo.wideresnet import *

# model_list = [CCAT(load=True)]
# model_list = [(Model1(load=True)), (Model2(load=True)), (Model3(load=True)), (Model4(load=True)), (Model5(load=True)),
#         (Model6(load=True)), (Model7(load=True)), (Model8(load=True)), (Model9(load=True)), (Model10(load=True))]
model_list = [AWP_RST_wrn28_10(load=True), AWP_TRADES_wrn34_10(load=True), FeatureScatter(load=True), JEM(load=True)]
# model_list = [EnResNet(load=True), kWTA(load=True), MART(load=True), Hydra(load=True)]
# model_list = [AWP_RST_wrn28_10_compression(load=True), JEM_compression(load=True)]
# model_list = [AWP_RST_wrn28_10_transformation(load=True), JEM_transformation(load=True)]
# model_list = [TurningWeakness(load=True)]

# for f_net in model_list:
#     attack_eval([aa], f_net, epsilon, norm, ntest, dataset='cifar10', use_vertex_removal=False)

for f_net in model_list:
    aa.detector = f_net.instance.detector
    attack_eval([aa], f_net, epsilon, norm, ntest, dataset='cifar10', use_vertex_removal=False)
