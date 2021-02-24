""" This script runs Autoattack evaluation on detector networks.
Uncomment the the models to run. May need to adjust epsilon and rand version
"""

import eagerpy as ep
from attack.attack_pool import LinfAutoAttack, LinfAPGDAttack, LinfFABAttack, LinfSquareAttack, NoAttack
from attack.attack_search import attack_eval
from attack.attack_base import RepeatAttack, SeqAttack, TargetedAttack

device = 'cuda'
epsilon = 0.03
# epsilon = 8/255
norm = ep.inf
ntest = 10000

# aa = LinfAutoAttack()
aa = SeqAttack([NoAttack(), LinfAPGDAttack({'n_restarts': 5}), TargetedAttack(LinfAPGDAttack({'loss': 'dlr'}), 9),
                TargetedAttack(LinfFABAttack({'n_restarts': 1}), 9), LinfSquareAttack()])


from zoo.cifar_model_pool import *
from zoo.benchmark import *
from zoo.wideresnet import *

model_list = [CCAT(load=True)]
# model_list = [(Model5(load=True))]
# model_list = [TurningWeakness(load=True)]

for f_net in model_list:
    aa.detector = f_net.instance.detector
    attack_eval([aa], f_net, epsilon, norm, ntest, dataset='cifar10', use_vertex_removal=False)
