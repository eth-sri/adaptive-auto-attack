""" This script runs A^3 to search models in group B from the paper
Model code: model name
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
B23: TurningWeakness
"""

import eagerpy as ep
from utils.utils import print_eval_result
from attack.attack_search import seq_attack_search
from attack.search_space import Linf_search_space
from zoo.cifar_model_pool import *
from zoo.benchmark import *
from zoo.wideresnet import *

device = 'cuda'
epsilon = 8/255
norm = ep.inf
logdir = 'logB'

# Hyperparameters for search
num_attack = 3
ntrials = 64
nbase = 100
search_space = Linf_search_space({"APGD": True})
tl = 3  # timelimit

Models = [(AWP_RST_wrn28_10, tl), (AWP_TRADES_wrn34_10, tl), (FeatureScatter, tl), (JEM, tl)]
# Models = [(kWTA, tl), (EnResNet, tl), (MART, tl), (Hydra, tl)]
# Models = [(AWP_RST_wrn28_10_compression, tl), (JEM_compression, tl),
#           (AWP_RST_wrn28_10_transformation, tl), (JEM_transformation, tl)]
# Models = [(TurningWeakness, tl)]


results = []
for model, timelimit in Models:
    f_net = model(load=True, device=device)
    _, _, eval_result = seq_attack_search(f_net, epsilon, norm, num_attack=num_attack, ntrials=ntrials, nbase=nbase,
                      search_space=search_space, algo='tpe', tau=4, timelimit=timelimit, nbase_decrease_rate=1,
                      ntrials_decrease_rate=1, timelimit_increase_rate=1, use_vertex_removal=True, sha=True, eval=False, logdir=logdir)
    print("\n\n\n")
    results.append(eval_result)

# for i, result in enumerate(results):
#     print("The ", i, "th network result is: ")
#     print_eval_result(result)
