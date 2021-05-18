""" This script runs A^3 to search models in group B from the paper
Model code: model name
C15: CCAT
C16: FeatureScatter
C17: JEM
C18: kWTA
C19: EnResNet
C20: AWP_RST_wrn28_10_compression
C21: JEM_compression
C22: AWP_RST_wrn28_10_transformation
C23: JEM_transformation
C24: TurningWeakness
"""

import eagerpy as ep
from utils.utils import print_eval_result
from attack.attack_search import seq_attack_search
from attack.search_space import Linf_search_space
from zoo.cifar_model_pool import *
from zoo.benchmark import *
from zoo.wideresnet import *

device = 'cuda'
epsilon = 8/255  # 0.03 for CCAT
norm = ep.inf
logdir = 'logB'

# Hyperparameters for search
num_attack = 3
ntrials = 64
nbase = 100
search_space = Linf_search_space({"APGD": True})
tl = 3  # timelimit

# Models = [(CCAT, tl)]
# Models = [(FeatureScatter, tl), (JEM, tl), (kWTA, tl), (EnResNet, tl)]
# Models = [(AWP_RST_wrn28_10_compression, tl), (JEM_compression, tl),
#           (AWP_RST_wrn28_10_transformation, tl), (JEM_transformation, tl)]
# Models = [(TurningWeakness, tl)]
Models = [(AWP_RST_wrn28_10, tl), (AWP_TRADES_wrn34_10, tl), (MART, tl), (Hydra, tl)]

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
