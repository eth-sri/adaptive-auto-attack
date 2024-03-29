""" This script runs A^3 on extended search space for block A models
Model code: model name
A1: Model2
A2: Model3
A3: Model4
A4: Model5
A5: Model6
A6: Model7
A7: Model8
A8: Model9

A9 is excluded because the variance is too large across runs due to the obfuscated nature of the defense
"""

import eagerpy as ep
from utils.utils import print_eval_result
from attack.attack_search import seq_attack_search
from attack.search_space import Linf_extended_search_space
from zoo.cifar_model_pool import *

device = 'cuda'
epsilon = 4 / 255
norm = ep.inf
logdir = 'logA'

# Hyperparameters for search
num_attack = 3
ntrials = 64
nbase = 100
search_space = Linf_extended_search_space({"APGD": True})

Models = [(Model2, 0.5), (Model3, 0.5), (Model4, 0.5), (Model5, 0.5), (Model6, 0.5), (Model7, 0.5),
          (Model8, 0.5), (Model9, 1)]

results = []
for model, timelimit in Models:
    f_net = model(load=True, device=device)
    _, _, eval_result = seq_attack_search(f_net, epsilon, norm, num_attack=num_attack, ntrials=ntrials, nbase=nbase,
                      search_space=search_space, algo='tpe', tau=4, timelimit=timelimit, nbase_decrease_rate=1,
                      ntrials_decrease_rate=1, timelimit_increase_rate=1, use_vertex_removal=True, sha=True, eval=False, logdir=logdir)
    print("\n\n\n")
    results.append(eval_result)

# for i, result in enumerate(results):
#     print("The ", i+1, "th network result is: ")
#     print_eval_result(result)
