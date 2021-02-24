from zoo.cifar_model_pool import *

from utils.loader import get_loaders
from utils.eval import *
from utils.utils import collect_attack_search_statistics, collect_scores, get_progression, collect_scores_df
from attack.attack_util import *
from attack.search_space import *
from attack.network_transformation import defense_property_test
from attack.attack_search import search_g_net, get_trial_statistics
import argparse


def density_plot(f_net, epsilon, norm, ntrials, nbase, search_space, algo='tpe', tau=4, sample_mult=2
                           , timelimit=1., use_vertex_removal=False, dataset='cifar10', fn='trials'):
    """
    Main method to do the attack parameter search
    """
    device = 'cuda'
    _, _, test_loader, _, _, _ = get_loaders(dataset, 128, 128)

    # model and threat model
    rawdata = get_data_tensor(test_loader, num=10000, offset=0, device=device)

    # 1. get network property
    defense_property = defense_property_test(f_net.instance.model, test_loader)

    # 2. network processing strategy
    g_net = search_g_net(norm, epsilon, f_net, test_loader, defense_property, use_vertex_removal=use_vertex_removal)

    filter_attack_list = [NoAttack()]
    data = get_robust_img(filter_attack_list, f_net.instance, g_net.instance.model, rawdata, epsilon, norm)

    global SEARCH_RESULT_LIST
    SEARCH_RESULT_LIST = []

    # 3. define search space
    config = dict(g_net=g_net, f_net=f_net, defense_property=defense_property, epsilon=epsilon, norm=norm, repeat=1, timelimit=timelimit)
    points_to_evaluate = None

    algos = algo
    spaces = search_space.define_space(config, defense_property)

    # 4. start search in a sequential manner
    get_trial_statistics(data, spaces, config, ntrials, nbase, points_to_evaluate=points_to_evaluate,
                         algos=algos, tau=tau, sample_mult=sample_mult, device=device, fn=fn, get_plot=True)
    return None


if __name__ == "__main__":
    import os
    import os.path as osp

    parser = argparse.ArgumentParser(description='--dir <name of the directory>')
    # parser.add_argument('--algo', type=int, required=True, help='flag to enable search algorithm comparison')
    parser.add_argument('--dir', type=str, required=True, help='name of the output directory')
    args = parser.parse_args()

    # # Change the output directory here
    dir = args.dir
    # algoFlag = args.algo

    device = 'cuda'
    epsilon = 4 / 255
    norm = ep.inf

    ntrials = 100
    nbase = 200

    search_space = Linf_search_space({"APGD": True})
    algo = 'tpe'

    # Models = [(Model1, 0.3, 'model1'), (Model2, 0.3, 'model2'), (Model3, 0.3, 'model3'), (Model4, 0.3, 'model4'),
    #           (Model5, 0.3, 'model5'), (Model6, 0.5, 'model6'), (Model7, 0.5, 'model7'), (Model8, 0.3, 'model8'),
    #           (Model9, 1, 'model9'), (Model10, 1, 'model10')]

    Models = [(Model2, 0.5, 'model2')]

    if not osp.exists(dir):
        os.mkdir(dir)

    for model, timelimit, name in Models:
        f_net = model(load=True, device=device)
        fn = osp.join(dir, name)
        density_plot(f_net, epsilon, norm, ntrials=ntrials, nbase=nbase,
                    search_space=search_space, algo='tpe', tau=4, timelimit=timelimit, fn=fn)
        print("\n\n\n")