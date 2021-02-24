""" The script process the trials result from progression_comparison.py
Modifications can be made under if __name__ == "__main__"
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from attack.attack_pool import *
from utils.utils import collect_scores_df, collect_param_scores_df
import os.path as osp

Models = ['model2', 'model3', 'model4', 'model5', 'model6', 'model7', 'model8', 'model9', 'model10']

def get_statistics(prefixes, names, suffix=''):
    scores_tpe = []
    scores_random = []
    indices_tpe = []
    indices_random = []
    for prefix in prefixes:
        for name in names:
            fn = osp.join(prefix, name + suffix)
            data = pickle.load(open(fn, 'rb'))
            if type(data) == list:
                data = collect_param_scores_df(data)
            N = int(len(data)/2)
            score_frame_tpe = data.iloc[:N]
            score_frame_random = data.iloc[N:]

            # print(score_frame_tpe)
            tpe_scores = score_frame_tpe["score"]
            max_tpe_scores = [max(tpe_scores[:i]) for i in range(1, N)]
            plt.plot(max_tpe_scores, label='TPE')

            random_scores = score_frame_random["score"]
            max_random_scores = [max(random_scores[:i]) for i in range(1, N)]
            plt.plot(max_random_scores, label='random')

            plt.legend()
            plt.show()
            exit()

            scores_tpe.append(max(score_frame_tpe['score']))
            # indices_tpe.append(score_frame_tpe.idxmax())

            scores_random.append(max(score_frame_random['score']))
            # indices_random.append(score_frame_random.idxmax())

    return np.asarray(scores_tpe), np.asarray(indices_tpe), np.asarray(scores_random), np.asarray(indices_random)


def get_progression(prefixes, names, suffix=''):
    if type(names) != list:
        names = [names]

    overall_scores_1 = []
    overall_scores_2 = []
    for name in names:
        for prefix in prefixes:
            fn = osp.join(prefix, name + suffix)
            data = pickle.load(open(fn, 'rb'))
            if type(data) == list:
                data = collect_param_scores_df(data)
            N = int(len(data) / 2)
            score_frame_1 = data.iloc[:N]
            score_frame_2 = data.iloc[N:]

            # print(score_frame_tpe)
            # if tpe_scores is None:
            scores_1 = score_frame_1["score"]
            scores_2 = score_frame_2["score"]
            # else:
            #     tpe_scores += score_frame_tpe["score"]
            #     random_scores += score_frame_random["score"]
            max_scores_1 = [max(scores_1[:i]) for i in range(1, N)]
            max_scores_2 = [max(scores_2[:i]) for i in range(1, N)]
            overall_scores_1.append(max_scores_1)
            overall_scores_2.append(max_scores_2)
    average_max_scores_1 = np.mean(np.asarray(overall_scores_1), axis=0)
    average_max_scores_2 = np.mean(np.asarray(overall_scores_2), axis=0)
    return average_max_scores_1, average_max_scores_2


def get_improvement_per_model(prefixes, names, mode, suffix=''):
    for name in names:
        average_max_scores_1, average_max_scores_2 = get_progression(prefixes, [name], suffix)
        print((average_max_scores_1[-1] - average_max_scores_2[-1]) / average_max_scores_2[-1])
    return


def get_plot(prefixes, names, mode, suffix=''):
    average_max_scores_1, average_max_scores_2 = get_progression(prefixes, names, suffix)
    # random_scores = random_scores / (len(prefixes) * len(names))
    print("Percentage Improvemeent: ")
    print( (average_max_scores_1[-1] - average_max_scores_2[-1]) /  average_max_scores_2[-1])
    if mode:
        plt.plot(average_max_scores_1, label='TPE')

        plt.plot(average_max_scores_2, label='Random')
    else:
        plt.plot(average_max_scores_1, label='With Loss Formulation')
        plt.plot(average_max_scores_2, label='No Loss Formulation')
    # if len(names) == 1:
    #     plt.title("The best score progression of " + names[0])
    # else:
    #     plt.title("The overall score progression")
    plt.xlabel("Number of Trials")
    plt.ylabel("Score (ASR%)")
    plt.legend(loc='lower right')
    plt.show()
    return


def get_dual_plot(prefixes1, prefixes2, names, suffix=''):
    average_max_scores_1, average_max_scores_2 = get_progression(prefixes1, names, suffix)
    average_max_scores_3, average_max_scores_4 = get_progression(prefixes2, names, suffix)
    plt.plot(average_max_scores_1, label='TPE + Loss Formulation')
    plt.plot(average_max_scores_2, label='Random + Loss Formulation')
    plt.plot(average_max_scores_4, label='TPE + Default Loss')
    plt.xlabel("Number of Trials")
    plt.ylabel("Score (%)")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='--name <name of the file> --dir <directory_name>')
    parser.add_argument('--algo', type=int, required=True, help='flag to indicate search algorithm comparison')
    parser.add_argument('--dual', type=bool, help='enable plotting both tpe random and no loss formulation')
    args = parser.parse_args()

    plot_dual_mode = args.dual

    """
    Change prefixes to the names of the result directory
    """
    prefixes = ['tpe_random1/', 'tpe_random2/', 'tpe_random3/', 'tpe_random4/', 'tpe_random5/']
    prefixes1 = prefixes
    prefixes2 = ['loss_noloss1/', 'loss_noloss2/', 'loss_noloss3/', 'loss_noloss4/', 'loss_noloss5/']  # used in dual plot

    names = Models
    suffix = ''
    # suffix = '_random'

    # get_improvement_per_model(prefixes=prefixes1, names=names, mode=args.algo, suffix=suffix)

    if plot_dual_mode:
        # data contains the dataframe with parameters of the attack
        get_dual_plot(prefixes1=prefixes, prefixes2=prefixes2, names=names, suffix=suffix)
    else:
        get_plot(prefixes=prefixes1, names=names, mode=args.algo, suffix=suffix)
        # tpe_scores, tpe_indices, random_scores, random_indices = get_statistics(prefixes=prefixes, suffix='_random')
        # print(np.mean(tpe_scores))
        # print(np.mean(random_scores))
        # # Result comparison
        # print(np.mean(tpe_scores - random_scores))
        # print(np.std(tpe_scores - random_scores))
        # # print(np.sum(tpe_scores < random_scores-0.1))

