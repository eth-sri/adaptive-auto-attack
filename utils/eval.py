import numpy as np
from utils.loader import get_data_tensor
import logging


def get_result_from_loader(attack_list, model_list, test_loader, epsilon, ntest, norm, offset=0,
                           device='cpu', to_raise=True, verbose=False, to_print=True, eval_net=None):
    data = get_data_tensor(test_loader, ntest, offset=offset, device=device)
    return get_result_from_data(attack_list, model_list, data, epsilon, norm, to_raise=to_raise,
                                verbose=verbose, to_print=to_print, eval_net=eval_net)


def get_result_from_data(attack_list, model_list, data, epsilon, norm,
                         to_raise=True, verbose=False, to_print=False, eval_net=None):
    """
    model_list: list containing instances of utils.basic_instance.
    eval_net: the network to evaluate adversarial attack on
    result format: [ <raw accuracy>, <accuracy>, <disturbance>, <ce_loss>, <time> ]
    return numpy array of result
    """
    total_result = []
    if not isinstance(attack_list, list):
        attack_list = [attack_list]
    if not isinstance(model_list, list):
        model_list = [model_list]
    for attack in attack_list:
        model_results = []
        for model in model_list:
            if to_raise:
                result = model.eval(data, epsilon, attack, norm=norm, verbose=verbose, eval_instance=eval_net)
            else:
                try:
                    result = model.eval(data, epsilon, attack, norm=norm, verbose=verbose, eval_instance=eval_net)
                except:
                    print("An error occured during testing")
                    result = [-1., -1., -1., -1., -1]
            model_results.append(result)
        total_result.append(model_results)
    result = np.asarray(total_result)
    if to_print:
        print_robustness_time_disturbance_table(result)
    return result


def get_idx_table_from_result(idx, result):
    N = len(result)  # number of attacks
    M = len(result[0])  # number of models
    table = []
    for i in range(N):
        attack_table = []
        for j in range(M):
            attack_table.append(result[i][j][idx])
        table.append(attack_table)
    return table


def logprint(s):
    ns = str(s)
    print(ns)
    logging.info(ns)


def print_table(result):
    for i in result:
        logprint(i)


def print_robustness_time_disturbance_table(result):
    logprint("Accuracy table(%): ")
    print_table(get_idx_table_from_result(0, result))
    logprint("Attack accuracy table(%): ")
    print_table(get_idx_table_from_result(1, result))
    logprint("Time used (s): ")
    print_table(get_idx_table_from_result(5, result))
    logprint("Robustness of the network (%): ")
    print_table(get_idx_table_from_result(3, result))
    logprint("Avg disturbance: ")
    print_table(get_idx_table_from_result(2, result))
    logprint("Avg ce loss: ")
    print_table(get_idx_table_from_result(4, result))


def get_attackwise_average(result):
    """
    This function shows the meaning of indices in the result list
    """
    acc = np.mean(np.array(get_idx_table_from_result(0, result)))
    attack_accuracy = np.mean(np.array(get_idx_table_from_result(1, result)))
    disturbance = np.mean(np.array(get_idx_table_from_result(2, result)))
    robustness = np.mean(np.array(get_idx_table_from_result(3, result)))
    loss = np.mean(np.array(get_idx_table_from_result(4, result)))
    time = np.mean(np.array(get_idx_table_from_result(5, result)))
    return [acc, attack_accuracy, disturbance, robustness, time, loss]


def time_attack_eval_function(result, ntest):
    assert(len(result) == 1), "Only one attack is allowed"
    attack_accuracy = np.mean(np.array(get_idx_table_from_result(1, result)))
    time = np.mean(np.array(get_idx_table_from_result(5, result)))
    return float(attack_accuracy - time/(ntest*100))


def loss_attack_eval_function(result, ntest):
    assert(len(result) == 1), "Only one attack is allowed"
    attack_accuracy = np.mean(np.array(get_idx_table_from_result(1, result)))
    ce_loss = np.mean(np.array(get_idx_table_from_result(4, result)))
    return float(attack_accuracy + ce_loss/100)


def overall_attack_eval_function(result, ntest):
    assert(len(result) == 1), "Only one attack is allowed"
    attack_accuracy = np.mean(np.array(get_idx_table_from_result(1, result)))
    time = np.mean(np.array(get_idx_table_from_result(5, result)))
    ce_loss = np.mean(np.array(get_idx_table_from_result(4, result)))
    return float(attack_accuracy + ce_loss/100 - time/(ntest*500))


def score_eval_function(result, ntest):
    assert(len(result) == 1), "Only one attack is allowed"
    attack_accuracy = np.mean(np.array(get_idx_table_from_result(1, result)))
    return float(attack_accuracy)
