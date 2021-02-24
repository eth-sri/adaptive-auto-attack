""" This is the main file of A^3.

seq_attack_search: the main function for A^3 the search.
search_g_net: the function for searching network transformation.
"""

from utils.loader import get_loaders
from utils.utils import collect_attack_search_statistics, collect_scores_df
from utils.eval import *
from attack.attack_util import *
from attack.network_transformation import defense_property_test, search_g_net
from attack.hyperopt_fmin import hyperopt_fmin
from hyperopt import tpe, rand, STATUS_OK
import time
import math
import numpy as np
import threading
import pickle
from datetime import datetime
import os
import os.path as osp
from tqdm import tqdm
try:
    import thread
except ImportError:
    import _thread as thread

SEARCH_RESULT_LIST = []
MAX_TIME = 1000000

# FLAGS of the algorithm
CHECK_TIMEOUT = False


def quit_function(fn_name):
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def get_default_timeout_result():
    return [[[-1, -1, -1, -1, 0, 0]]]


def run_single_test(attack, instance, data, epsilon, norm, timeout, to_raise=True, to_print=False):
    """
    Run samples to test if timeout happens. If it happens return True, else return False
    """
    timeout_flag = False

    @exit_after(timeout)
    def run_trial(attack, g_net, f_net, data, epsilon, norm, to_raise=to_raise, to_print=to_print):
        return get_result_from_data(attack, g_net, data, epsilon, norm, to_raise=to_raise, to_print=to_print, eval_net=f_net)
    try:
        run_trial(attack, instance, data, epsilon, norm, to_raise=to_raise, to_print=to_print)
    except KeyboardInterrupt:
        timeout_flag = True
    return timeout_flag


def get_trial_statistics(data, spaces, config, ntrials, nbase_test, points_to_evaluate=None, algos='tpe',
                         tau=4, sample_mult=2, device='cuda', fn='trials', get_plot=False):
    """
    Get the statistics of the tpe and random trials and produce the plot
    """
    config['data'] = get_data_samples(data, nbase_test, offset=0, random=True)

    if type(spaces) != list:
        spaces = [spaces]

    if type(algos) != list:
        algos = [algos]

    print("The number of search spaces is: ", len(spaces))
    print("The number of search algorithms is: ", len(algos))
    # get the scores ready
    for space in spaces:
        for algo in algos:
            hyperopt(space, points_to_evaluate=points_to_evaluate, num_samples=ntrials, algo=algo)

    global SEARCH_RESULT_LIST
    df = collect_scores_df(SEARCH_RESULT_LIST)
    pickle.dump(SEARCH_RESULT_LIST, open(fn, "wb"))
    SEARCH_RESULT_LIST = []

    if get_plot:
        import plotly.express as px
        fig = px.violin(df, y="score", x="attack", box=True, points="all", hover_data=df.columns)
        fig.show()
        return


def run_attack(attack, g_net, f_net, data, epsilon, norm, timeout, to_raise=True, to_print=False):
    """
    Get the result of the attack. If timeout, gets the default timeout result
    """
    @exit_after(timeout)
    def run_trial(attack, g_net, f_net, data, epsilon, norm, to_raise=to_raise, to_print=to_print):
        return get_result_from_data(attack, g_net, data, epsilon, norm, to_raise=to_raise, to_print=to_print, eval_net=f_net)

    try:
        run_result = run_trial(attack, g_net, f_net, data, epsilon, norm, to_raise=to_raise, to_print=to_print)
    except KeyboardInterrupt:
        run_result = np.array(get_default_timeout_result())
    return run_result


def evaluate_attack(arg):
    """
    Main method to evaluate the score of the attack
    """
    score_function = loss_attack_eval_function
    # score_function = time_attack_eval_function
    # score_function = overall_attack_eval_function
    config = {**arg['config'], **arg['param']}  # param overwrites config
    g_net = config["g_net"]
    f_net = config["f_net"]
    epsilon = config["epsilon"]
    norm = config["norm"]
    data = config["data"]
    ntest = data[0].shape[0]
    timeout = config["timelimit"] * ntest

    # construct attack
    attack = attack_from_config(config)
    result = run_attack(attack, g_net, f_net, data, epsilon, norm, timeout, to_raise=True, to_print=False)

    acc, attack_accuracy, disturbance, robustness, time, ce_loss = get_attackwise_average(result)
    score = score_function(result, ntest)
    global SEARCH_RESULT_LIST
    SEARCH_RESULT_LIST.append((arg['param'], result, score,))

    return dict(loss=-score, status=STATUS_OK, model_accuracy=acc, attack_accuracy=attack_accuracy, time=time, disturbance=disturbance, arg=arg)


def sha_step(arglist, data, prev_scores):
    """
    Successive halving step to select the best attacks
    """
    assert(len(prev_scores) == len(arglist))
    result_arglist = []
    scores = []
    for i, arg in enumerate(tqdm(arglist)):
        arg['config']['timelimit'] = MAX_TIME  # no timelimit at this stage
        arg['config']['data'] = data
        result = evaluate_attack(arg)
        result_arglist.append(result['arg'])
        # The score is the average between the previous score and the current score
        scores.append((-result['loss'] + prev_scores[i]) / 2)
    return scores, result_arglist


def hyperopt(space, points_to_evaluate=None, num_samples=10, algo='tpe', logdir='log'):
    """
    Use hyperopt to do meta-learning style optimization
    """
    if algo == 'tpe':
        suggest_algo = tpe.suggest
    elif algo == 'random':
        suggest_algo = rand.suggest
    else:
        assert False, "Not recognized suggest algorithm, expecting 'tpe' or 'random'"

    trials = hyperopt_fmin(evaluate_attack, space, algo=suggest_algo, max_evals=num_samples,
                           points_to_evaluate=points_to_evaluate)
    arglist = []
    scores = []
    for trial in trials.results:
        arglist.append(trial['arg'])
    for loss in trials.losses():
        scores.append(-loss)

    logsave([scores, arglist], logdir, suffix='opt')
    return scores, arglist


def logsave(data, logdir, suffix):
    if not osp.exists(logdir):
        os.mkdir(logdir)
    fn = osp.join(logdir, datetime.now().strftime("%d-%m-%Y-%H:%M:%S") + '-' + suffix + '.pickle')
    fh = open(fn, "wb")
    paramlist = []
    for i in range(len(data[1])):
        paramlist.append(data[1][i]['param'])
    pickle.dump([data[0], paramlist], fh)
    fh.close()
    print("trials info saved to: ", fn)


def hyperopt_sha_search(data, space, config, ntrials, nbase_test, points_to_evaluate=None, algo='tpe',
                        tau=4, sample_mult=2, sha=True, logdir='log', device='cuda'):
    """
    Implementation for SHA
    The first iteration will run hyperopt to search for the best candidates (hyperopt)
    The iterations after will run on more samples to get better evaluations (fmin_SHA)
    tau: the factor of reducing the number of trials
    sample_mult: the multiplier to the sample of testing
    sha: enable SHA
    """

    start_time = time.time()
    config['data'] = get_data_samples(data, nbase_test, offset=0, random=True)

    # initial search using hyperopt TPE
    scores, arglist = hyperopt(space, points_to_evaluate=points_to_evaluate, num_samples=ntrials, algo=algo, logdir=logdir)

    # SHA iteration:
    k = math.ceil(ntrials / float(tau))
    n = nbase_test

    if sha:
        while k > 1:
            ktop, best_k_args, best_k_scores = get_ktop_config(scores, k, arglist)
            # sample with replacement
            data_test = get_data_samples(data, n, offset=0, random=True)
            scores, arglist = sha_step(best_k_args, data_test, prev_scores=best_k_scores)
            logsave([scores, arglist], logdir, suffix='sha')
            # update k if there are too many meaningless attacks. If the attacks are meaningless, then return None
            k = len(ktop)
            if k==0:
                return None  # handle the all time out cases
            print("The top k values occur at positions: ", ktop)
            k = math.ceil(k / tau)
            n *= sample_mult
        assert(k == 1), "only final sample should remain after SHA"

    ktop, best_k_args, best_score = get_ktop_config(scores, 1, arglist)
    for arg in best_k_args:
        config_repr = get_config_string(arg['param'])
        print(config_repr)

    best_attack = attack_from_config({**best_k_args[0]['config'], **best_k_args[0]['param']})
    print("Time used for the search ", time.time() - start_time)
    return best_attack


def adaptive_attack_search(instance, epsilon, norm, ntrials, nbase, space_fn, algo='tpe', tau=4, timelimit=1.,
                           dataset='cifar10'):
    """
    Wrapper for searching only one attack
    """
    return seq_attack_search(instance, epsilon, norm, num_attack=1, ntrials=ntrials, nbase=nbase, search_space=space_fn,
                             algo=algo, tau=tau, timelimit=timelimit, dataset=dataset)


def seq_attack_search(f_net, epsilon, norm, num_attack, ntrials, nbase, search_space, algo='tpe', tau=4, sample_mult=2
                      , timelimit=1., nbase_decrease_rate=1.0, ntrials_decrease_rate=1.0, timelimit_increase_rate=1.0,
                      use_vertex_removal=False, dataset='cifar10', sha=True, eval=False, logdir='log'):
    """
    Main method to do the attack parameter search
    """

    start_time = time.time()

    device = 'cuda'
    eval_result = []
    attack_list = []

    _, _, test_loader, _, _, _ = get_loaders(dataset, 128, 128)
    data = get_data_tensor(test_loader, num=10000, offset=0, device=device)
    n_samples = len(data[0])

    # Network Transformation
    if f_net.instance.detector is None:
        is_detector = False
    else:
        is_detector = True
    defense_property = defense_property_test(f_net.instance.model, test_loader)
    defense_property.detector = is_detector

    g_net = search_g_net(norm, epsilon, f_net, test_loader, defense_property, use_vertex_removal=use_vertex_removal)

    # Define Search Space
    config = dict(g_net=g_net, f_net=f_net, defense_property=defense_property, epsilon=epsilon, norm=norm, repeat=1,
                  timelimit=timelimit)
    space = search_space.define_space(config, defense_property)
    points_to_evaluate = None

    # Remove non accurate samples
    attackUpdater = AttackUpdater(data, f_net.instance, g_net.instance.model, epsilon, is_detector, defense_property.is_random)
    data = attackUpdater.get_acc_img()

    # Run num_attack iterations to get sequence of attacks
    for attack_iter in range(num_attack):
        global SEARCH_RESULT_LIST
        SEARCH_RESULT_LIST = []
        attack = hyperopt_sha_search(data, space, config, ntrials, nbase, points_to_evaluate=points_to_evaluate,
                                     algo=algo, tau=tau, sample_mult=sample_mult, sha=sha, device=device, logdir=logdir)
        if attack is None:
            break  # if all the attacks get time out
        attack_list.append(attack)
        print("The attack distribution: ")
        print(collect_attack_search_statistics(SEARCH_RESULT_LIST), "\n")

        # remove the samples broken by the attack
        data = attackUpdater.get_robust_img([attack])

        # update search parameters
        ntrials = math.ceil(ntrials / ntrials_decrease_rate)
        nbase = math.ceil(nbase / nbase_decrease_rate)
        timelimit = timelimit * timelimit_increase_rate

        # print current progress
        eval_result = attackUpdateEval(attackUpdater, n_samples, start_time)

    final_attack = SeqAttack(attack_list)

    # If eval is True, use the attacks to perform evaluate again
    if eval:
        get_result_from_loader([final_attack], [g_net], test_loader, epsilon, 10000, norm, to_print=True, eval_net=f_net)

    return final_attack, g_net, eval_result


def attack_eval(attack, f_net, epsilon, norm, ntest, use_vertex_removal, dataset='cifar10'):
    """
    Evaluate attack with network network transformation applied
    """
    _, _, test_loader, _, _, _ = get_loaders(dataset, 128, 128)
    defense_property = defense_property_test(f_net.instance.model, test_loader)
    g_net = search_g_net(norm, epsilon, f_net, test_loader, defense_property, use_vertex_removal)
    return get_result_from_loader(attack, g_net, test_loader, epsilon, ntest, norm, eval_net=f_net)
