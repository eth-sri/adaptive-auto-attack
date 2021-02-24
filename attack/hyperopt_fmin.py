# This file is from hyperopt. The only change is to customize the return for easier processing

from __future__ import print_function
from __future__ import absolute_import

import os
import sys

import numpy as np

import hyperopt.base as base
from hyperopt.base import validate_timeout, validate_loss_threshold, Trials
from hyperopt.fmin import generate_trials_to_calculate, FMinIter, space_eval


def hyperopt_fmin(
    fn,
    space,
    algo,
    max_evals=sys.maxsize,
    timeout=None,
    loss_threshold=None,
    trials=None,
    rstate=None,
    allow_trials_fmin=True,
    pass_expr_memo_ctrl=None,
    catch_eval_exceptions=False,
    verbose=True,
    return_argmin=True,
    points_to_evaluate=None,
    max_queue_len=1,
    show_progressbar=True,
    # early_stop_fn=None,
):
    """Minimize a function over a hyperparameter space.

    More realistically: *explore* a function over a hyperparameter space
    according to a given algorithm, allowing up to a certain number of
    function evaluations.  As points are explored, they are accumulated in
    `trials`


    Parameters
    ----------

    fn : callable (trial point -> loss)
        This function will be called with a value generated from `space`
        as the first and possibly only argument.  It can return either
        a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`. Particular
        optimization algorithms may look for other keys as well.  An
        optional sub-dictionary associated with an 'attachments' key will
        be removed by fmin its contents will be available via
        `trials.trial_attachments`. The rest (usually all) of the returned
        dictionary will be stored and available later as some 'result'
        sub-dictionary within `trials.trials`.

    space : hyperopt.pyll.Apply node
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).

    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.

    max_evals : int
        Allow up to this many function evaluations before returning.

    timeout : None or int, default None
        Limits search time by parametrized number of seconds.
        If None, then the search process has no time constraint.

    loss_threshold : None or double, default None
        Limits search time when minimal loss reduced to certain amount.
        If None, then the search process has no constraint on the loss,
        and will stop based on other parameters, e.g. `max_evals`, `timeout`

    trials : None or base.Trials (or subclass)
        Storage for completed, ongoing, and scheduled evaluation points.  If
        None, then a temporary `base.Trials` instance will be created.  If
        a trials object, then that trials object will be affected by
        side-effect of this call.

    rstate : numpy.RandomState, default numpy.random or `$HYPEROPT_FMIN_SEED`
        Each call to `algo` requires a seed value, which should be different
        on each call. This object is used to draw these seeds via `randint`.
        The default rstate is
        `numpy.random.RandomState(int(env['HYPEROPT_FMIN_SEED']))`
        if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
        string, otherwise np.random is used in whatever state it is in.

    verbose : bool
        Print out some information to stdout during search. If False, disable
            progress bar irrespectively of show_progressbar argument

    allow_trials_fmin : bool, default True
        If the `trials` argument

    pass_expr_memo_ctrl : bool, default False
        If set to True, `fn` will be called in a different more low-level
        way: it will receive raw hyperparameters, a partially-populated
        `memo`, and a Ctrl object for communication with this Trials
        object.

    return_argmin : bool, default True
        If set to False, this function returns nothing, which can be useful
        for example if it is expected that `len(trials)` may be zero after
        fmin, and therefore `trials.argmin` would be undefined.

    points_to_evaluate : list, default None
        Only works if trials=None. If points_to_evaluate equals None then the
        trials are evaluated normally. If list of dicts is passed then
        given points are evaluated before optimisation starts, so the overall
        number of optimisation steps is len(points_to_evaluate) + max_evals.
        Elements of this list must be in a form of a dictionary with variable
        names as keys and variable values as dict values. Example
        points_to_evaluate value is [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]

    max_queue_len : integer, default 1
        Sets the queue length generated in the dictionary or trials. Increasing this
        value helps to slightly speed up parallel simulatulations which sometimes lag
        on suggesting a new trial.

    show_progressbar : bool or context manager, default True (or False is verbose is False).
        Show a progressbar. See `hyperopt.progress` for customizing progress reporting.

    early_stop_fn: callable ((result, *args) -> (Boolean, *args)).
        Called after every run with the result of the run and the values returned by the function previously.
        Stop the search if the function return true.
        Default None.

    Returns
    -------

    argmin : dictionary
        If return_argmin is True returns `trials.argmin` which is a dictionary.  Otherwise
        this function  returns the result of `hyperopt.space_eval(space, trails.argmin)` if there
        were succesfull trails. This object shares the same structure as the space passed.
        If there were no succesfull trails, it returns None.
    """
    if rstate is None:
        env_rseed = os.environ.get("HYPEROPT_FMIN_SEED", "")
        if env_rseed:
            rstate = np.random.RandomState(int(env_rseed))
        else:
            rstate = np.random.RandomState()

    validate_timeout(timeout)
    validate_loss_threshold(loss_threshold)

    if allow_trials_fmin and hasattr(trials, "fmin"):
        assert False
    #     return trials.fmin(
    #         fn,
    #         space,
    #         algo=algo,
    #         max_evals=max_evals,
    #         timeout=timeout,
    #         loss_threshold=loss_threshold,
    #         max_queue_len=max_queue_len,
    #         rstate=rstate,
    #         pass_expr_memo_ctrl=pass_expr_memo_ctrl,
    #         verbose=verbose,
    #         catch_eval_exceptions=catch_eval_exceptions,
    #         return_argmin=return_argmin,
    #         show_progressbar=show_progressbar,
    #         early_stop_fn=early_stop_fn,
    #     )

    if trials is None:
        if points_to_evaluate is None:
            trials = base.Trials()
        else:
            assert type(points_to_evaluate) == list
            trials = generate_trials_to_calculate(points_to_evaluate)

    domain = base.Domain(fn, space, pass_expr_memo_ctrl=pass_expr_memo_ctrl)

    rval = FMinIter(
        algo,
        domain,
        trials,
        max_evals=max_evals,
        timeout=timeout,
        loss_threshold=loss_threshold,
        rstate=rstate,
        verbose=verbose,
        max_queue_len=max_queue_len,
        show_progressbar=show_progressbar,
        # early_stop_fn=early_stop_fn,
    )
    rval.catch_eval_exceptions = catch_eval_exceptions

    # next line is where the fmin is actually executed
    rval.exhaust()

    if len(trials.trials) == 0:
        raise Exception(
            "There are no evaluation tasks, cannot return argmin of task losses."
        )
    return trials
