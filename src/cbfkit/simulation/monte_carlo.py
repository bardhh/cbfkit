"""Functions to aid in conducting Monte Carlo simulations.

Functions
---------
_map_function(args)
    Helper function for multiprocessing.
conduct_monte_carlo(execute, n_trials, n_processes, **kwargs)
    Conducts a Monte Carlo simulation."""

import multiprocessing as mp
from typing import Callable


def _map_function(args):
    """_summary_.

    Args:
        args (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    func, trial_no, kwargs = args
    return func(trial_no, **kwargs)


def conduct_monte_carlo(execute: Callable, n_trials: int, n_processes: int = 1, **kwargs):
    """_summary_.

    Args:
        execute (_type_): _description_
    """
    args_list = [(execute, trial_no, kwargs) for trial_no in range(n_trials)]

    if n_processes > 1:
        # Create a multiprocessing Pool
        with mp.Pool(processes=n_processes) as pool:
            # Process the items in parallel
            results = pool.map(_map_function, args_list)
    else:
        # Execute sequentially
        results = list(map(_map_function, args_list))

    return results
