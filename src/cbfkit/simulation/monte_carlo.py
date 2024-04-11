"""
title
================

Description of module

Functions
---------
-function(a): description

Notes
-----
Various notes here

Examples
--------
>>> import title
>>> run code

"""
import multiprocessing as mp
from typing import Callable


def _map_function(args):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    func, trial_no, kwargs = args
    return func(trial_no, **kwargs)


def conduct_monte_carlo(execute: Callable, n_trials: int, n_processes: int = 8, **kwargs):
    """_summary_

    Args:
        execute (_type_): _description_
    """

    # Create a multiprocessing Pool
    with mp.Pool(processes=n_processes) as pool:
        # Process the items in parallel
        results = pool.map(
            _map_function, [(execute, trial_no, kwargs) for trial_no in range(n_trials)]
        )

    return results
