"""Functions to aid in conducting Monte Carlo simulations.

Functions
---------
_map_function(args)
    Helper function for multiprocessing.
conduct_monte_carlo(execute, n_trials, n_processes, **kwargs)
    Conducts a Monte Carlo simulation."""

import inspect
import multiprocessing as mp
from typing import Callable, Optional
import numpy as np
from jax import random


def _map_function(args):
    """Helper function to execute a single trial.

    Args:
        args (tuple): Tuple containing (func, trial_no, worker_seed, kwargs).

    Returns
    -------
        Any: Result of the trial execution.
    """
    func, trial_no, worker_seed, kwargs = args

    # Seed global numpy random for legacy support and ensure diversity
    if worker_seed is not None:
        # np.random.seed requires uint32
        np.random.seed(int(worker_seed % (2**32)))

    # Determine if we can inject JAX key
    inject_key = False
    if worker_seed is not None:
        try:
            sig = inspect.signature(func)
            params = sig.parameters
            if "key" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
                inject_key = True
        except (ValueError, TypeError):
            # Fallback: if we can't inspect, assume compliant with docstring (accepts kwargs)
            inject_key = True

    if inject_key and worker_seed is not None:
        kwargs = kwargs.copy()
        kwargs["key"] = random.PRNGKey(worker_seed)

    return func(trial_no, **kwargs)


def conduct_monte_carlo(
    execute: Callable, n_trials: int, n_processes: int = 1, seed: Optional[int] = None, **kwargs
):
    """Conducts a Monte Carlo simulation.

    Args:
        execute (Callable): The function to execute for each trial.
                            Must accept `trial_no` (int) as the first argument and `**kwargs`.
        n_trials (int): Number of trials to run.
        n_processes (int, optional): Number of parallel processes. Defaults to 1.
        seed (Optional[int], optional): Base seed for random number generation.
                                        If provided, each trial receives a unique JAX PRNGKey
                                        passed as 'key' in kwargs.
                                        Requires `execute` to accept `**kwargs` or `key`.
        **kwargs: Additional keyword arguments passed to `execute`.

    Returns
    -------
        list: A list of results from each trial.
    """
    args_list = []

    # Generate integer seeds for each trial
    # If seed is None, we use entropy to ensure trials are different
    rng = np.random.default_rng(seed)
    # Generate random integers safely within range for PRNGKey
    worker_seeds = rng.integers(low=0, high=2**32 - 1, size=n_trials)

    for trial_no in range(n_trials):
        worker_seed = worker_seeds[trial_no]

        # Note: We do NOT convert to JAX key here to avoid pickling issues with multiprocessing
        args_list.append((execute, trial_no, worker_seed, kwargs))

    if n_processes > 1:
        # Create a multiprocessing Pool
        with mp.Pool(processes=n_processes) as pool:
            # Process the items in parallel
            results = pool.map(_map_function, args_list)
    else:
        # Execute sequentially
        results = list(map(_map_function, args_list))

    return results
