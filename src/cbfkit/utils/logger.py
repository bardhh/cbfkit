"""Logger module.

This module provides functions required for logging simulation data.

Functions
---------
write_log(filepath, data): writes the data contained in the log to a .csv file
extract_log(key, data): retrieves data corresponding to key from log

Notes
-----
This is currently implemented inefficiently, as it is designed to work
for any type of simulation with arbitrary data.

Examples
--------
>>> import logger
>>> data = [{'x': 1, 'y':2}]
>>> fpath = "point.csv"
>>> logger.write_log(fpath, data)
>>> x = logger.extract_log("x", data)
"""

import os
from typing import Any, Dict, List, Union

LogEntry = Dict[str, Any]


def write_log(filepath: str, data: Union[List[LogEntry], Dict[str, Any]]) -> None:
    """Writes logged data out to csv file specified at filepath.

    Args:
    filepath (str): path to save file
    data (Union[List[LogEntry], Dict[str, Any]]): list of log entries or dict of arrays

    Returns
    -------
    None
    """
    folder_path = os.path.dirname(filepath)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if filepath[-4:] != ".csv":
        if filepath[-4] == ".":
            raise ValueError("filepath must have no extension, or have extension `.csv`")
        filepath += ".csv"

    import csv

    if isinstance(data, list):
        if not data:
            return
        keys = data[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    elif isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            return

        # Check that all columns have the same length to prevent data loss with zip
        length = len(data[keys[0]])
        for k in keys[1:]:
            if len(data[k]) != length:
                raise ValueError("All arrays must be of the same length")

        rows = zip(*[data[k] for k in keys])
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)
    else:
        raise ValueError("data must be a list of dicts or a dict of lists")


def extract_log(key: str, data: List[LogEntry]) -> List[Any]:
    """Extracts the key data from the log.

    Args:
    key (str): key to the log
    data (List[LogEntry]): list of log entries

    Returns
    -------
    key_data (list): data from log corresponding to key
    """
    return [entry[key] for entry in data]


def print_progress(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "█",
    printEnd: str = "\r",
) -> None:
    r"""Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
