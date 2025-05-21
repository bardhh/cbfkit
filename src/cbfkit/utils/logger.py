"""
logger
================

This module provides functions required for logging simulation data.

Functions
---------
log(new_data): adds the latest data to the log
write_log(filepath): writes the data contained in the log to a .csv file
extract_log(key): retrieves data corresponding to key from log

Notes
-----
This is currently implemented inefficiently, as it is designed to work
for any type of simulation with arbitrary data.

Examples
--------
>>> import logger
>>> data = {'x': 1, 'y':2}
>>> logger.log(data)
>>> fpath = "point.csv"
>>> logger.write_log(fpath)
>>> x = logger.extract_log("x")

"""

import os
from typing import Dict, Any, List
import pandas as pd

# Main logger variable
LOG = []


def log(new_data: Dict[str, Any]) -> None:
    """Adds new_data to the log.

    Args:
    new_data (Dict): dictionary containing all data to be written with pertinent keys

    Returns:
    None

    """
    LOG.append(new_data)


def write_log(filepath: str) -> None:
    """Writes logged data out to csv file specified at filepath.

    Args:
    filepath (str): path to save file

    Returns:
    None

    """
    folder_path = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if filepath[-4:] != ".csv":
        if filepath[-4] == ".":
            raise ValueError("filepath must have no extension, or have extension `.csv`")
        filepath += ".csv"

    df = pd.DataFrame.from_dict(LOG)
    df.to_csv(filepath)


def extract_log(key: str) -> List[Any]:
    """Extracts the key data from the log.

    Args:
    key (str): key to the log

    Returns:
    key_data (list): data from log corresponding to key

    """
    return [entry[key] for entry in LOG]
