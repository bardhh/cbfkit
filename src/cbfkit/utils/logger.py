import pandas as pd
from typing import Dict, Any

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
    df = pd.DataFrame.from_dict(LOG)
    df.to_csv(filepath)


def extract_log(key: str) -> None:
    """Extracts the key data from the log.

    Args:
    key (str): key to the log

    Returns:
    key_data (list): data from log corresponding to key

    """
    return [entry[key] for entry in LOG]
