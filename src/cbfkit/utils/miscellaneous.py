def print_progress(iteration: int, total: int) -> None:
    """Prints iteration progress only when percent complete is a whole integer.

    Arguments:
        iteration: current iteration number
        total: total number of iterations

    Returns:
        None

    """
    percent = (iteration / total) * 100
    if percent.is_integer():
        print(f"Progress: {int(percent)}%")
