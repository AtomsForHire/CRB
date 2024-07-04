import datetime


def print_with_time(string):
    """Function for printing a string with the date time

    Parameters
    ----------
    - string: `string`
        whatever message to print out

    Returns
    -------
    None
    """
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_str}] " + string)
