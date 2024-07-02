import datetime


def print_with_time(string):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_str}] " + string)
