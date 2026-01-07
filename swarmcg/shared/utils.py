import warnings
from functools import wraps


def catch_warnings(*warning_categories):
    if len(warning_categories) == 1 and isinstance(warning_categories[0], (list, tuple, set)):
        warning_categories = tuple(warning_categories[0])

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for warning in warning_categories:
                    warnings.filterwarnings("ignore", category=warning)
                return function(*args, **kwargs)

        return wrapper

    return decorator


def parse_string_args(x):
    try:
        to_float = float(x)
        if int(to_float) - to_float != 0:
            return to_float
        else:
            return int(to_float)
    except ValueError as _:
        return x
