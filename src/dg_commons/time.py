from functools import wraps
from time import process_time
from typing import Callable

from dg_commons import logger


def time_function(func: Callable):
    """Decorator to time the execution time of a function/method call"""

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = process_time()
        try:
            return func(*args, **kwargs)
        finally:
            delta = process_time() - start
            msg = f'Execution time of "{func.__qualname__}" defined in "{func.__module__}": {delta} s'
            logger.info(msg)

    return _time_it
