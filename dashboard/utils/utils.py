import base64
import functools
import time

from loguru import logger


def encode_image(data):
    """
    Encodes binary image data to a base64 ASCII string.
    Args:
        data (bytes): The binary image data to encode.
    Returns:
        str: The base64 encoded ASCII string representation of the image data.
    """

    return base64.b64encode(data).decode('ascii')


def log_execution_time(level="DEBUG"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            log_method = getattr(logger, level.lower())

            log_method(
                f"'{func.__name__}' executed in {elapsed_time:.2f} seconds")

            return result
        return wrapper
    return decorator
