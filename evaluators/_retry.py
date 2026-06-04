import time
from functools import wraps


def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    """Retry a callable on exception with linear backoff (delay * attempt).

    Re-raises the last exception if all attempts fail. Used to wrap judge-LLM
    calls that occasionally hit transient 429/500 responses.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            raise last_exc
        return wrapper
    return decorator
