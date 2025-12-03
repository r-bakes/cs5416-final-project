import sys
import functools
import time


# Timing decorator for profiled functions
def profile_with_timing(func):
    """Decorator that adds timing to profiled functions"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"\n[TIMING] {func_name} - START")
        sys.stdout.flush()
        start_time = time.time()

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        print(f"[TIMING] {func_name} - END (took {elapsed:.2f}s)")
        sys.stdout.flush()

        return result

    return wrapper
