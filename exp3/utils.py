import sys
import functools
import time
import requests.exceptions


# Timing decorator for profiled functions
def profile_with_timing(func):
    """Decorator that adds timing to profiled functions with exception handling"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"\n[TIMING] {func_name} - START")
        sys.stdout.flush()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"[TIMING] {func_name} - END (took {elapsed:.2f}s)")
            sys.stdout.flush()
            return result
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            print(f"[TIMING] {func_name} - FAILED (TIMEOUT after {elapsed:.2f}s): {type(e).__name__}")
            print(f"[PROFILE] ⚠️  WARNING: Profiling data below may be incomplete due to timeout")
            sys.stdout.flush()
            raise
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            print(f"[TIMING] {func_name} - FAILED (REQUEST_ERROR after {elapsed:.2f}s): {type(e).__name__}")
            print(f"[PROFILE] ⚠️  WARNING: Profiling data below may be incomplete due to request error")
            sys.stdout.flush()
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[TIMING] {func_name} - FAILED (ERROR after {elapsed:.2f}s): {type(e).__name__}: {str(e)}")
            print(f"[PROFILE] ⚠️  WARNING: Profiling data below may be incomplete due to error")
            sys.stdout.flush()
            raise

    return wrapper