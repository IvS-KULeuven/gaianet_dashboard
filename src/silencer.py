import sys
import os


def suppress_print(func, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        return func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
