"""Utility helpers."""

import time
from contextlib import contextmanager


@contextmanager
def timer(description: str = "Operation"):
    """Context manager that measures and prints elapsed time."""
    start = time.time()
    yield lambda: time.time() - start
    elapsed = time.time() - start
    print(f"{description} completed in {elapsed:.1f}s")


def format_number(n) -> str:
    """Format a number with comma separators."""
    return f"{n:,.0f}"
