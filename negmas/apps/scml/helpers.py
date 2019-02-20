from random import gauss
import numpy as np

__all__ = [
    'pos_gauss', 'safe_max', 'zero_runs'
]


def pos_gauss(mu, sigma):
    """Returns a sample from a rectified gaussian"""
    x = gauss(mu, sigma)
    return abs(x)


def safe_max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def zero_runs(a: np.array) -> np.array:
    """
    Finds all runs of zero in an array

    Args:
        a: Input array (assumed to be 1D)

    Returns:
        np.array: A 2D array giving beginning and end (exclusive) of zero stretches in the input array.
    """
    if len(a) == 0:
        return []
    if np.all(np.equal(a, 0).view(np.int8)):
        return np.array([[0, len(a)]])
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    if len(ranges) == 0 and a[0] == 0:
        return np.array([[0, len(a)]])
    return ranges
