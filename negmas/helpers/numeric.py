#!/usr/bin/env python
"""
A set of utilities that can be used by agents developed for the platform.

This set of utlities can be extended but must be backward compatible for at
least two versions
"""
from __future__ import annotations

import random
from typing import Iterable, TypeVar

import numpy as np
from scipy.stats import tmean

T = TypeVar("T")

__all__ = [
    "get_one_float",
    "get_one_int",
    "make_range",
    "truncated_mean",
    "sample",
]


def get_one_int(i: int | tuple[int, int]):
    if isinstance(i, int):
        return i
    return random.randint(*i)


def get_one_float(rng: float | tuple[float, float]):
    if isinstance(rng, float):
        return rng
    return random.random() * (rng[1] - rng[0]) + rng[0]


def make_range(x: T | tuple[T, T]) -> tuple[T, T]:
    if isinstance(x, Iterable):
        return x  # type: ignore
    return (x, x)


def truncated_mean(
    scores: np.ndarray,
    limits: tuple[float, float] | None = None,
    top_limit=2.0,
    bottom_limit=float("inf"),
    base="tukey",
    return_limits=False,
) -> float | tuple[float, tuple[float, float] | None]:
    """
    Calculates the truncated mean

    Args:
        scores: A list of scores for which to calculate the truncated mean
        limits: The limits to use for trimming the scores. If not given, they will
                be calculated based on `top_limit`, `bottom_limit` and `base.`
                You can pass the special value "mean" as a string to disable limits and
                calcualte the mean. You can pass the special value "median" to calculate
                the median (which is the same as passing top_limit==bottom_limit=0.5
                and base == "scores").
        top_limit: top limit on scores to use for truncated mean calculation. See `base`
        bottom_limit: bottom limit on scores to use for truncated mean calculation. See `base`
        base: The base for calculating the limits used to apply the `top_limit` and `bottom_limit`.
              Possible values are:
              - zscore: the number of sigmas to remove above/below. A good default choice is 3. Pass inf to disable a side.
              - tukey: the fraction of IQR to remove above/below. A good default choice is 1.5 or 3 (we use 2). Pass inf to disable a side.
              - iqr : same as tukey
              - iqr_fraction: the fraction is interpreted as the fraction of scores above/below the 1st/3rd qauntile
              - scores: the fraction is interpreted as fraction of highest and lowest scores
              - fraction: the fraction is interpreted as literal fraction of the values (i.e. given 10 values and 0.1, removes 1 value)
              - mean: simply returns the mean (limits ignored)
              - median: simply returns the median (limits ignored)
        return_limits: If true, the method will also return the limiting scores used in its mean calculation.
    """

    scores = np.asarray(scores)
    scores = scores[~np.isnan(scores)]

    if isinstance(limits, str) and limits.lower() == "mean":
        return tmean(scores, None) if not return_limits else (tmean(scores, None), None)
    if isinstance(limits, str) and limits.lower() == "median":
        return np.median(scores) if not return_limits else (np.median(scores), None)
    if limits is not None:
        return np.mean(scores) if not return_limits else (np.mean(scores), None)

    if base == "zscore":
        m, s = np.nanmean(scores), np.nanstd(scores)
        limits = (m - s * bottom_limit, m + s * top_limit)
    elif base in ("tukey", "iqr"):
        q1, q3 = np.quantile(scores, 0.25), np.quantile(scores, 0.75)
        iqr = q3 - q1
        limits = (
            q1 - (bottom_limit * iqr if not np.isinf(bottom_limit) else bottom_limit),
            q3 + (top_limit * iqr if not np.isinf(top_limit) else top_limit),
        )
    elif base == "iqr_fraction":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        limits = (np.quantile(scores, 0.25), np.quantile(scores, 0.75))
        high = np.sort(scores[scores > limits[1]])
        low = np.sort(scores[scores < limits[0]])
        limits = (  # type: ignore
            low[int((len(low) - 1) * bottom_limit)] if len(low) > 0 else None,
            high[int((len(high) - 1) * (1 - top_limit))] if len(high) > 0 else None,
        )
    elif base == "fraction":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        scores = np.sort(scores)
        top_indx = int((len(scores) - 1) * (1 - top_limit))
        bottom_indx = int((len(scores) - 1) * bottom_limit)
        if top_indx < bottom_indx:
            return float("nan") if not return_limits else (float("nan"), limits)
        m = np.mean(scores[bottom_indx : top_indx + 1])
        return m if not return_limits else (m, (scores[bottom_indx], scores[top_indx]))
    elif base == "scores":
        bottom_limit = min(1, max(0, bottom_limit))
        top_limit = min(1, max(0, top_limit))
        limits = (
            np.quantile(scores, bottom_limit),
            np.quantile(scores, 1 - top_limit),
        )
        if limits[0] > limits[1]:
            return float("nan") if not return_limits else (float("nan"), limits)
    elif base == "mean":
        return np.mean(scores) if not return_limits else (np.mean(scores), None)
    elif base == "median":
        return np.median(scores) if not return_limits else (np.median(scores), None)
    else:
        raise ValueError(f"Unknown base for truncated_mean ({base})")
    if len(scores) == 0 or limits[1] < limits[0]:
        return float("nan") if not return_limits else (float("nan"), limits)
    try:
        # this is an inclusive trimmed mean
        # tm = tmean(scores, limits)
        scores = scores[scores >= limits[0]]
        scores = scores[scores <= limits[1]]
        if len(scores) == 0:
            return float("nan") if not return_limits else (float("nan"), limits)
        tm = np.mean(scores)
        return tm if not return_limits else (tm, limits)
    except ValueError:
        return float("nan") if not return_limits else (float("nan"), limits)


def sample(n, k, grid=False, compact=True, endpoints=True):
    """
    Samples `k` items from `n` in the range (0, `n`-1) optionally explring them

    Args:
        n: The number of items to sample (assumed to range from index 0 to n-1)
        k: The number of samples to take
        grid: Sample on a grid (equally distanced as much as possible)
        compact: If True, the samples will be choosen near each other (see endpoints though)
        endpoints: If given, the first and last index are guaranteed to be in the samples
    """
    if k is None:
        return range(n)
    if n == 0:
        return []
    if k == 1:
        return (0,) if n else []
    if n < 2:
        return (0,) if k else []
    if k > n:
        return range(n)
    pre, post = [], []
    if endpoints:
        pre, post = [0], [n - 1]
        n, k = n - 2, k - 2
    if grid:
        step = int(n // k)
        if step < 1:
            step = 1
        l = 1 + step * ((n - 1) // step)
        l = min(l, n - 1)
        rng = range(step, l, step)
    elif compact:
        before = (n - k) // 2
        rng = range(before, min(n, before + k))
    else:
        rng = random.sample(range(n), k)
    if not endpoints:
        return rng
    return pre + list(rng) + post
