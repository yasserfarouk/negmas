import random
from typing import Any, Callable, Iterable, Literal

import numpy as np

from negmas.helpers.misc import distribute_integer_randomly, floatin, intin
from negmas.negotiators.helpers import PolyAspiration, TimeCurve

__all__ = [
    "sample_between",
    "make_pareto",
    "make_endpoints",
    "make_non_pareto",
    "generate_utility_values",
    "ParetoGenerator",
    "GENERATOR_MAP",
    "make_curve_pareto",
    "make_piecewise_linear_pareto",
    "make_zero_sum_pareto",
]


def make_curve(
    curve: TimeCurve
    | Literal["boulware"]
    | Literal["conceder"]
    | Literal["linear"]
    | float,
    starting_utility: float = 1.0,
) -> TimeCurve:
    """
    Generates a `TimeCurve` or `Aspiration` with optional `starting_utility`self.

    Default behavior is to return a `PolyAspiration` object.
    """
    if isinstance(curve, TimeCurve):
        return curve
    return PolyAspiration(starting_utility, curve)


def sample_between(
    start: float,
    end: float,
    n: int,
    endpoint=True,
    main_range: tuple[float, float] = (0.3, 0.5),
):
    """
    Samples n values between the start and end given, optionally with the endpoint included.

    Remarks:
        - The samples are drawn to be somewhere between main_range limits of the range between start and end.
        - Samples that end up smaller than start are sent back to the first 50% of the range between start and end.
        - Samples that end up larger than end are sent back to the first 50% of the range between start and end.
    """
    if n == 0:
        return []
    if end - start < 1e-8:
        return [start] * n
    samples = np.linspace(start, end, num=n, endpoint=endpoint)
    if len(samples) > 2:
        samples[1:-1] += (
            main_range[0]
            * (end - start)
            * (np.random.random(len(samples) - 2) - main_range[1])
        )
        samples[samples < start] = start + np.random.random(
            len(samples[samples < start])
        ) * 0.5 * (end - start)
        samples[samples > end] = end - np.random.random(
            len(samples[samples > end])
        ) * 0.5 * (end - start)
    return samples


def make_endpoints(
    n_segments: int | tuple[int, int] | list[int],
    xrange: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
) -> list[tuple[float, ...]]:
    """Create endpoints for n_segments with the ranges given"""
    n = intin(n_segments) + 1
    x = np.sort(sample_between(*xrange[0], n))
    y = np.sort(sample_between(*xrange[1], n))[::-1]
    return list(zip(x, y))


def make_pareto(
    endpoints: list[tuple[float, ...]], n_outcomes: int
) -> list[tuple[float, ...]]:
    """
    Generates a piecewise linear curve with a specified number of segments and all slopes negative.

    Args:
      endpoints: A list of (x, y) coordinates for the endpoints of each segment.
      n_outcomes: number of values to distribute in the piecewise linear lines connecting the endpoints

    Returns:
        list of tuples representing utilities at the pareto-frontier

    Remarks:
      - If the number of end points is less than n_outcomes + 1,
    """
    if not endpoints and n_outcomes < 1:
        return []
    if len(endpoints) < 2:
        return [endpoints[0]] * n_outcomes
    if n_outcomes < 2:
        return [endpoints[0]] * n_outcomes
    num_segments = len(endpoints) - 1
    if n_outcomes <= num_segments:
        n_per_segment = [1] * (n_outcomes - 1) + [0] * (num_segments - n_outcomes) + [1]
    else:
        n_per_segment = distribute_integer_randomly(n_outcomes - 1, num_segments)

    # Calculate slopes based on difference in y-coordinates
    slopes = [
        (endpoints[i + 1][1] - endpoints[i][1])
        / (endpoints[i + 1][0] - endpoints[i][0])
        for i in range(num_segments)
    ]

    points = []

    # Iterate through each segment and generate its points
    for i in range(num_segments):
        start_x, start_y = endpoints[i]
        end_x = endpoints[i + 1][0]
        slope = slopes[i]
        n = n_per_segment[i] - 1
        if n == 0:
            end_y = endpoints[i + 1][1]
            if i == 0:
                points.append((start_x, start_y))
            elif i < num_segments - 1:
                r = random.random()
                if r < 0.5:
                    points.append((start_x, start_y))
                else:
                    points.append((end_x, end_y))
            else:
                points.append((end_x, end_y))
            continue
        if n < 0:
            continue
        samples = sample_between(start_x, end_x, n, endpoint=False)
        points.append((start_x, start_y))

        # Calculate x and y values for each point on the segment
        for x in samples:
            y = start_y + slope * (x - start_x)
            points.append((x, y))
    # append the last point representing the last endpoint
    if len(points) < n_outcomes:
        points.append((endpoints[-1][0], endpoints[-1][1]))
    return points


def make_non_pareto(
    pareto: list[tuple[float, ...]],
    n: int,
    limit: tuple[float, float] = (0, 0),
    eps: float = 1e-4,
) -> list[tuple[float, ...]]:
    """Adds extra outcomes dominated by the given Pareto outcomes

    Args:
        pareto: list of pareto points in the utility space.
        n: number of outcomes to add that are dominated by the pareto outcomes
        limit: The minimum x and y values allowed
        eps: A small margin
    """
    if n < 1:
        return []
    if len(pareto) < 1:
        raise ValueError(
            f"Cannot use a pareto frontier with zero points. No such thing can exist!!"
        )
    ndims = len(pareto[0])
    available_indices, ok = [], []
    # remove points in the pareto under the limit from consideration
    for p in pareto:
        available = []
        for d in range(ndims):
            if p[d] < limit[d] + eps:
                continue
            available.append(d)
            available_indices.append(tuple(available))
        if not available_indices:
            continue
        ok.append(len(available_indices[-1]) > 0)
    _ok = np.asarray(ok, dtype=bool)
    pareto = np.asarray(pareto)[_ok].tolist()
    available_indices = [_ for _, ok in zip(available_indices, _ok) if ok]
    n_pareto = len(pareto)
    if n_pareto < 1:
        return []
    if n_pareto == 1:
        points = np.zeros(n, dtype=int)
    else:
        points = np.random.randint(0, n_pareto - 1, size=n, dtype=int)
    extra = []
    if len(points) < 1:
        return []
    for i in points:
        p, available = pareto[i], available_indices[i]
        new = [_ for _ in p]
        if len(available) == 1:
            changes = available
        else:
            changes = random.sample(available, k=random.randint(1, len(available)))
        for d in changes:
            new[d] -= random.random() * (p[d] - limit[d] - eps) - eps
        extra.append(tuple(new))

    return extra


def make_zero_sum_pareto(n_pareto: int):
    """Generate a zero-sum Pareto generator of the given number of segments"""
    return make_piecewise_linear_pareto(n_pareto, n_segments=1)


def make_piecewise_linear_pareto(
    n_pareto: int, *, n_segments: int | list[int] | tuple[int, int] = (2, 5)
):
    """Generate a piecewise-linear Pareto generator of the given number of segments"""
    endpoints = make_endpoints(n_segments)
    if len(endpoints) < n_pareto:
        endpoints = endpoints[: n_pareto - 1]
    return make_pareto(endpoints, n_pareto)


def make_curve_pareto(
    n_pareto: int,
    *,
    shape: TimeCurve
    | Literal["boulware"]
    | Literal["conceder"]
    | Literal["linear"]
    | float
    | tuple[float, float]
    | list[float] = (0.1, 6),
):
    """Generate a Pareto curve of the given shape."""
    if isinstance(shape, Iterable) and not isinstance(shape, str):
        shape = floatin(shape, log_uniform=True)
    curve = make_curve(shape)
    x = 1.0 - np.sort(np.random.random(n_pareto), axis=None)
    y = [curve.utility_range(_) for _ in x]
    y = [(a + b) / 2 for a, b in y]
    return list(zip(x, y, strict=True))


ParetoGenerator = Callable[[int], list[tuple[float, ...]]]
"""Type of Pareto generators. Receives a number of points and returns a list of utility values corresponding to a Pareto front of that size"""

GENERATOR_MAP: dict[str, ParetoGenerator] = dict(
    piecewise_linear=make_piecewise_linear_pareto,
    curve=make_curve_pareto,
    zero_sum=make_zero_sum_pareto,
)


def generate_utility_values(
    n_pareto: int,
    n_outcomes: int,
    n_ufuns: int = 2,
    pareto_first=False,
    pareto_generator: ParetoGenerator | str = "piecewise_linear",
    generator_params: dict[str, Any] | None = None,
) -> list[tuple[float, ...]]:
    """
    Generates ufuns that have a controllable Pareto frontier

    Args:
        n_pareto: Number of outcomes on the Pareto frontier
        n_outcomes: Total number of outcomes to generate
        pareto_first: If given, Pareto outcomes come first in the returned results.
        pareto_generator: The method used to generate the Pareto front utilities
        generator_params: The parameters passed to the Pareto generator

    Returns:
        A list of tuples each giving the utilities of one outcome
    """
    if n_ufuns != 2:
        raise NotImplementedError(
            f"We only support generation of two ufuns using this method. {n_ufuns} ufuns are requested."
        )
    if not generator_params:
        generator_params = dict()
    if isinstance(pareto_generator, str):
        pareto_generator = GENERATOR_MAP[pareto_generator]
    pareto = pareto_generator(n_pareto, **generator_params)
    extra = make_non_pareto(pareto, n_outcomes - n_pareto)
    # print(len(xvals), len(extra), n_segments, n_pareto, n_outcomes)
    points = list(pareto) + extra
    if not pareto_first:
        random.shuffle(points)
    return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rich import print

    fig, subplots = plt.subplots(4, 4, sharex=True, sharey=True, squeeze=True)
    subplots = subplots.flatten()
    choices = list(GENERATOR_MAP.keys())

    for ax in subplots:
        if random.random() > 0.33:
            n_pareto = random.randint(30, 100)
            n_outcomes = 400
        else:
            n_pareto = n_outcomes = 400
        method = random.choice(choices)
        if method == "zero_sum":
            choices = [_ for _ in choices if _ != "zero_sum"]
        params = dict(
            piecewise_linear=dict(n_segments=random.randint(1, min(n_pareto, 20))),
            curve=dict(
                shape=4
                * round(
                    random.random() if random.random() > 0.5 else 1 / random.random(), 2
                )
            ),
        ).get(method, dict())
        points = generate_utility_values(
            n_pareto,
            n_outcomes,
            pareto_first=True,
            generator_params=params,
            pareto_generator=method,
        )
        assert len(points) == n_outcomes, f"{len(points)=}, {n_outcomes=}"
        print(
            f"Generated {len(set(points))=} unique outcomes for: {n_outcomes=}, {n_pareto=}, {method=}, {params=}"
        )
        ax.scatter([_[0] for _ in points], [_[1] for _ in points], color="b")
        ax.plot(
            [_[0] for _ in points[:n_pareto]],
            [_[1] for _ in points[:n_pareto]],
            color="r",
            marker="x",
        )
        ax.set_title(f"{method}: {str(params)}")
    plt.show()
