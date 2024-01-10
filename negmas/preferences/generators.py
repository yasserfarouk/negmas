import itertools
import random
from typing import Any, Callable, Iterable, Literal, overload

import numpy as np

from negmas.helpers.misc import (
    distribute_integer_randomly,
    floatin,
    generate_random_weights,
    intin,
)
from negmas.negotiators.helpers import PolyAspiration, TimeCurve
from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.value_fun import TableFun

__all__ = [
    "sample_between",
    "make_pareto",
    "make_endpoints",
    "make_non_pareto",
    "generate_utility_values",
    "generate_multi_issue_ufuns",
    "generate_single_issue_ufuns",
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


def zip_cycle(A, *args):
    """Zips generators A and B, cycling through B as needed if it's shorter.

    Args:
        A: The first generator.
        args: Other generators

    Yields:
        Pairs of elements from A and B, cycling through B as needed.

    Raises:
        ValueError: If B is longer than A.
    """

    if len(A) < min(len(_) for _ in args):  # Check for invalid length relationship
        raise ValueError("Generator A must be equal to or longer than generator B.")

    cyc = (itertools.cycle(_) for _ in args)
    return zip(A, *cyc)


def zip_cycle_longest(A, B):
    """Zips generators A and B, cycling through B as needed if it's shorter.

    Args:
        A: The first generator.
        B: The second generator.

    Yields:
        Pairs of elements from A and B, cycling through B as needed.

    Raises:
        ValueError: If B is longer than A.
    """

    if len(A) < len(B):  # Check for invalid length relationship
        raise ValueError("Generator A must be equal to or longer than generator B.")

    B_cycle = itertools.cycle(B)  # Create a cycle of the shorter generator B
    return zip(A, B_cycle)  # Zip A with the cycled B


def _adjust_ufuns(
    ufuns,
    os,
    linear,
    rational_fractions,
    selector: Callable[[float, float], float] = max,
):
    outcomes = None
    if rational_fractions:
        for u, f in zip_cycle_longest(ufuns, rational_fractions):
            outcomes = os.enumerate_or_sample()
            vals = sorted(u(_) for _ in outcomes)
            n_outcomes = len(vals)
            limit = max(0, min(len(vals) - 1, int((1 - f) * len(vals) + 0.5)))
            if f * n_outcomes <= 1:
                r = vals[-1] + 0.001
            elif f >= 1.0:
                r = vals[0] - 0.001
            else:
                r = vals[limit] - 1e-9
            u.reserved_value = selector(r, u.reserved_value)

    if linear:
        return ufuns
    maps = []
    if outcomes is None:
        outcomes = os.enumerate_or_sample()
    for u in ufuns:
        maps.append(
            MappingUtilityFunction(
                dict(zip(outcomes, [u(_) for _ in outcomes])),
                outcome_space=os,
                name=u.name,
                reserved_value=u.reserved_value,
            )
        )
    return tuple(maps)


def generate_single_issue_ufuns(
    n_pareto: int,
    n_outcomes: int,
    n_ufuns: int = 2,
    pareto_first=False,
    pareto_generator: ParetoGenerator | str = "piecewise_linear",
    generator_params: dict[str, Any] | None = None,
    reserved_values: list[float] | tuple[float, float] | float = 0.0,
    rational_fractions: list[float] | None = None,
    reservation_selector: Callable[[float, float], float] = max,
    issue_name: str = "portions",
    os_name: str = "S",
    ufun_names: tuple[str, ...] | list[str] | None = None,
    numeric: bool = False,
    linear: bool = True,
) -> tuple[LinearAdditiveUtilityFunction | MappingUtilityFunction, ...]:
    """Generates a set of single-issue ufuns

    Args:
        pareto_first: return the pareto outcomes first in the outcome-space
        n_pareto: number of pareto outcomes
        n_outcomes: number of outcomes. Must be >= `n_pareto`
        n_ufuns: number of ufuns
        pareto_generator: The generation method. See `GENERATOR_MAP`
        generator_params: parameters of the generator
        reserved_values: Reserved values to use for generated ufuns.
                         A list to cycle through, or a tuple to sample within or a single value to repeat.
        rational_fractions: Fraction of rational outcomes for each ufun. Should have `n_ufuns` values
                            (or it will cycle). If given with `reserved_values`, the `reservation_selector`
                            will be used to select the final reservation value
        reservation_selector: A function that receives the reserved value suggested by rational_fraction and
                               that suggested by reserved_value and selects the final reserved values. The
                               default is `max`. You can replace that with `min`, the first, the second,
                               mean, or any combination.
        issue_name: Name of the single issue.
        os_name: Name of the outcomes space in the ufuns created
        ufun_names: Names of utility functions
        numeric: All issue values are numeric (integers)
        linear: Whether to use a linear-additive-ufun instead of the general mapping ufun

    Returns:
        A tuple of single-issue ufuns
    """
    if (
        isinstance(reserved_values, tuple)
        and len(reserved_values) == 2
        and reserved_values[0] <= reserved_values[1]
    ):
        reserved_values = [
            floatin(reserved_values, log_uniform=False) for _ in range(n_ufuns)
        ]
    elif not isinstance(reserved_values, Iterable):
        reserved_values = [float(reserved_values)] * n_ufuns
    vals = generate_utility_values(
        n_pareto, n_outcomes, n_ufuns, pareto_first, pareto_generator, generator_params
    )
    n = len(vals)

    if numeric:
        issues = (make_issue(n, issue_name),)
    else:
        issues = (make_issue([f"{i}_{n-1 - i}" for i in range(n)], issue_name),)
    os = make_os(issues, name=os_name)
    if not ufun_names:
        ufun_names = tuple(f"u{i}" for i in range(n_ufuns))
    ufuns = tuple(
        LinearAdditiveUtilityFunction(
            values=(
                TableFun({_: float(vals[i][k]) for i, _ in enumerate(issues[0].all)}),
            ),
            name=uname,
            outcome_space=os,
            reserved_value=r,
        )
        for k, (r, uname) in enumerate(zip_cycle_longest(ufun_names, reserved_values))
        # for k, (uname, r) in enumerate(zip(("First", "Second"), reserved_ranges))
    )
    return _adjust_ufuns(ufuns, os, linear, rational_fractions, reservation_selector)


@overload
def generate_multi_issue_ufuns(
    n_issues: int,
    n_values: int | tuple[int, int] = 0,
    sizes: None = None,
    n_ufuns: int = 2,
    pareto_generators: tuple[ParetoGenerator | str, ...] = ("piecewise_linear",),
    generator_params: tuple[dict[str, Any], ...] | None = None,
    reserved_values: list[float] | tuple[float, float] | float = 0.0,
    rational_fractions: list[float] | None = None,
    reservation_selector: Callable[[float, float], float] = max,
    issue_names: tuple[str, ...] | list[str] | None = None,
    os_name: str | None = None,
    ufun_names: tuple[str, ...] | None = None,
    numeric: bool = False,
    linear: bool = True,
) -> tuple[LinearAdditiveUtilityFunction | MappingUtilityFunction, ...]:
    ...


@overload
def generate_multi_issue_ufuns(
    n_issues: int,
    n_values: None = None,
    sizes: tuple[int, ...] | list[int] = tuple(),
    n_ufuns: int = 2,
    pareto_generators: tuple[ParetoGenerator | str, ...] = ("piecewise_linear",),
    generator_params: tuple[dict[str, Any], ...] | None = None,
    reserved_values: list[float] | tuple[float, float] | float = 0.0,
    rational_fractions: list[float] | None = None,
    reservation_selector: Callable[[float, float], float] = max,
    issue_names: tuple[str, ...] | list[str] | None = None,
    os_name: str | None = None,
    ufun_names: tuple[str, ...] | None = None,
    numeric: bool = False,
    linear: bool = True,
) -> tuple[LinearAdditiveUtilityFunction | MappingUtilityFunction, ...]:
    ...


def generate_multi_issue_ufuns(
    n_issues: int,
    n_values: int | tuple[int, int] | None = None,
    sizes: None | tuple[int, ...] | list[int] = None,
    n_ufuns: int = 2,
    pareto_generators: tuple[ParetoGenerator | str, ...] = tuple(GENERATOR_MAP.keys()),
    generator_params: tuple[dict[str, Any], ...] | None = None,
    reserved_values: list[float] | tuple[float, float] | float = 0.0,
    rational_fractions: list[float] | None = None,
    reservation_selector: Callable[[float, float], float] = max,
    issue_names: tuple[str, ...] | list[str] | None = None,
    os_name: str | None = None,
    ufun_names: tuple[str, ...] | None = None,
    numeric: bool = False,
    linear: bool = True,
) -> tuple[LinearAdditiveUtilityFunction | MappingUtilityFunction, ...]:
    """Generates a set of ufuns with an outcome space of the given number of issues.

    Args:
        n_issues: Number of issues
        n_values: Range of issue sizes. Only used if `sizes` is not passed
        sizes: Sizes of issues
        n_ufuns: Number of ufuns to generate
        pareto_generators: The generators to use internally to generate value functions
        generator_params: parameters of the generators.
        reserved_values: Reserved values to use for generated ufuns.
                         A list to cycle through, or a tuple to sample within or a single value to repeat.
        rational_fractions: Fraction of rational outcomes for each ufun. Should have `n_ufuns` values
                            (or it will cycle). If given with `reserved_values`, the `reservation_selector`
                            will be used to select the final reservation value
        reservation_selector: A function that receives the reserved value suggested by rational_fraction and
                               that suggested by reserved_value and selects the final reserved values. The
                               default is `max`. You can replace that with `min`, the first, the second,
                               mean, or any combination.
        os_name: Name of the outcomes space in the ufuns created
        ufun_names: Names of utility functions
        numeric: All issue values are numeric (integers)
        linear: Whether to use a linear-additive-ufun instead of the general mapping ufun

    Returns:
        A tuple of `n_ufuns` utility functions.
    """
    if (
        isinstance(reserved_values, tuple)
        and len(reserved_values) == 2
        and reserved_values[0] <= reserved_values[1]
    ):
        reserved_values = [
            floatin(reserved_values, log_uniform=False) for _ in range(n_ufuns)
        ]
    elif not isinstance(reserved_values, Iterable):
        reserved_values = [float(reserved_values)] * n_ufuns
    if ufun_names is None:
        ufun_names = tuple(f"u{i+1}" for i in range(n_ufuns))
    vals = [dict() for _ in range(n_ufuns)]
    if not generator_params:
        generator_params = [dict() for _ in pareto_generators]  # type: ignore
    if not issue_names:
        issue_names = tuple(f"i{k+1}" for k in range(n_issues))
    gp = list(zip(pareto_generators, generator_params))  # type: ignore
    if sizes is None:
        assert n_values is not None
        sizes = tuple(intin(n_values) for _ in range(n_issues))  # type: ignore
    assert sizes is not None
    for i, n in enumerate(sizes):
        g, p = random.choice(gp)
        v = generate_utility_values(
            n, n, n_ufuns, pareto_generator=g, generator_params=p
        )
        for j in range(n_ufuns):
            vals[j][i] = [float(_[j]) for _ in v]
    weights = [float(_) for _ in generate_random_weights(n_issues)]
    os = make_os(
        issues=[
            make_issue(ni, name=iname)
            if numeric
            else make_issue([f"v{k+1}" for k in range(ni)], name=f"i{i+1}")
            for i, (ni, iname) in enumerate(zip(sizes, issue_names))
        ],
        name=os_name,
    )
    ufuns = tuple(
        LinearAdditiveUtilityFunction(
            values=[
                TableFun(
                    dict(
                        zip(
                            [
                                k if numeric else f"v{k+1}"
                                for k in range(len(vals[j][i]))
                            ],
                            vals[j][i],
                        )
                    )
                )
                for i in range(n_issues)
            ],
            weights=weights,
            outcome_space=os,
            name=name,
            reserved_value=r,
        )
        for j, (name, r) in enumerate(zip_cycle_longest(ufun_names, reserved_values))
    )
    return _adjust_ufuns(ufuns, os, linear, rational_fractions, reservation_selector)


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
