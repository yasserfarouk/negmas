"""Replicator dynamics over an empirical payoff matrix
(:class:`negmas.tournaments.analysis.payoff.PayoffTable`).

Implements the continuous-time single-population (symmetric) and
two-population (asymmetric) replicator dynamics equations from the EGTA
survey (Wellman, Tuyls & Greenwald 2025), section 4.2/4.4, integrated with
``scipy.integrate.solve_ivp``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from .payoff import PayoffTable

__all__ = [
    "ReplicatorTrajectory",
    "symmetric_replicator_dynamics",
    "asymmetric_replicator_dynamics",
    "final_mixed_strategy",
]


class ReplicatorTrajectory(NamedTuple):
    """Trajectory returned by the replicator dynamics functions."""

    times: np.ndarray
    """1D array of shape (n_steps,) with the time points."""
    x: np.ndarray
    """2D array of shape (n_steps, n_strategies): population 1's mixture over time."""
    strategies_x: list[str]
    """Strategy names for population 1 (rows of the payoff matrix)."""
    y: np.ndarray | None = None
    """2D array of shape (n_steps, n_strategies_y): population 2's mixture over
    time (only set for :func:`asymmetric_replicator_dynamics`)."""
    strategies_y: list[str] | None = None
    """Strategy names for population 2 (only set for the asymmetric case)."""


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def symmetric_replicator_dynamics(
    payoff: PayoffTable,
    x0: dict[str, float] | None = None,
    t_max: float = 100.0,
    n_steps: int = 200,
) -> ReplicatorTrajectory:
    """Single-population (symmetric) replicator dynamics::

        dx_i/dt = x_i * ((A x)_i - x^T A x)

    where ``A`` is the payoff matrix. Starts from a uniform mixture over all
    strategies unless ``x0`` is given, and integrates up to ``t_max`` with
    ``scipy.integrate.solve_ivp``, treating any missing (NaN) payoff cell as 0.

    This is the natural evolutionary-dynamics companion to
    :func:`negmas.tournaments.analysis.equilibria.pure_symmetric_nash_equilibria`:
    every pure symmetric Nash equilibrium is a fixed point, but only
    (asymptotically) stable fixed points -- evolutionarily stable strategies --
    are attractors of this dynamic.
    """
    A = np.nan_to_num(payoff.matrix, nan=0.0)
    n = payoff.n
    x_init = _uniform(n) if x0 is None else _dict_to_vector(x0, payoff.strategies)

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        fitness = A @ x
        avg = float(x @ fitness)
        return x * (fitness - avg)

    times = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(rhs, (0.0, t_max), x_init, t_eval=times, method="RK45")
    x = np.clip(sol.y.T, 0.0, None)
    x = x / x.sum(axis=1, keepdims=True)
    return ReplicatorTrajectory(times=sol.t, x=x, strategies_x=list(payoff.strategies))


def asymmetric_replicator_dynamics(
    payoff_row: PayoffTable,
    payoff_col: PayoffTable | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    t_max: float = 100.0,
    n_steps: int = 200,
) -> ReplicatorTrajectory:
    """Two-population (asymmetric) replicator dynamics::

        dx_i/dt = x_i * ((A y)_i - x^T A y)
        dy_j/dt = y_j * ((x^T B)_j - x^T B y)

    where population 1 plays row strategies with payoff matrix ``A`` and
    population 2 plays column strategies with payoff matrix ``B``.

    If ``payoff_col`` is not given, ``B = payoff_row.matrix.T`` is used,
    i.e. both populations are drawn from the same symmetric game (this lets
    two distinct starting mixtures for the "same" strategy set converge
    independently, unlike :func:`symmetric_replicator_dynamics` which tracks
    a single shared population).
    """
    A = np.nan_to_num(payoff_row.matrix, nan=0.0)
    strategies_x = list(payoff_row.strategies)
    if payoff_col is None:
        B = A.T
        strategies_y = strategies_x
    else:
        B = np.nan_to_num(payoff_col.matrix, nan=0.0)
        strategies_y = list(payoff_col.strategies)

    nx, ny = len(strategies_x), len(strategies_y)
    x_init = _uniform(nx) if x0 is None else _dict_to_vector(x0, strategies_x)
    y_init = _uniform(ny) if y0 is None else _dict_to_vector(y0, strategies_y)

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        x, y = z[:nx], z[nx:]
        ay = A @ y
        xb = x @ B
        avg_x = float(x @ ay)
        avg_y = float(xb @ y)
        dx = x * (ay - avg_x)
        dy = y * (xb - avg_y)
        return np.concatenate([dx, dy])

    times = np.linspace(0.0, t_max, n_steps)
    z_init = np.concatenate([x_init, y_init])
    sol = solve_ivp(rhs, (0.0, t_max), z_init, t_eval=times, method="RK45")
    z = np.clip(sol.y.T, 0.0, None)
    x = z[:, :nx]
    y = z[:, nx:]
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    return ReplicatorTrajectory(
        times=sol.t, x=x, strategies_x=strategies_x, y=y, strategies_y=strategies_y
    )


def _dict_to_vector(d: dict[str, float], strategies: list[str]) -> np.ndarray:
    v = np.array([d.get(s, 0.0) for s in strategies], dtype=float)
    total = v.sum()
    if total <= 0:
        raise ValueError("initial mixture must have at least one positive entry")
    return v / total


def final_mixed_strategy(
    trajectory: ReplicatorTrajectory, eps: float = 1e-3, population: str = "x"
) -> dict[str, float]:
    """Returns the final mixture of a replicator dynamics trajectory, keeping
    only strategies with probability above ``eps``.

    Args:
        population: ``"x"`` for population 1, ``"y"`` for population 2 (only
            valid if ``trajectory`` came from :func:`asymmetric_replicator_dynamics`).
    """
    if population == "x":
        final, names = trajectory.x[-1], trajectory.strategies_x
    elif population == "y":
        if trajectory.y is None or trajectory.strategies_y is None:
            raise ValueError("trajectory has no population 'y' (symmetric case)")
        final, names = trajectory.y[-1], trajectory.strategies_y
    else:
        raise ValueError("population must be 'x' or 'y'")
    return {s: float(p) for s, p in zip(names, final) if p > eps}
