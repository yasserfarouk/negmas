"""
Opponent offering-strategy modeling.

Models that predict the opponent's *offering strategy* (the survey's "bidding strategy") —
the function mapping the negotiation state to the opponent's next offer (or, more
tractably, to the utility the opponent's own offers will yield over time). This corresponds
to attribute §5.4 ("Learning the bidding strategy") of the opponent-modeling taxonomy of
Baarslag, Hendrikx, Hindriks & Jonker, *Learning about the opponent in automated bilateral
negotiation: a comprehensive survey of opponent modeling techniques*, JAAMAS 30:849–898
(2016). We use ``offering`` in the class names to match negmas naming conventions (e.g.
:mod:`negmas.gb.components.offering`).

The survey groups these models into *regression analysis* and *time-series forecasting*
techniques. The concrete example below (:class:`TimeSeriesOfferingModel`) belongs to the
time-series-forecasting family: it forecasts the concession trajectory of the opponent from
the sequence of utilities of its past offers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from negmas.models.future import FutureUtilityRegressor

__all__ = [
    "OpponentOfferingModel",
    "TimeSeriesOfferingModel",
    "PolynomialOfferingModel",
    "DerivativeOfferingModel",
    "MarkovChainOfferingModel",
]


class OpponentOfferingModel(ABC):
    """Abstract base class for models that predict the opponent's offering strategy.

    The model observes the utility of each opponent offer (as estimated by an opponent
    utility model, or any agreed monotone proxy) against relative time, and predicts the
    utility the opponent's offers will reach at a future time — i.e. how far the opponent
    is expected to have conceded by then.

    Relative time is assumed to run from ``0`` (start) to ``1`` (deadline).
    """

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records one observation of an opponent offer.

        Args:
            relative_time: The relative time (0-1) at which the offer was received.
            opponent_utility: An estimate (or monotone proxy) of the utility of the offer.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_utility(self, relative_time: float) -> float:
        """Predicts the utility of the opponent's offer at ``relative_time``.

        Args:
            relative_time: The relative time (0-1) to forecast for.

        Returns:
            float: The forecast utility of the opponent's offer at that time.
        """
        raise NotImplementedError()


class TimeSeriesOfferingModel(OpponentOfferingModel):
    """Forecasts the opponent's concession curve via Gaussian-process regression.

    This is a time-series-forecasting offering-strategy model (survey §5.4). It fits a
    regressor of *utility-of-opponent-offer* against *relative time* and uses it to
    forecast how the opponent will offer in the future. It re-uses
    :class:`~negmas.models.future.FutureUtilityRegressor` (a Gaussian-process regressor)
    rather than reinventing the regression machinery.
    """

    def __init__(self, min_observations: int = 3, **regressor_kwargs):
        """Initializes the time-series offering model.

        Args:
            min_observations: Minimum observations before regression is used. Below this,
                :meth:`predict_utility` falls back to the most recent observed utility.
            **regressor_kwargs: Passed to :class:`FutureUtilityRegressor`.
        """
        self.min_observations = min_observations
        self._regressor_kwargs = regressor_kwargs
        self._times: list[float] = []
        self._utils: list[float] = []
        self._regressor: FutureUtilityRegressor | None = None
        self._fitted_n = 0

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records an observation of an opponent offer."""
        self._times.append(float(relative_time))
        self._utils.append(float(opponent_utility))

    def _ensure_fitted(self) -> bool:
        if len(self._times) < self.min_observations:
            return False
        if self._regressor is None or self._fitted_n != len(self._times):
            self._regressor = FutureUtilityRegressor(**self._regressor_kwargs)
            self._regressor.fit(self._times, self._utils)
            self._fitted_n = len(self._times)
        return True

    def predict_utility(self, relative_time: float) -> float:
        """Forecasts the opponent's offer utility at ``relative_time``.

        Args:
            relative_time: The relative time (0-1) to forecast for.

        Returns:
            float: Forecast utility; falls back to the last observed utility when there is
            insufficient data to fit the regressor.
        """
        if not self._ensure_fitted():
            return self._utils[-1] if self._utils else 0.0
        assert self._regressor is not None
        return float(self._regressor.predict_utility(np.array([relative_time]))[0])


class PolynomialOfferingModel(OpponentOfferingModel):
    """Forecasts the opponent's concession curve by polynomial regression.

    A regression-analysis offering-strategy model (survey §5.4.1). It fits a
    polynomial of a given ``degree`` to the ``(relative_time, opponent_utility)``
    trace (least squares) and evaluates it at the queried time. This is the
    polynomial-interpolation estimator compared by Papaioannou et al. [151,153]
    (they also evaluate cubic splines and a genetic-algorithm fit).

    Args:
        degree: Degree of the fitted polynomial (``3`` ≈ the cubic used in the
            survey). The effective degree is capped at ``n_observations - 1``.
        min_observations: Minimum observations before regression is used; below it
            :meth:`predict_utility` falls back to the last observed utility.
            Defaults to ``degree + 1``.

    *AI Generated (Papaioannou et al. polynomial offer forecaster).*
    """

    def __init__(self, degree: int = 3, min_observations: int | None = None):
        """Initializes the polynomial offering model."""
        self.degree = degree
        self.min_observations = (
            min_observations if min_observations is not None else degree + 1
        )
        self._times: list[float] = []
        self._utils: list[float] = []

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records an observation of an opponent offer."""
        self._times.append(float(relative_time))
        self._utils.append(float(opponent_utility))

    def predict_utility(self, relative_time: float) -> float:
        """Forecasts the opponent's offer utility at ``relative_time``.

        Returns:
            float: Forecast utility clamped to ``[0, 1]``; falls back to the last
            observed utility when there is insufficient data to fit.
        """
        if len(self._times) < self.min_observations:
            return self._utils[-1] if self._utils else 0.0
        deg = min(self.degree, len(self._times) - 1)
        coeffs = np.polyfit(np.asarray(self._times), np.asarray(self._utils), deg)
        val = float(np.polyval(coeffs, relative_time))
        return min(1.0, max(0.0, val))


class DerivativeOfferingModel(OpponentOfferingModel):
    """Forecasts the opponent's offers from the derivatives of its concession curve.

    A time-series-forecasting offering-strategy model (survey §5.4.2) following
    Brzostowski et al. [29]. Finite differences of the observed concession curve
    estimate its local first and second derivatives, which are extrapolated
    (second-order Taylor step) to forecast future offers. The model also exposes a
    :meth:`time_influence` metric — the sign-consistency of the differences (survey
    eq. 8) — measuring how strongly the opponent behaves like a pure time-dependent
    tactician (``1`` = perfectly consistent, ``0`` = inconsistent).

    Args:
        min_observations: Minimum observations before extrapolation is used; below
            it :meth:`predict_utility` falls back to the last observed utility.

    *AI Generated (Brzostowski et al. derivative-based offer forecaster).*
    """

    def __init__(self, min_observations: int = 3):
        """Initializes the derivative offering model."""
        self.min_observations = min_observations
        self._times: list[float] = []
        self._utils: list[float] = []

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records an observation of an opponent offer."""
        self._times.append(float(relative_time))
        self._utils.append(float(opponent_utility))

    def predict_utility(self, relative_time: float) -> float:
        """Forecasts the opponent's offer utility via second-order extrapolation.

        Returns:
            float: Forecast utility clamped to ``[0, 1]``; falls back to the last
            observed utility (or a linear step) when data are scarce.
        """
        n = len(self._utils)
        if n == 0:
            return 0.0
        if n < 2:
            return self._utils[-1]
        t = np.asarray(self._times, dtype=float)
        u = np.asarray(self._utils, dtype=float)
        dt_last = relative_time - t[-1]
        if dt_last <= 0:
            return float(min(1.0, max(0.0, u[-1])))
        # First derivative from the last interval.
        d1 = (u[-1] - u[-2]) / (t[-1] - t[-2]) if t[-1] != t[-2] else 0.0
        if n < self.min_observations:
            val = u[-1] + d1 * dt_last
            return float(min(1.0, max(0.0, val)))
        # Second derivative from the last two intervals.
        d1_prev = (u[-2] - u[-3]) / (t[-2] - t[-3]) if t[-2] != t[-3] else d1
        span = t[-1] - t[-3]
        d2 = (d1 - d1_prev) / span if span != 0 else 0.0
        val = u[-1] + d1 * dt_last + 0.5 * d2 * dt_last**2
        return float(min(1.0, max(0.0, val)))

    def time_influence(self) -> float:
        """Returns the sign-consistency of the first differences (survey eq. 8).

        Returns:
            float: A value in ``[0, 1]``; ``1`` means every consecutive change had the
            same sign (consistent with a pure time-dependent tactic), ``0`` means the
            changes alternated. Returns ``nan`` with fewer than two differences.
        """
        u = np.asarray(self._utils, dtype=float)
        if u.size < 3:
            return float("nan")
        diffs = np.diff(u)
        signs = np.sign(diffs[np.abs(diffs) > 1e-12])
        if signs.size == 0:
            return 1.0
        pos = float(np.sum(signs > 0))
        neg = float(np.sum(signs < 0))
        return max(pos, neg) / (pos + neg)


class MarkovChainOfferingModel(OpponentOfferingModel):
    """Forecasts the opponent's offers with a Markov chain over concession states.

    A time-series-forecasting offering-strategy model (survey §5.4.2) following
    Narayanan & Jennings [140]. The opponent's utility is discretized into
    ``n_states`` bins; the observed sequence of states estimates a (Laplace-smoothed)
    transition matrix. The chain is then rolled forward from the current state to
    forecast the expected utility of a future offer.

    Args:
        n_states: Number of discrete utility states (bins over ``[0, 1]``).
        alpha: Laplace-smoothing pseudo-count added to every transition.

    *AI Generated (Narayanan & Jennings Markov-chain offer forecaster).*
    """

    def __init__(self, n_states: int = 10, alpha: float = 1.0):
        """Initializes the Markov-chain offering model."""
        self.n_states = n_states
        self.alpha = alpha
        self._times: list[float] = []
        self._utils: list[float] = []
        self._counts = np.zeros((n_states, n_states))
        self._prev_state: int | None = None

    def _state(self, utility: float) -> int:
        s = int(utility * self.n_states)
        return min(self.n_states - 1, max(0, s))

    def _center(self, state: int) -> float:
        return (state + 0.5) / self.n_states

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records an observation and its state transition."""
        self._times.append(float(relative_time))
        self._utils.append(float(opponent_utility))
        state = self._state(float(opponent_utility))
        if self._prev_state is not None:
            self._counts[self._prev_state, state] += 1
        self._prev_state = state

    def _transition_matrix(self) -> np.ndarray:
        m = self._counts + self.alpha
        return m / m.sum(axis=1, keepdims=True)

    def predict_next_utility(self) -> float:
        """Forecasts the expected utility of the opponent's *next* offer (one step).

        Returns:
            float: Expected utility of the next state; the last observed utility if no
            transitions have been seen yet.
        """
        if self._prev_state is None:
            return self._utils[-1] if self._utils else 0.0
        p = self._transition_matrix()[self._prev_state]
        centers = np.array([self._center(s) for s in range(self.n_states)])
        return float((p * centers).sum())

    def predict_utility(self, relative_time: float) -> float:
        """Forecasts the opponent's offer utility at ``relative_time``.

        The number of steps to roll the chain forward is estimated from the average
        time between observed offers and the remaining time to ``relative_time``.

        Returns:
            float: Expected forecast utility; the last observed utility when the time is
            not in the future or no transitions are known.
        """
        if self._prev_state is None or len(self._times) < 2:
            return self._utils[-1] if self._utils else 0.0
        dt = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        remaining = relative_time - self._times[-1]
        if remaining <= 0 or dt <= 0:
            return self._utils[-1]
        steps = max(1, int(round(remaining / dt)))
        p = self._transition_matrix()
        dist = np.zeros(self.n_states)
        dist[self._prev_state] = 1.0
        for _ in range(steps):
            dist = dist @ p
        centers = np.array([self._center(s) for s in range(self.n_states)])
        return float((dist * centers).sum())
