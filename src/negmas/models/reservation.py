"""
Reservation-value modeling.

Models that estimate the opponent's *reservation value* — the minimum utility the
opponent still deems an acceptable agreement (the utility of its best alternative
to no agreement). This corresponds to attribute §5.1.1 ("Learning the acceptance
strategy by estimating the reservation value") of the opponent-modeling taxonomy
of Baarslag, Hendrikx, Hindriks & Jonker, *Learning about the opponent in
automated bilateral negotiation: a comprehensive survey of opponent modeling
techniques*, JAAMAS 30:849-898 (2016).

The survey's methods all rest on the idea that an agent stops conceding near its
reservation value as the deadline approaches. Two representative families are
provided here:

- `ConcessionExtrapolatingReservationModel` — the regression family (Hou [85];
  Yu et al. [204]): fit and extrapolate the opponent's concession curve to the
  deadline; the utility it flattens toward is the reservation value.
- `BayesianReservationValueModel` — Zeng & Sycara [205,206]: maintain a Bayesian
  posterior over a set of candidate reservation values, updated each round from
  how consistent the observed concession is with each candidate.

Both are framework-agnostic: they consume ``(relative_time, opponent_utility)``
observations, exactly like :mod:`negmas.models.deadline`.

*AI Generated (reservation-value opponent models from Baarslag et al. 2016 §5.1.1).*
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "ReservationValueModel",
    "ConcessionExtrapolatingReservationModel",
    "BayesianReservationValueModel",
]


class ReservationValueModel(ABC):
    """Abstract base class for models that estimate the opponent's reservation value.

    A reservation-value model observes, round by round, the (relative) time of each
    opponent offer together with an estimate of the utility that offer yields *to the
    opponent* (or any monotone proxy of the opponent's concession). From this
    concession trace it estimates the utility floor below which the opponent would
    rather walk away — its reservation value.

    Relative time is assumed to run from ``0`` (start) to ``1`` (deadline), and
    utilities are assumed normalized to ``[0, 1]``.

    *AI Generated.*
    """

    def __init__(self):
        """Initializes the reservation-value model."""
        self._times: list[float] = []
        self._utils: list[float] = []

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records one observation of the opponent's concession.

        Args:
            relative_time: The relative time (0-1) at which the offer was received.
            opponent_utility: An estimate (or monotone proxy) of the utility of the offer
                to the opponent.
        """
        self._times.append(float(relative_time))
        self._utils.append(float(opponent_utility))

    @abstractmethod
    def predict_reservation(self) -> float:
        """Returns the estimated reservation value of the opponent.

        Returns:
            float: Estimated reservation value in ``[0, 1]``; ``float('nan')`` if there is
            not yet enough evidence to estimate one.
        """
        raise NotImplementedError()


class ConcessionExtrapolatingReservationModel(ReservationValueModel):
    """Estimates the reservation value by extrapolating the concession curve.

    A straight line is fit (least squares) to the observed ``(relative_time,
    opponent_utility)`` trace and extrapolated to the deadline (``t = 1``); the
    utility there is taken as the reservation value. This is the simplest instance of
    the regression family of survey §5.1.1 (Hou [85]; Yu et al. [204]), which assume
    the opponent uses a time-dependent concession tactic.

    *AI Generated (regression-based reservation-value estimator).*
    """

    def __init__(self, min_observations: int = 3):
        """Initializes the extrapolating reservation-value model.

        Args:
            min_observations: Minimum number of observations required before a value is
                estimated. With fewer observations ``predict_reservation`` returns ``nan``.
        """
        super().__init__()
        self.min_observations = min_observations

    def predict_reservation(self) -> float:
        """Fits and extrapolates the concession line to the deadline (``t = 1``).

        Returns:
            float: The estimated reservation value, clamped to ``[0, 1]``.
        """
        if len(self._times) < self.min_observations:
            return float("nan")
        t = np.asarray(self._times, dtype=float)
        u = np.asarray(self._utils, dtype=float)
        slope, intercept = np.polyfit(t, u, 1)
        rv = slope * 1.0 + intercept
        return float(min(1.0, max(0.0, rv)))


class BayesianReservationValueModel(ReservationValueModel):
    """Estimates the reservation value via Bayesian learning — Zeng & Sycara [205,206].

    A grid of candidate reservation values ``v`` is maintained with a posterior
    probability each. The opponent is assumed to concede roughly linearly from its
    first observed offer toward its reservation value by the deadline, so the expected
    utility at time ``t`` under candidate ``v`` is ``e_v(t) = v + (u_0 - v)(1 - t)``.
    Each observation updates the posterior by a Gaussian likelihood around ``e_v(t)``
    (accumulated in log-space). The estimate is the posterior-weighted mean candidate.

    Args:
        candidates: Number of candidate reservation values in the ``[0, 1]`` grid.
        sigma: Standard deviation of the observation-likelihood Gaussian.
        min_observations: Minimum observations before an estimate is returned.

    *AI Generated (Zeng & Sycara Bayesian reservation-value estimator).*
    """

    def __init__(
        self, candidates: int = 21, sigma: float = 0.15, min_observations: int = 2
    ):
        """Initializes the Bayesian reservation-value model."""
        super().__init__()
        self.sigma = sigma
        self.min_observations = min_observations
        self._grid = np.linspace(0.0, 1.0, candidates)
        self._logpost = np.zeros(candidates)  # uniform prior in log-space

    def update(self, relative_time: float, opponent_utility: float) -> None:
        """Records an observation and updates the log-posterior over candidates."""
        super().update(relative_time, opponent_utility)
        u0 = self._utils[0]
        t = float(relative_time)
        u = float(opponent_utility)
        # Expected utility under each candidate reservation value at this time.
        expected = self._grid + (u0 - self._grid) * (1.0 - t)
        self._logpost += -0.5 * ((u - expected) / self.sigma) ** 2
        # A candidate is impossible if the opponent offered strictly below it.
        self._logpost[self._grid > u + 1e-9] += -1e6

    def predict_reservation(self) -> float:
        """Returns the posterior-weighted mean reservation value.

        Returns:
            float: The estimated reservation value in ``[0, 1]``; ``nan`` before enough
            observations are gathered.
        """
        if len(self._times) < self.min_observations:
            return float("nan")
        m = float(np.max(self._logpost))
        post = np.exp(self._logpost - m)
        s = float(post.sum())
        if s <= 0:
            return float("nan")
        post = post / s
        return float((post * self._grid).sum())
