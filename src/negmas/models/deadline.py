"""
Deadline modeling.

Models that estimate the opponent's *deadline* — the (relative) time before which the
opponent must reach an agreement to prefer it over disagreement. This corresponds to
attribute §5.2 ("Learning the deadline") of the opponent-modeling taxonomy of Baarslag,
Hendrikx, Hindriks & Jonker, *Learning about the opponent in automated bilateral
negotiation: a comprehensive survey of opponent modeling techniques*, JAAMAS 30:849–898
(2016).

As the survey notes, the deadline is tightly coupled with the reservation value: an agent
is likely to concede strongly near its deadline. Most deadline estimators therefore reuse
the machinery of reservation-value estimation (§5.1.1) — e.g. extrapolating the opponent's
concession curve until it reaches the estimated reservation value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["DeadlineModel", "ConcessionExtrapolatingDeadlineModel"]


class DeadlineModel(ABC):
    """Abstract base class for models that estimate the opponent's deadline.

    A deadline model observes, round by round, the (relative) time of each opponent offer
    together with an estimate of the utility that offer yields *to the opponent* (or any
    monotone proxy of the opponent's concession, such as the utility of the offer to us).
    From this concession trace it estimates the relative time at which the opponent will
    concede down to its reservation value — i.e. its deadline.

    Relative time is assumed to run from ``0`` (start) to ``1`` (own deadline).
    """

    def __init__(self, reserved_value: float = 0.0):
        """Initializes the deadline model.

        Args:
            reserved_value: The opponent's estimated reservation value (utility below which
                the opponent prefers disagreement). The deadline is estimated as the time at
                which the opponent's concession curve reaches this value.
        """
        self.reserved_value = reserved_value
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
    def predict_deadline(self) -> float:
        """Returns the estimated relative-time deadline of the opponent.

        Returns:
            float: Estimated deadline as a relative time. Values ``>= 1`` mean the opponent
            is not expected to hit its deadline before ours (no earlier deadline detected);
            ``float('inf')`` means there is not yet enough evidence to estimate one.
        """
        raise NotImplementedError()


class ConcessionExtrapolatingDeadlineModel(DeadlineModel):
    """Estimates the deadline by extrapolating the opponent's concession curve.

    A straight line is fit (least squares) to the observed ``(relative_time,
    opponent_utility)`` trace, then extrapolated to find the time at which the opponent's
    utility reaches its reservation value. This is the simplest instance of the family of
    methods described in survey §5.1.1/§5.2 (Hou; Yu et al.; Sim et al.), which assume the
    opponent uses a time-dependent concession tactic.
    """

    def __init__(self, reserved_value: float = 0.0, min_observations: int = 3):
        """Initializes the extrapolating deadline model.

        Args:
            reserved_value: The opponent's estimated reservation value.
            min_observations: Minimum number of observations required before a deadline is
                estimated. With fewer observations ``predict_deadline`` returns ``inf``.
        """
        super().__init__(reserved_value=reserved_value)
        self.min_observations = min_observations

    def predict_deadline(self) -> float:
        """Fits and extrapolates the concession line to the reservation value.

        Returns:
            float: The estimated relative-time deadline (see :meth:`DeadlineModel.predict_deadline`).
        """
        if len(self._times) < self.min_observations:
            return float("inf")
        t = np.asarray(self._times, dtype=float)
        u = np.asarray(self._utils, dtype=float)
        # Fit u = slope * t + intercept.
        slope, intercept = np.polyfit(t, u, 1)
        if slope >= -1e-9:
            # Opponent is not conceding (flat or hardening): no earlier deadline detectable.
            return float("inf")
        # Solve slope * t_deadline + intercept = reserved_value.
        t_deadline = (self.reserved_value - intercept) / slope
        if t_deadline <= t[-1]:
            # Extrapolation lies in the past: opponent is already at/below reservation.
            return float(t[-1])
        return float(t_deadline)
