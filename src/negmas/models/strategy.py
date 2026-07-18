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

__all__ = ["OpponentOfferingModel", "TimeSeriesOfferingModel"]


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
