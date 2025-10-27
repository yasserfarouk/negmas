"""
Modeling self's future prospects in the negotiation.
"""

from __future__ import annotations

__all__ = ["FutureUtilityRegressor"]
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class FutureUtilityRegressor:
    """Represents a regressor for own-utility for the future of the negotiation.

    Remarks:
        - We assume that the negotiation goes from time 0 to 1 (relative_time).
    """

    def __init__(self, regressor_factory=GaussianProcessRegressor, **kwargs):
        """Initialize the instance.

        Args:
            regressor_factory: Regressor factory.
            **kwargs: Additional keyword arguments.
        """
        self.regressor = regressor_factory(**kwargs)
        self.inverse_regressor = regressor_factory(**kwargs)

    def fit(self, times, utils) -> FutureUtilityRegressor:
        """Fit.

        Args:
            times: Times.
            utils: Utils.

        Returns:
            FutureUtilityRegressor: The result.
        """
        times, utils = np.array(times), np.array(utils)
        times = times.flatten().reshape((len(times), 1))
        utils = utils.flatten().reshape((len(utils), 1))
        self.regressor.fit(times, utils)
        self.inverse_regressor.fit(utils, times)
        return self.inverse_regressor

    def predict_utility(self, times) -> np.ndarray:
        """Predict utility.

        Args:
            times: Times.

        Returns:
            np.ndarray: The result.
        """
        times = np.array(times)
        times = times.flatten().reshape((len(times), 1))
        return self.regressor.predict(times).flatten()

    def predict_utility_prob(self, times, return_cov=False) -> np.ndarray:
        """Predict utility prob.

        Args:
            times: Times.
            return_cov: Return cov.

        Returns:
            np.ndarray: The result.
        """
        times = times.flatten().reshape((1, len(times)))
        return self.regressor.predict(times, return_std=True, return_cov=return_cov)

    def predict_time(self, utils) -> np.ndarray:
        """Predict time.

        Args:
            utils: Utils.

        Returns:
            np.ndarray: The result.
        """
        utils = np.array(utils)
        utils = utils.flatten().reshape((1, len(utils)))
        return self.inverse_regressor.predict(utils).flatten()

    def predict_time_prob(self, utils, return_cov=False) -> np.ndarray:
        """Predict time prob.

        Args:
            utils: Utils.
            return_cov: Return cov.

        Returns:
            np.ndarray: The result.
        """
        utils = np.array(utils)
        utils = utils.flatten().reshape((1, len(utils)))
        return self.inverse_regressor.predict(
            utils, return_std=True, return_cov=return_cov
        )
