"""Modeling self's future prospects in the negotiation.
"""
__all__ = ["FutureUtilityRegressor"]
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class FutureUtilityRegressor:
    """Represents a regressor for own-utility for the future of the negotiation.

    Remarks:
        - We assume that the negotiation goes from time 0 to 1 (relative_time).
    """
    def __init__(self, regressor_factory=GaussianProcessRegressor, **kwargs):
        self.regressor = regressor_factory(**kwargs)
        self.inverse_regressor = regressor_factory(**kwargs)

    def fit(self, times, utils) -> "FutureUtilityRegressor":
        times = times.flatten().reshape((1, len(times)))
        utils = utils.flatten().reshape((1, len(utils)))
        self.regressor.fit(times, utils)
        self.inverse_regressor.fit(utils, times)

    def predict_utility(self, times) -> np.ndarray:
        times = times.flatten().reshape((1, len(times)))
        return self.regressor.predict(times)

    def predict_utility_prob(self, times, return_cov=False) -> np.ndarray:
        times = times.flatten().reshape((1, len(times)))
        return self.regressor.predict(times, return_std=True, return_cov=return_cov)

    def predict_time(self, utils) -> np.ndarray:
        utils = utils.flatten().reshape((1, len(utils)))
        return self.inverse_regressor.predict(utils)

    def predict_time_prob(self, utils, return_cov=False) -> np.ndarray:
        utils = utils.flatten().reshape((1, len(utils)))
        return self.inverse_regressor.predict(utils, return_std=True, return_cov=return_cov)
