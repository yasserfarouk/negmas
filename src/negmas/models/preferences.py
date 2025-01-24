"""
Opponent's UFun modeling.
"""

from negmas.preferences.base_ufun import BaseUtilityFunction


__all__ = []


class OpponentUtilityFunction(BaseUtilityFunction):
    """Base class for opponent models that model the partner(s) utility function"""
