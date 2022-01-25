# -*- coding: utf-8 -*-
r"""Models basic types of utility functions.

Utility functions are at the core of negotiation. Agents engage in negotiations with the goal of maximizing some utility
function. In most cases, these utility functions are assumed to be known a-periori and static for the duration of a
single negotiations.

Notes:
    We try to allow for applications that do not necessary have these two assumptions in the following ways:

    * A utility_function *value* (\ `UtilityValue`\ ) can always represent represent a utility_function distribution over all
      possible utility_function values (\ `Distribution`\ ) or a `KnownUtilityValue` which is a real number.

    * The base class of all utility_function *functions* is
      `UtilityFunction` and is assumed to map outcomes (\ `Outcome` objects) to the aforementioned generic utility *values*
      (\ `UtilityValue` objects).

    * Utility functions can be constructed using any `Callable` which makes it possible to construct them so that
      they change depending on the context or the progression of the negotiation.


"""
from __future__ import annotations
from .base import *
from .mixins import *
from .protocols import *
from .preferences import *
from .ufun import *
from .complex import *
from .discounted import *
from .mapping import *
from .linear import *
from .nonlinear import *
from .ops import *
from .probabilistic import *
from .random_ufun import *
from .const import *

__all__ = (
    base.__all__
    + protocols.__all__
    + preferences.__all__
    + ufun.__all__
    + complex.__all__
    + discounted.__all__
    + mapping.__all__
    + linear.__all__
    + nonlinear.__all__
    + ops.__all__
    + probabilistic.__all__
    + random_ufun.__all__
    + const.__all__
    + mixins.__all__
)
