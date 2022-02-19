# -*- coding: utf-8 -*-
r"""Models basic types of utility functions.

Utility functions are at the core of negotiation. Agents engage in negotiations with the goal of maximizing some utility
function. In most cases, these utility functions are assumed to be known a-periori and static for the duration of a
single negotiations.

Notes:
    We try to allow for applications that do not necessary have these two assumptions in the following ways:

    * A utility_function *value* (\ `Value`\ ) can always represent represent a utility_function distribution over all
      possible utility_function values (\ `Distribution`\ ) or a `KnownValue` which is a real number.

    * The base class of all utility_function *functions* is
      `UtilityFunction` and is assumed to map outcomes (\ `Outcome` objects) to the aforementioned generic utility *values*
      (\ `Value` objects).

    * Utility functions can be constructed using any `Callable` which makes it possible to construct them so that
      they change depending on the context or the progression of the negotiation.


"""
from __future__ import annotations
from .base import *
from .mixins import *
from .protocols import *
from .preferences import *
from .base_ufun import *
from .crisp_ufun import *
from .prob_ufun import *
from .inv_ufun import *
from .discounted import *
from .crisp import *
from .prob import *
from .ops import *
from .complex import *
from .value_fun import *

__all__ = (
    base.__all__
    + mixins.__all__
    + protocols.__all__
    + preferences.__all__
    + base_ufun.__all__
    + crisp_ufun.__all__
    + prob_ufun.__all__
    + inv_ufun.__all__
    + discounted.__all__
    + crisp.__all__
    + prob.__all__
    + ops.__all__
    + complex.__all__
    + value_fun.__all__
)
