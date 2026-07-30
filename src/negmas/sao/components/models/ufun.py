"""Utility function opponent models for SAO negotiations.

Re-exports the implementations from :mod:`negmas.gb.components.models.ufun`
(the single source of truth) so they are available under the SAO component
namespace too. SAO components are GB components (``negmas.sao.components.base``
re-exports ``negmas.gb.components.base``), so the same classes work in both.

``FrequencyUFunModel`` and ``FrequencyLinearUFunModel`` used to be unimplemented
stubs here (``eval`` raised ``NotImplementedError``); the GB module already has
full implementations, so this now re-exports those instead of duplicating them.

*AI Generated (SAO re-export of the ufun opponent models).*
"""

from __future__ import annotations

from negmas.gb.components.models.ufun import (
    FrequencyLinearUFunModel,
    FrequencyUFunModel,
    UFunModel,
    ZeroSumModel,
)

__all__ = [
    "UFunModel",
    "FrequencyUFunModel",
    "FrequencyLinearUFunModel",
    "ZeroSumModel",
]
