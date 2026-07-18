"""Issue-weight opponent models (survey §5.3.1) for SAO negotiations.

Re-exports the implementations from :mod:`negmas.gb.components.models.weights`
(the single source of truth) so they are available under the SAO component
namespace too. SAO components are GB components (``negmas.sao.components.base``
re-exports ``negmas.gb.components.base``), so the same classes work in both.

*AI Generated (SAO re-export of the §5.3.1 issue-weight opponent models).*
"""

from __future__ import annotations

from negmas.gb.components.models.weights import (
    ConcessionRatioUFunModel,
    KDEWeightUFunModel,
    ValueDifferenceUFunModel,
)

__all__ = ["ConcessionRatioUFunModel", "ValueDifferenceUFunModel", "KDEWeightUFunModel"]
