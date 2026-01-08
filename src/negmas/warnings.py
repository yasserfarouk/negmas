"""Module for warnings functionality."""

from __future__ import annotations

import warnings
from typing import Callable

from negmas.config import CONFIG_KEY_WARN_SLOW_OPS, negmas_config

__all__ = [
    "warn",
    "warn_if_slow",
    "deprecated",
    "NegmasWarning",
    "NegmasCannotStartNegotiation",
    "NegmasIgnoredValueWarning",
    "NegmasBridgePathWarning",
    "NegmasBridgeProcessWarning",
    "NegmasInfiniteNegotiationWarning",
    "NegmasStepAndTimeLimitWarning",
    "NegmasBrdigeParsingWarning",
    "NegmasSarializationWarning",
    "NegmasIOWarning",
    "NegmasShutdownWarning",
    "NegmasWorldConfigWarning",
    "NegmasImportWarning",
    "NegmasUnusedValueWarning",
    "NegmasMemoryWarning",
    "NegmasSpeedWarning",
    "NegmasCaughtExceptionWarning",
    "NegmasVisualizationWarning",
    "NegmasNoResponseWarning",
    "NegmasLoggingWarning",
    "NegmasNumericWarning",
    "NegmasDoubleAssignmentWarning",
    "NegmasUnexpectedValueWarning",
]


class NegmasWarning(UserWarning):
    """NegmasWarning implementation."""

    ...


def warn(message, category=NegmasWarning, stacklevel=2, source=None):
    """Issues a warning to the user. Defaults to `NegmasWarning` and stacklevel of 2."""
    return warnings.warn(message, category, stacklevel, source)


def deprecated(message, stacklevel=3):
    """Issues a deprecation warning"""
    return warnings.warn(
        message, category=DeprecationWarning, stacklevel=stacklevel, source=None
    )


class NegmasBridgePathWarning(NegmasWarning):
    """NegmasBridgePathWarning implementation."""

    ...


class NegmasBridgeProcessWarning(NegmasWarning):
    """NegmasBridgeProcessWarning implementation."""

    ...


class NegmasInfiniteNegotiationWarning(NegmasWarning):
    """NegmasInfiniteNegotiationWarning implementation."""

    ...


class NegmasStepAndTimeLimitWarning(NegmasWarning):
    """NegmasStepAndTimeLimitWarning implementation."""

    ...


class NegmasCannotStartNegotiation(NegmasWarning):
    """NegmasCannotStartNegotiation implementation."""

    ...


class NegmasBrdigeParsingWarning(NegmasWarning):
    """NegmasBrdigeParsingWarning implementation."""

    ...


class NegmasSarializationWarning(NegmasWarning):
    """NegmasSarializationWarning implementation."""

    ...


class NegmasIOWarning(NegmasWarning):
    """NegmasIOWarning implementation."""

    ...


class NegmasShutdownWarning(NegmasWarning):
    """NegmasShutdownWarning implementation."""

    ...


class NegmasWorldConfigWarning(NegmasWarning):
    """NegmasWorldConfigWarning implementation."""

    ...


class NegmasImportWarning(NegmasWarning, ImportWarning):
    """NegmasImportWarning implementation."""

    ...


class NegmasUnusedValueWarning(NegmasWarning):
    """NegmasUnusedValueWarning implementation."""

    ...


class NegmasMemoryWarning(NegmasWarning):
    """NegmasMemoryWarning implementation."""

    ...


class NegmasCaughtExceptionWarning(NegmasWarning):
    """NegmasCaughtExceptionWarning implementation."""

    ...


class NegmasVisualizationWarning(NegmasWarning):
    """NegmasVisualizationWarning implementation."""

    ...


class NegmasNoResponseWarning(NegmasWarning):
    """NegmasNoResponseWarning implementation."""

    ...


class NegmasLoggingWarning(NegmasWarning):
    """NegmasLoggingWarning implementation."""

    ...


class NegmasNumericWarning(NegmasWarning):
    """NegmasNumericWarning implementation."""

    ...


class NegmasSpeedWarning(NegmasWarning):
    """NegmasSpeedWarning implementation."""

    ...


class NegmasIgnoredValueWarning(NegmasWarning):
    """NegmasIgnoredValueWarning implementation."""

    ...


class NegmasDoubleAssignmentWarning(NegmasWarning):
    """NegmasDoubleAssignmentWarning implementation."""

    ...


class NegmasUnexpectedValueWarning(NegmasWarning):
    """NegmasUnexpectedValueWarning implementation."""

    ...


class NegmasSlowOperation(NegmasWarning):
    """NegmasSlowOperation implementation."""

    ...


def warn_if_slow(
    size: int | float,
    message: str = "Slow Operation Warning",
    op: Callable[[float], float] = lambda x: x,
) -> None:
    """
    Issues a warning for slow operations.

    Args:
        size: The size of this operation to be checked
        message: The message to print in the warning if the size is too large
        op: An operation to apply to the config value CONFIG_KEY_WARN_SLOW_OPS before comparison.

    Remarks:
        - The warning will be issued if ```op(size) >= config[CONFIG_KEY_WARN_SLOW_OPS]```
        - Default is to do no comparison (indicated by config[CONFIG_KEY_WARN_SLOW_OPS] <= 0)
        - See `negams_config` for details about getting the config value.
    """
    limit = float(negmas_config(CONFIG_KEY_WARN_SLOW_OPS, 0))
    if limit < 0.5:
        return
    nops = op(size)
    if nops < limit:
        return
    warnings.warn(
        f"{message}: n.ops={int(nops):,} (n={int(size):,}), limit={int(limit):,}",
        NegmasSlowOperation,
        2,
        None,
    )
