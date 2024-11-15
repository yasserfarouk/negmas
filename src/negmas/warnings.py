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
    ...


def warn(message, category=NegmasWarning, stacklevel=2, source=None):
    """Issues a warning to the user. Defaults to `NegmasWarning` and stacklevel of 2."""
    return warnings.warn(message, category, stacklevel, source)


def deprecated(message):
    """Issues a deprecation warning"""
    return warnings.warn(
        message, category=DeprecationWarning, stacklevel=2, source=None
    )


class NegmasBridgePathWarning(NegmasWarning):
    ...


class NegmasBridgeProcessWarning(NegmasWarning):
    ...


class NegmasInfiniteNegotiationWarning(NegmasWarning):
    ...


class NegmasStepAndTimeLimitWarning(NegmasWarning):
    ...


class NegmasCannotStartNegotiation(NegmasWarning):
    ...


class NegmasBrdigeParsingWarning(NegmasWarning):
    ...


class NegmasSarializationWarning(NegmasWarning):
    ...


class NegmasIOWarning(NegmasWarning):
    ...


class NegmasShutdownWarning(NegmasWarning):
    ...


class NegmasWorldConfigWarning(NegmasWarning):
    ...


class NegmasImportWarning(NegmasWarning, ImportWarning):
    ...


class NegmasUnusedValueWarning(NegmasWarning):
    ...


class NegmasMemoryWarning(NegmasWarning):
    ...


class NegmasCaughtExceptionWarning(NegmasWarning):
    ...


class NegmasVisualizationWarning(NegmasWarning):
    ...


class NegmasNoResponseWarning(NegmasWarning):
    ...


class NegmasLoggingWarning(NegmasWarning):
    ...


class NegmasNumericWarning(NegmasWarning):
    ...


class NegmasSpeedWarning(NegmasWarning):
    ...


class NegmasIgnoredValueWarning(NegmasWarning):
    ...


class NegmasDoubleAssignmentWarning(NegmasWarning):
    ...


class NegmasUnexpectedValueWarning(NegmasWarning):
    ...


class NegmasSlowOperation(NegmasWarning):
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
