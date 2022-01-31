import warnings

__all__ = [
    "warn",
    "deprecated",
    "NegmasWarning",
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
