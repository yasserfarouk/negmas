from __future__ import annotations

from negmas.negotiators import Negotiator

from .mixins import *
from .negotiator import Negotiator

__all__ = [
    "EvaluatorNegotiator",
    "RealComparatorNegotiator",
    "BinaryComparatorNegotiator",
    "NLevelsComparatorNegotiator",
    "RankerNegotiator",
    "RankerWithWeightsNegotiator",
    "SorterNegotiator",
]


class EvaluatorNegotiator(EvaluatorMixin, Negotiator):
    """
    A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `evaluate`.

    It has the `evaluate` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        EvaluatorMixin.init(self)


class RealComparatorNegotiator(RealComparatorMixin, Negotiator):
    """
    A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `compare_real`

    It has the `compare-real` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RealComparatorMixin.init(self)


class BinaryComparatorNegotiator(BinaryComparatorMixin, Negotiator):
    """
    A negotiator that can be asked to compare two outcomes using is_better. By default is just consults the ufun.

    To change that behavior, override `is_better`.

    It has the `compare-binary` capability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        BinaryComparatorMixin.init(self)


class NLevelsComparatorNegotiator(NLevelsComparatorMixin, Negotiator):
    """
    A negotiator that can be asked to compare two outcomes using compare_nlevels which returns the strength of
    the difference between two outcomes as an integer from [-n, n] in the C compare sense.
    By default is just consults the ufun.

    To change that behavior, override `compare_nlevels`.

    It has the `compare-nlevels` capability.

    """

    def __init__(self, *args, thresholds: list[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        NLevelsComparatorMixin.init(self)
        self.thresholds = thresholds  # type: ignore I am not sure why


class RankerWithWeightsNegotiator(RankerWithWeightsMixin, Negotiator):
    """
    A negotiator that can be asked to rank outcomes returning rank and weight. By default is just consults the ufun.

    To change that behavior, override `rank_with_weights`.

    It has the `rank-weighted` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RankerWithWeightsMixin.init(self)


class RankerNegotiator(RankerMixin, Negotiator):
    """
    A negotiator that can be asked to rank outcomes. By default is just consults the ufun.

    To change that behavior, override `rank`.

    It has the `rank` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RankerMixin.init(self)


class SorterNegotiator(SorterMixin, Negotiator):
    """
    A negotiator that can be asked to rank outcomes returning rank without weight.
    By default is just consults the ufun.

    To change that behavior, override `sort`.

    It has the `sort` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SorterMixin.init(self)
