"""Python-native Genius BOA Negotiator implementations.

This module contains Python implementations of classic Genius agents as BOANegotiator
subclasses. These negotiators use the transcompiled Genius BOA components and do NOT
require the Java Genius bridge.

The naming convention uses a 'G' prefix to distinguish these Python-native implementations
from the Java-bridge versions in gnegotiators.py.

References:
    - ANAC (Automated Negotiating Agents Competition): https://ii.tudelft.nl/ANAC/
    - Genius: https://ii.tudelft.nl/genius/
"""

from __future__ import annotations

from negmas.gb.negotiators.modular.boa import BOANegotiator
from negmas.gb.components.genius import (
    # Acceptance policies
    GACNext,
    GACCombi,
    GACConst,
    GACPrevious,
    GACTrue,
    GACCombiMax,
    GACCombiMaxInWindow,
    # Offering policies
    GTimeDependentOffering,
    GBoulwareOffering,
    GConcederOffering,
    GLinearOffering,
    GHardlinerOffering,
    GRandomOffering,
    # Opponent models
    GDefaultModel,
    GHardHeadedFrequencyModel,
    GSmithFrequencyModel,
    GAgentXFrequencyModel,
)

__all__ = [
    # Classic time-dependent agents
    "GBoulware",
    "GConceder",
    "GLinear",
    "GHardliner",
    # ANAC competition winners/notable agents
    "GHardHeaded",
    "GAgentK",
    "GAgentSmith",
    "GNozomi",
    "GFSEGA",
    "GCUHKAgent",
    "GAgentLG",
    "GAgentX",
    # Utility agents
    "GRandom",
]


# =============================================================================
# Classic Time-Dependent Agents
# =============================================================================


class GBoulware(BOANegotiator):
    """
    Python-native Boulware negotiator.

    A time-dependent agent with e < 1 (typically e=0.2), which concedes slowly
    and only makes significant concessions near the deadline. This is a conservative
    negotiation strategy.

    The Boulware strategy is named after the labor negotiator Lemuel Boulware,
    known for making "take it or leave it" offers.

    Uses:
        - Offering: GBoulwareOffering (e=0.2)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        offering = GBoulwareOffering()
        acceptance = GACNext(offering_policy=offering)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GConceder(BOANegotiator):
    """
    Python-native Conceder negotiator.

    A time-dependent agent with e > 1 (typically e=2.0), which concedes quickly
    early in the negotiation. This is an accommodating negotiation strategy.

    Uses:
        - Offering: GConcederOffering (e=2.0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        offering = GConcederOffering()
        acceptance = GACNext(offering_policy=offering)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GLinear(BOANegotiator):
    """
    Python-native Linear negotiator.

    A time-dependent agent with e = 1, which concedes at a constant rate
    throughout the negotiation.

    Uses:
        - Offering: GLinearOffering (e=1.0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        offering = GLinearOffering()
        acceptance = GACNext(offering_policy=offering)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GHardliner(BOANegotiator):
    """
    Python-native Hardliner negotiator.

    A time-dependent agent with e = 0, which never concedes and always offers
    its best outcome. This is the most aggressive negotiation strategy.

    Uses:
        - Offering: GHardlinerOffering (e=0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        offering = GHardlinerOffering()
        acceptance = GACNext(offering_policy=offering)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


# =============================================================================
# ANAC Competition Winners and Notable Agents
# =============================================================================


class GHardHeaded(BOANegotiator):
    """
    Python-native HardHeaded agent (ANAC 2011 Winner).

    HardHeaded (KLH) won the ANAC 2011 competition. It uses a Boulware-style
    time-dependent offering strategy combined with frequency-based opponent
    modeling to estimate the opponent's preferences.

    The agent tracks which issues remain unchanged in opponent offers to infer
    which issues are most important to the opponent.

    Uses:
        - Offering: GTimeDependentOffering (e=0.2, Boulware-style)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - van Krimpen, T.,";"; man, D.;"; Hindriks, K. (2011). "HardHeaded".
          ANAC 2011.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize HardHeaded agent.

        Args:
            e: The time-dependency exponent (default 0.2 for Boulware behavior)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACNext(offering_policy=offering)
        model = GHardHeadedFrequencyModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentK(BOANegotiator):
    """
    Python-native AgentK (ANAC 2010).

    AgentK was one of the top performers in ANAC 2010. It uses a time-dependent
    offering strategy with adaptive acceptance based on a combination of
    conditions (AC_Combi).

    Uses:
        - Offering: GTimeDependentOffering (e=0.2)
        - Acceptance: GACCombi (combined acceptance conditions)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Kawaguchi, S.; Fujita, K.; Ito, T. (2010). "AgentK: Compromising
          Strategy based on Estimated Maximum Utility for Automated Negotiating
          Agents". ANAC 2010.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize AgentK.

        Args:
            e: The time-dependency exponent (default 0.2)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACCombi(offering_policy=offering)
        model = GDefaultModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentSmith(BOANegotiator):
    """
    Python-native AgentSmith (ANAC 2010).

    AgentSmith was a notable agent in ANAC 2010, using a time-dependent strategy
    with frequency-based opponent modeling. It uses AC_Const acceptance which
    accepts any offer above a fixed utility threshold.

    Uses:
        - Offering: GTimeDependentOffering (e=0.3)
        - Acceptance: GACConst (accepts above fixed threshold)
        - Model: GSmithFrequencyModel (Smith-style frequency model)

    References:
        - de Jonge, D. (2010). "AgentSmith". ANAC 2010.
    """

    def __init__(self, e: float = 0.3, c: float = 0.8, **kwargs):
        """
        Initialize AgentSmith.

        Args:
            e: The time-dependency exponent (default 0.3)
            c: The constant acceptance threshold (default 0.8)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACConst(c=c)
        model = GSmithFrequencyModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GNozomi(BOANegotiator):
    """
    Python-native Nozomi agent (ANAC 2010).

    Nozomi was a competitive agent in ANAC 2010. It uses a Boulware-style
    time-dependent offering with AC_Previous acceptance, which accepts if
    the opponent's offer is better than their previous offer.

    Uses:
        - Offering: GTimeDependentOffering (e=0.2)
        - Acceptance: GACPrevious (accepts if better than opponent's previous offer)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Fujita, K. (2010). "Nozomi". ANAC 2010.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize Nozomi.

        Args:
            e: The time-dependency exponent (default 0.2)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACPrevious()
        model = GDefaultModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GFSEGA(BOANegotiator):
    """
    Python-native FSEGA agent (ANAC 2010).

    AgentFSEGA (Faculty of Computer Science Agent) from ANAC 2010 uses a
    conceding strategy with constant threshold acceptance.

    Uses:
        - Offering: GConcederOffering (e=2.0)
        - Acceptance: GACConst (accepts above fixed threshold)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Zaharia, G.; et al. (2010). "AgentFSEGA". ANAC 2010.
    """

    def __init__(self, c: float = 0.7, **kwargs):
        """
        Initialize FSEGA.

        Args:
            c: The constant acceptance threshold (default 0.7)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GConcederOffering()
        acceptance = GACConst(c=c)
        model = GDefaultModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GCUHKAgent(BOANegotiator):
    """
    Python-native CUHKAgent (ANAC 2012 Winner).

    CUHKAgent won ANAC 2012. It uses a sophisticated time-dependent strategy
    with combined acceptance conditions. The agent is particularly good at
    handling domains with discount factors.

    Uses:
        - Offering: GTimeDependentOffering (e=0.15, more conservative)
        - Acceptance: GACCombi (combined acceptance conditions)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - Hao, J.; Leung, H. (2012). "CUHKAgent: A Strategy for Bilateral
          Multi-issue Negotiation". ANAC 2012.
    """

    def __init__(self, e: float = 0.15, **kwargs):
        """
        Initialize CUHKAgent.

        Args:
            e: The time-dependency exponent (default 0.15 for conservative behavior)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACCombi(offering_policy=offering)
        model = GHardHeadedFrequencyModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentLG(BOANegotiator):
    """
    Python-native AgentLG (ANAC 2012).

    AgentLG was a notable agent in ANAC 2012. It uses a time-dependent
    strategy with AC_CombiMax acceptance which considers both the opponent's
    best offer and the time remaining.

    Uses:
        - Offering: GTimeDependentOffering (e=0.25)
        - Acceptance: GACCombiMax (maximum-based combined acceptance)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - ANAC 2012 AgentLG.
    """

    def __init__(self, e: float = 0.25, **kwargs):
        """
        Initialize AgentLG.

        Args:
            e: The time-dependency exponent (default 0.25)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACCombiMax(offering_policy=offering)
        model = GHardHeadedFrequencyModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentX(BOANegotiator):
    """
    Python-native AgentX (ANAC 2015).

    AgentX was a notable agent in ANAC 2015. It uses adaptive time-dependent
    offering with window-based acceptance and exponential smoothing opponent
    modeling.

    Uses:
        - Offering: GTimeDependentOffering (e=0.3, adaptive)
        - Acceptance: GACCombiMaxInWindow (window-based maximum acceptance)
        - Model: GAgentXFrequencyModel (exponential smoothing opponent model)

    References:
        - ANAC 2015 AgentX.
    """

    def __init__(self, e: float = 0.3, t: float = 0.98, **kwargs):
        """
        Initialize AgentX.

        Args:
            e: The time-dependency exponent (default 0.3)
            t: Time threshold after which window-based acceptance kicks in (default 0.98)
            **kwargs: Additional arguments passed to BOANegotiator
        """
        offering = GTimeDependentOffering(e=e)
        acceptance = GACCombiMaxInWindow(offering_policy=offering, t=t)
        model = GAgentXFrequencyModel()
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


# =============================================================================
# Utility Agents
# =============================================================================


class GRandom(BOANegotiator):
    """
    Python-native Random negotiator.

    A simple agent that makes random offers and accepts any offer. Useful for
    testing and as a baseline.

    Uses:
        - Offering: GRandomOffering (random bid selection)
        - Acceptance: GACTrue (accepts any offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        offering = GRandomOffering()
        acceptance = GACTrue()
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)
