"""Genius negotiator implementations - ANAC 2014 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentM",
    "AgentQuest",
    "AgentTD",
    "AgentTRP",
    "AgentYK",
    "AnacSampleAgent",
    "ArisawaYaki",
    "Aster",
    "Atlas",
    "BraveCat",
    "DoNA",
    "E2Agent",
    "Flinch",
    "Gangester",
    "Gangster",
    "KGAgent",
    "Simpatico",
    "Sobut",
    "TUDelftGroup2",
    "WhaleAgent",
]


class AgentM(GeniusNegotiator):
    """
    AgentM negotiation agent.

    **ANAC 2014**.

    Uses simulated annealing for bid search with multi-session learning.

    **Offering Strategy:**
        - Simulated annealing bid search with neighborhood exploration
        - Tracks opponent bid frequencies per issue value
        - Concession rate adapts based on opponent bid variance
        - Persists last agreed bid across sessions for continuity

    **Acceptance Strategy:**
        - Accepts if opponent bid exceeds current target utility
        - Target utility decreases over time based on opponent behavior

    **Opponent Modeling:**
        Frequency-based tracking of opponent preferences per issue value.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentM.AgentM"
        super().__init__(**kwargs)


class AgentQuest(GeniusNegotiator):
    """
    AgentQuest negotiation agent.

    **ANAC 2014**.

    Uses Euclidean distance comparison with hardheadedness monitoring.

    **Offering Strategy:**
        - Maintains top-10 bids by utility for efficient selection
        - Compares bids using Euclidean distance to opponent offers
        - Response time estimation for strategic timing

    **Acceptance Strategy:**
        - Concession probability function based on negotiation progress
        - Monitors opponent hardheadedness over sliding window
        - Adjusts acceptance threshold based on opponent flexibility

    **Opponent Modeling:**
        Sliding window analysis of opponent behavior to detect hardheadedness.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentQuest.AgentQuest"
        super().__init__(**kwargs)


class AgentTD(GeniusNegotiator):
    """
    AgentTD negotiation agent.

    **ANAC 2014**.

    Simple time-dependent strategy with phased concession.

    **Offering Strategy:**
        - Random bids above threshold until t=0.95
        - After t=0.95, offers best bid received from opponent
        - Straightforward time-pressure based approach

    **Acceptance Strategy:**
        - Before t=0.7: accepts if utility >= 0.85
        - Before t=0.98: accepts if utility >= 0.75
        - After t=0.98: accepts best available offer

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTD.AgentTD"
        super().__init__(**kwargs)


class AgentTRP(GeniusNegotiator):
    """
    AgentTRP negotiation agent.

    **ANAC 2014**.

    Multi-mode strategy with session persistence and adaptive mode selection.

    **Offering Strategy:**
        - Multiple search modes: simulated annealing and neighbor search
        - Mode selection based on agreement rate history from past sessions
        - Persists negotiation data across sessions

    **Acceptance Strategy:**
        - Time-dependent threshold with mode-specific adjustments
        - Considers past session agreement patterns

    **Multi-Session Learning:**
        Tracks agreement rates per mode and selects best-performing strategy.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentTRP.AgentTRP"
        super().__init__(**kwargs)


class AgentYK(GeniusNegotiator):
    """
    AgentYK negotiation agent.

    **ANAC 2014**.

    Hill climbing search with bid element history tracking.

    **Offering Strategy:**
        - Hill climbing with configurable stop criteria
        - Tracks bid element history with weighted appearance counts
        - Time-bonus for unexplored bids to encourage diversity

    **Acceptance Strategy:**
        - Accepts if opponent bid meets utility threshold
        - Threshold decreases over negotiation time

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentYK.AgentYK"
        super().__init__(**kwargs)


class AnacSampleAgent(GeniusNegotiator):
    """
    AnacSampleAgent negotiation agent.

    **ANAC 2014**.

    AgentK-based strategy with simulated annealing and session persistence.
    This is the same implementation as E2Agent.

    **Offering Strategy:**
        - Simulated annealing for bid search
        - Target utility adaptation based on discount factor
        - Session data persistence for multi-session learning

    **Acceptance Strategy:**
        - Time-dependent acceptance with discount factor awareness
        - Adapts based on opponent behavior history

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class ArisawaYaki(GeniusNegotiator):
    """
    ArisawaYaki negotiation agent.

    **ANAC 2014**.

    Simulated annealing with sigmoid threshold and adaptive minimum utility.

    **Offering Strategy:**
        - Simulated annealing for bid search
        - Sigmoid-based threshold function for concession
        - Adjusts minimum utility based on opponent bid average changes

    **Acceptance Strategy:**
        - Accepts if opponent offer exceeds sigmoid-based threshold
        - Threshold adapts based on time and opponent behavior

    **Opponent Modeling:**
        Monitors changes in opponent bid averages to detect cooperation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.ArisawaYaki.ArisawaYaki"
        super().__init__(**kwargs)


class Aster(GeniusNegotiator):
    """
    Aster negotiation agent.

    **ANAC 2014**.

    Simulated annealing with multi-session learning and agreement history tracking.

    **Offering Strategy:**
        - Simulated annealing bid search
        - Concession degree monitoring for adaptive behavior
        - Immediate decision-making based on past agreement patterns

    **Acceptance Strategy:**
        - Considers agreement history from previous sessions
        - Adapts threshold based on learned opponent patterns

    **Multi-Session Learning:**
        Tracks and uses agreement history for improved decision-making.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Aster.Aster"
        super().__init__(**kwargs)


class Atlas(GeniusNegotiator):
    """
    Atlas negotiation agent.

    **ANAC 2014**.

    Dual-mode agent with extensive session learning and discount awareness.

    **Offering Strategy:**
        - Two modes: "tension" (aggressive) and "weak" (concessive)
        - Simulated annealing and neighbor search for bid generation
        - Mode selection based on session history and opponent type

    **Acceptance Strategy:**
        - Mode-dependent acceptance thresholds
        - Discount factor aware for time-discounted domains

    **Multi-Session Learning:**
        Extensive tracking of opponent behavior and agreement patterns.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Atlas.Atlas"
        super().__init__(**kwargs)


class BraveCat(GeniusNegotiator):
    """
    BraveCat negotiation agent.

    **ANAC 2014**.

    BOA framework agent with DBOMModel opponent modeling.

    **Offering Strategy:**
        - BRTOfferingStrategy for bid selection
        - BestBid strategy component for optimal bid search

    **Acceptance Strategy:**
        - AC_LAST acceptance condition (accepts if better than last offer)

    **Opponent Modeling:**
        DBOMModel - Decoupled Bayesian Opponent Model for preference learning.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.BraveCat.BraveCat"
        super().__init__(**kwargs)


class DoNA(GeniusNegotiator):
    """
    DoNA negotiation agent.

    **ANAC 2014**.

    Domain-size adaptive agent with statistical sampling.

    **Offering Strategy:**
        - Falls back to ClearDefaultStrategy for small domains (<1M bids)
        - Statistical sampling for large outcome spaces
        - Reservation value and discount factor aware

    **Acceptance Strategy:**
        - Adapts based on domain size and complexity
        - Considers reservation value in acceptance decisions

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.DoNA.DoNA"
        super().__init__(**kwargs)


class E2Agent(GeniusNegotiator):
    """
    E2Agent negotiation agent.

    **ANAC 2014**.

    AgentK-based strategy with simulated annealing and session persistence.
    This is the same implementation as AnacSampleAgent.

    **Offering Strategy:**
        - Simulated annealing for bid search
        - Target utility adaptation based on discount factor
        - Session data persistence for multi-session learning

    **Acceptance Strategy:**
        - Time-dependent acceptance with discount factor awareness
        - Adapts based on opponent behavior history

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.E2Agent.AnacSampleAgent"
        super().__init__(**kwargs)


class Flinch(GeniusNegotiator):
    """
    Flinch negotiation agent.

    **ANAC 2014**.

    Genetic algorithm bid search with kernel-based opponent modeling.

    **Offering Strategy:**
        - Genetic algorithm for bid optimization
        - Population-based search for diverse bid exploration

    **Acceptance Strategy:**
        - Time-dependent threshold with discount factor awareness
        - Adapts based on estimated opponent utility

    **Opponent Modeling:**
        Kernel-based estimation of opponent utility from bid history.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Flinch.Flinch"
        super().__init__(**kwargs)


class Gangester(GeniusNegotiator):
    """
    Gangester negotiation agent (alternate spelling of Gangster).

    **ANAC 2014**.

    Genetic algorithm for nonlinear domains with bid storage.

    **Offering Strategy:**
        - Genetic algorithm for nonlinear utility spaces
        - Local and global search strategies
        - Concession based on max distance to opponent bids
        - Bid storage with reproposal capability

    **Acceptance Strategy:**
        - Accepts if opponent bid utility exceeds threshold
        - Threshold decreases based on time and opponent behavior

    **Opponent Modeling:**
        Distance-based analysis of opponent bid patterns.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class Gangster(GeniusNegotiator):
    """
    Gangster negotiation agent.

    **ANAC 2014**.

    Genetic algorithm for nonlinear domains with bid storage.

    **Offering Strategy:**
        - Genetic algorithm for nonlinear utility spaces
        - Local and global search strategies
        - Concession based on max distance to opponent bids
        - Bid storage with reproposal capability

    **Acceptance Strategy:**
        - Accepts if opponent bid utility exceeds threshold
        - Threshold decreases based on time and opponent behavior

    **Opponent Modeling:**
        Distance-based analysis of opponent bid patterns.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Gangster.Gangster"
        super().__init__(**kwargs)


class KGAgent(GeniusNegotiator):
    """
    KGAgent negotiation agent.

    **ANAC 2014**.

    Genetic algorithm optimization with opponent utility estimation.

    **Offering Strategy:**
        - Genetic algorithm for bid optimization
        - Time pressure adjustments for concession rate

    **Acceptance Strategy:**
        - Accepts based on estimated mutual benefit
        - Time-dependent threshold adaptation

    **Opponent Modeling:**
        Estimates opponent utility from bid history for Pareto-optimal targeting.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.KGAgent.KGAgent"
        super().__init__(**kwargs)


class Simpatico(GeniusNegotiator):
    """
    Simpatico negotiation agent.

    **ANAC 2014**.

    Neighborhood search around opponent bids with cooperation detection.

    **Offering Strategy:**
        - Neighborhood search around opponent's previous bids
        - Random search with vicinity exploration
        - Adapts search based on opponent cooperation level

    **Acceptance Strategy:**
        - Accepts if opponent shows cooperative behavior
        - Time-dependent threshold with cooperation bonus

    **Opponent Modeling:**
        Detects opponent cooperation from bid pattern changes.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.SimpaticoAgent.Simpatico"
        super().__init__(**kwargs)


class Sobut(GeniusNegotiator):
    """
    Sobut negotiation agent.

    **ANAC 2014**.

    Simple random walker with multi-session learning.

    **Offering Strategy:**
        - Random bid selection above minimum utility threshold
        - Minimum bid utility derived from past session outcomes

    **Acceptance Strategy:**
        - Accepts if opponent bid exceeds learned minimum threshold
        - Threshold based on discounted utility from past agreements

    **Multi-Session Learning:**
        Saves discounted utility outcomes to inform future sessions.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.Sobut.Sobut"
        super().__init__(**kwargs)


class TUDelftGroup2(GeniusNegotiator):
    """
    TUDelftGroup2 negotiation agent.

    **ANAC 2014**.

    BOA framework agent with custom opponent model and bidding strategy.

    **Offering Strategy:**
        - Group2_BS custom bidding strategy
        - Decoupled component-based approach

    **Acceptance Strategy:**
        - Group2_AS custom acceptance strategy

    **Opponent Modeling:**
        Group2_OM - Custom opponent model for preference estimation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.TUDelftGroup2.Group2Agent"
        super().__init__(**kwargs)


class WhaleAgent(GeniusNegotiator):
    """
    WhaleAgent negotiation agent.

    **ANAC 2014**.

    Multi-session learning with action type switching.

    **Offering Strategy:**
        - Action type switching: aggressive vs passive modes
        - Simulated annealing and neighbor search for bid generation
        - Sigmoid-based threshold functions

    **Acceptance Strategy:**
        - Mode-dependent acceptance thresholds
        - Adapts based on session history and opponent behavior

    **Multi-Session Learning:**
        Learns optimal action type (aggressive/passive) across sessions.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Baarslag, T., et al. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2014.AgentWhale.WhaleAgent"
        super().__init__(**kwargs)
