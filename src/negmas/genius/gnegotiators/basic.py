"""Genius negotiator implementations - Basic utility agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "ABMPAgent2",
    "BayesianAgent",
    "BoulwareNegotiationParty",
    "ConcederNegotiationParty",
    "FunctionalAcceptor",
    "FuzzyAgent",
    "ImmediateAcceptor",
    "OptimalBidderSimple",
    "RandomCounterOfferNegotiationParty",
    "RandomParty",
    "RandomParty2",
    "SimilarityAgent",
    "SimpleAgent",
    "TimeDependentAgentBoulware",
    "TimeDependentAgentConceder",
    "TimeDependentAgentHardliner",
    "TimeDependentAgentLinear",
    "UtilityBasedAcceptor",
]


class ABMPAgent2(GeniusNegotiator):
    """
    ABMP (Automated Bilateral Multi-issue Protocol) Agent v2.

    Uses the ABMP negotiation protocol which combines time-dependent concession
    with similarity-based opponent modeling. Makes offers based on a target
    utility that decreases over time while trying to maximize joint utility.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Jonker, C. M., & Treur, J. (2001). An Agent Architecture for
        Multi-Attribute Negotiation. IJCAI.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.ABMPAgent2"
        super().__init__(**kwargs)


class BayesianAgent(GeniusNegotiator):
    """
    Bayesian learning negotiation agent.

    Uses Bayesian inference to model opponent preferences by maintaining
    probability distributions over possible opponent utility functions.
    Updates beliefs after each opponent offer and uses this model to
    find mutually beneficial outcomes.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Hindriks, K., & Tykhonov, D. (2008). Opponent Modelling in Automated
        Multi-Issue Negotiation Using Bayesian Learning. AAMAS.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.BayesianAgent"
        super().__init__(**kwargs)


class BoulwareNegotiationParty(GeniusNegotiator):
    """
    Boulware (hardheaded) time-dependent negotiation agent.

    Implements a Boulware concession strategy with exponent e < 1, meaning
    it concedes slowly at first and rapidly near the deadline. This is an
    aggressive strategy that maintains high demands for most of the negotiation.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.BoulwareNegotiationParty"
        super().__init__(**kwargs)


class ConcederNegotiationParty(GeniusNegotiator):
    """
    Conceder time-dependent negotiation agent.

    Implements a Conceder strategy with exponent e > 1, meaning it concedes
    rapidly at the start and slowly near the deadline. This is a cooperative
    strategy that quickly lowers demands to reach agreement.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.ConcederNegotiationParty"
        super().__init__(**kwargs)


class FunctionalAcceptor(GeniusNegotiator):
    """
    Functional acceptance strategy agent.

    An agent that only decides whether to accept or reject opponent offers
    based on a utility threshold function. Does not generate counter-offers.
    Useful as a baseline or as a component in BOA (Bidding-Opponent-Acceptance)
    architecture.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Baarslag, T., et al. (2014). Decoupling Negotiating Agents to Explore
        the Space of Negotiation Strategies. AAMAS.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.FunctionalAcceptor"
        super().__init__(**kwargs)


class FuzzyAgent(GeniusNegotiator):
    """
    Fuzzy logic-based negotiation agent.

    Uses fuzzy logic rules to determine negotiation behavior including
    concession making and offer evaluation. Fuzzy membership functions
    model concepts like "good offer" or "near deadline" for more
    human-like decision making.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Kowalczyk, R., & Bui, V. (2000). On Fuzzy E-Negotiation Agents.
        World Congress on Computational Intelligence.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.FuzzyAgent"
        super().__init__(**kwargs)


class ImmediateAcceptor(GeniusNegotiator):
    """
    Immediate acceptance agent.

    Accepts any offer immediately without negotiation. Useful as a baseline
    agent for testing or when agreement is prioritized over utility.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.ImmediateAcceptor"
        super().__init__(**kwargs)


class OptimalBidderSimple(GeniusNegotiator):
    """
    Simple optimal bidding agent.

    Searches for optimal bids by evaluating offers based on own utility
    function. Uses a straightforward approach to find high-utility offers
    without complex opponent modeling.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.OptimalBidderSimple"
        super().__init__(**kwargs)


class RandomCounterOfferNegotiationParty(GeniusNegotiator):
    """
    Random counter-offer negotiation agent.

    Generates random counter-offers from the outcome space. Accepts offers
    above a certain utility threshold. Useful as a baseline for comparing
    more sophisticated strategies.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs[
            "java_class_name"
        ] = "negotiator.parties.RandomCounterOfferNegotiationParty"
        super().__init__(**kwargs)


class RandomParty(GeniusNegotiator):
    """
    Random negotiation agent.

    Makes completely random offers and acceptance decisions. Useful as a
    baseline agent for evaluating negotiation strategies.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.RandomParty"
        super().__init__(**kwargs)


class RandomParty2(GeniusNegotiator):
    """
    Random negotiation agent (variant 2).

    Alternative implementation of a random negotiation agent. Makes random
    offers from the outcome space with slightly different behavior than
    RandomParty.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.RandomParty2"
        super().__init__(**kwargs)


class SimilarityAgent(GeniusNegotiator):
    """
    Similarity-based negotiation agent.

    Uses bid similarity measures to evaluate and generate offers. Compares
    offers based on how similar they are to previously successful bids or
    to the opponent's apparent preferences.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Hindriks, K., et al. (2009). The Genius Negotiation Environment.
        COIN Workshop at AAMAS.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.SimilarityAgent"
        super().__init__(**kwargs)


class SimpleAgent(GeniusNegotiator):
    """
    Simple baseline negotiation agent.

    Basic negotiation agent implementing straightforward negotiation logic.
    Makes offers and accepts based on simple utility thresholds. Useful as
    a baseline for testing and comparison.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = ["agents.SimpleAgent"]
        super().__init__(**kwargs)


class TimeDependentAgentBoulware(GeniusNegotiator):
    """
    Boulware time-dependent negotiation agent.

    Time-dependent agent with Boulware (hardheaded) concession. Uses exponent
    e < 1, conceding slowly initially and rapidly near deadline. Maintains
    high demands throughout most of the negotiation.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentBoulware"
        super().__init__(**kwargs)


class TimeDependentAgentConceder(GeniusNegotiator):
    """
    Conceder time-dependent negotiation agent.

    Time-dependent agent with Conceder strategy. Uses exponent e > 1,
    conceding rapidly at the start and slowly near deadline. Cooperative
    approach that prioritizes reaching agreement.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentConceder"
        super().__init__(**kwargs)


class TimeDependentAgentHardliner(GeniusNegotiator):
    """
    Hardliner time-dependent negotiation agent.

    Extreme time-dependent agent that makes minimal or no concessions.
    Maintains maximum utility demands throughout negotiation, only accepting
    offers very close to its ideal outcome.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentHardliner"
        super().__init__(**kwargs)


class TimeDependentAgentLinear(GeniusNegotiator):
    """
    Linear time-dependent negotiation agent.

    Time-dependent agent with linear concession (e = 1). Concedes at a
    constant rate throughout the negotiation, from maximum utility at
    start to reservation value at deadline.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Faratin, P., Sierra, C., & Jennings, N. R. (1998). Negotiation Decision
        Functions for Autonomous Agents. Robotics and Autonomous Systems.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentLinear"
        super().__init__(**kwargs)


class UtilityBasedAcceptor(GeniusNegotiator):
    """
    Utility-based acceptance strategy agent.

    Accepts offers based purely on utility threshold comparison. If an
    offer's utility exceeds the current threshold (which may vary over
    time), it accepts; otherwise rejects.

    Note:
        AI-generated summary. May not be fully accurate.

    References:
        Baarslag, T., et al. (2014). Decoupling Negotiating Agents to Explore
        the Space of Negotiation Strategies. AAMAS.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.UtilityBasedAcceptor"
        super().__init__(**kwargs)
