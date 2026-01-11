"""Genius negotiator implementations - ANAC 2011 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentK2",
    "BramAgent",
    "BramAgent2",
    "Gahboninho",
    "HardHeaded",
    "IAMhaggler2011",
    "NiceTitForTat",
    "TheNegotiator",
    "ValueModelAgent",
]


class AgentK2(GeniusNegotiator):
    """AgentK2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.AgentK2.Agent_K2"
        super().__init__(**kwargs)


class BramAgent(GeniusNegotiator):
    """BramAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class BramAgent2(GeniusNegotiator):
    """BramAgent2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class Gahboninho(GeniusNegotiator):
    """Gahboninho implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.Gahboninho.Gahboninho"
        super().__init__(**kwargs)


class HardHeaded(GeniusNegotiator):
    """
    HardHeaded negotiation agent.

    **ANAC 2011 Winner** (Individual Utility category).

    As the name implies, HardHeaded (developed by Thijs van Krimpen) is an
    aggressive negotiator that maintains high demands throughout most of the
    negotiation and only concedes near the deadline.

    **Offering Strategy:**
        - Uses a monotonic concession function that generates bids in decreasing
          utility order
        - Cycles through the same range of high-utility bids for most of the
          negotiation
        - Resets to a random bid after reaching the dynamic concession limit
        - Selects bids that maximize estimated opponent utility among equivalent
          bids for itself

    **Acceptance Strategy:**
        - Accepts if opponent's offer exceeds the lowest utility offered so far
        - Accepts if opponent's offer is better than the next planned offer
        - Very conservative early acceptance thresholds

    **Opponent Modeling:**
        Frequency-based learning approach:

        - Tracks unchanged issues between consecutive opponent bids to estimate
          issue weights
        - Counts value frequencies to estimate value utilities
        - Uses learned model to select bids favorable to opponent among
          equivalent options

    References:
        van Krimpen, T., Looije, D., & Hajizadeh, S. (2013). HardHeaded.
        In Complex Automated Negotiations: Theories, Models, and Software
        Competitions. Studies in Computational Intelligence, vol 435.
        Springer, Berlin, Heidelberg.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.HardHeaded.KLH"
        super().__init__(**kwargs)


class IAMhaggler2011(GeniusNegotiator):
    """IAMhaggler2011 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.IAMhaggler2011.IAMhaggler2011"
        super().__init__(**kwargs)


class NiceTitForTat(GeniusNegotiator):
    """
    NiceTitForTat negotiation agent.

    NiceTitForTat (developed by Tim Baarslag) implements a cooperative
    tit-for-tat strategy with respect to utility space, aiming for the
    Nash bargaining solution.

    **Offering Strategy:**
        - Initially cooperates with high utility bids
        - Responds in kind to opponent's concessions
        - Calculates opponent's concession factor relative to Nash point
        - Mirrors opponent's concession proportionally
        - Time bonus near deadline to encourage agreement
        - Selects bids that maximize opponent utility among equivalents

    **Acceptance Strategy:**
        - Accepts if opponent's offer >= planned counter-offer utility
        - Near deadline: probabilistic acceptance based on expected utility
          of waiting for better offers
        - Considers recent bid history to estimate probability of improvement

    **Opponent Modeling:**
        Bayesian opponent model that:

        - Updates beliefs about opponent preferences after each bid
        - Estimates opponent's utility function
        - Used to find Nash point and select opponent-favorable bids
        - Guides concession strategy to match opponent's behavior

    References:
        Baarslag, T., Hindriks, K., & Jonker, C. (2013). A tit for tat
        negotiation strategy for real-time bilateral negotiations.
        Studies in Computational Intelligence, 435:229-233.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.Nice_Tit_for_Tat.NiceTitForTat"
        super().__init__(**kwargs)


class TheNegotiator(GeniusNegotiator):
    """The negotiator."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.TheNegotiator.TheNegotiator"
        super().__init__(**kwargs)


class ValueModelAgent(GeniusNegotiator):
    """ValueModelAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.ValueModelAgent.ValueModelAgent"
        super().__init__(**kwargs)
