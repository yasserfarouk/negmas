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
    """ABMPAgent2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.ABMPAgent2"
        super().__init__(**kwargs)


class BayesianAgent(GeniusNegotiator):
    """BayesianAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.BayesianAgent"
        super().__init__(**kwargs)


class BoulwareNegotiationParty(GeniusNegotiator):
    """BoulwareNegotiationParty implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.BoulwareNegotiationParty"
        super().__init__(**kwargs)


class ConcederNegotiationParty(GeniusNegotiator):
    """ConcederNegotiationParty implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.ConcederNegotiationParty"
        super().__init__(**kwargs)


class FunctionalAcceptor(GeniusNegotiator):
    """FunctionalAcceptor implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.FunctionalAcceptor"
        super().__init__(**kwargs)


class FuzzyAgent(GeniusNegotiator):
    """FuzzyAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.FuzzyAgent"
        super().__init__(**kwargs)


class ImmediateAcceptor(GeniusNegotiator):
    """ImmediateAcceptor implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.ImmediateAcceptor"
        super().__init__(**kwargs)


class OptimalBidderSimple(GeniusNegotiator):
    """OptimalBidderSimple implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.OptimalBidderSimple"
        super().__init__(**kwargs)


class RandomCounterOfferNegotiationParty(GeniusNegotiator):
    """RandomCounterOfferNegotiationParty implementation."""

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
    """RandomParty implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.RandomParty"
        super().__init__(**kwargs)


class RandomParty2(GeniusNegotiator):
    """RandomParty2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "negotiator.parties.RandomParty2"
        super().__init__(**kwargs)


class SimilarityAgent(GeniusNegotiator):
    """SimilarityAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.SimilarityAgent"
        super().__init__(**kwargs)


class SimpleAgent(GeniusNegotiator):
    """SimpleAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = ["agents.SimpleAgent"]
        super().__init__(**kwargs)


class TimeDependentAgentBoulware(GeniusNegotiator):
    """TimeDependentAgentBoulware implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentBoulware"
        super().__init__(**kwargs)


class TimeDependentAgentConceder(GeniusNegotiator):
    """TimeDependentAgentConceder implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentConceder"
        super().__init__(**kwargs)


class TimeDependentAgentHardliner(GeniusNegotiator):
    """TimeDependentAgentHardliner implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentHardliner"
        super().__init__(**kwargs)


class TimeDependentAgentLinear(GeniusNegotiator):
    """TimeDependentAgentLinear implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.TimeDependentAgentLinear"
        super().__init__(**kwargs)


class UtilityBasedAcceptor(GeniusNegotiator):
    """UtilityBasedAcceptor implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.UtilityBasedAcceptor"
        super().__init__(**kwargs)
