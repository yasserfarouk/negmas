"""Genius negotiator implementations - ANAC 2019 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentGG",
    "AgentGP",
    "AgentLarry",
    "DandikAgent",
    "EAgent",
    "FSEGA2019",
    "GaravelAgent",
    "Gravity",
    "Group1BOA",
    "HardDealer",
    "KAgent",
    "KakeSoba",
    "MINF",
    "PodAgent",
    "SACRA",
    "SAGA",
    "SolverAgent",
    "TheNewDeal",
    "WinkyAgent",
]


class AgentGG(GeniusNegotiator):
    """
    AgentGG negotiation agent.

    **ANAC 2019 Winner** (Individual Utility category).

    AgentGG (developed by Shaobo Xu) uses importance maps (a frequentist
    approach) to estimate both self and opponent preferences, focusing on
    bid importance rather than raw utility values.

    **Offering Strategy:**
        - Time-based concession with importance thresholds
        - Early phase (t < 0.2): random bid selection within threshold
        - Middle phase: selects bids maximizing estimated opponent importance
        - Thresholds decrease over time based on estimated Nash point
        - Uses importance maps instead of utility for bid evaluation

    **Acceptance Strategy:**
        - Accepts if received bid's importance ratio exceeds current threshold
        - Near deadline (t >= 0.9989): accepts if importance exceeds
          reservation + 0.2
        - Thresholds adapt based on estimated Nash point

    **Opponent Modeling:**
        Frequentist importance maps that estimate:

        - Self preferences from own utility function analysis
        - Opponent preferences from their bidding patterns
        - Uses estimated opponent importance to select favorable bids
        - Updates opponent model during early negotiation (t < 0.3)

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of the Automated
        Negotiating Agents Competition (ANAC) 2019. In Multi-Agent Systems and
        Agreement Technologies. EUMAS AT 2020. Lecture Notes in Computer Science,
        vol 12520. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.agentgg.AgentGG"
        super().__init__(**kwargs)


class AgentGP(GeniusNegotiator):
    """AgentGP implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.agentgp.AgentGP"
        super().__init__(**kwargs)


class AgentLarry(GeniusNegotiator):
    """AgentLarry implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.agentlarry.AgentLarry"
        super().__init__(**kwargs)


class DandikAgent(GeniusNegotiator):
    """DandikAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.dandikagent.dandikAgent"
        super().__init__(**kwargs)


class EAgent(GeniusNegotiator):
    """EAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.eagent.EAgent"
        super().__init__(**kwargs)


class FSEGA2019(GeniusNegotiator):
    """FSEGA2019 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.fsega2019.agent.FSEGA2019"
        super().__init__(**kwargs)


class GaravelAgent(GeniusNegotiator):
    """GaravelAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.garavelagent.GaravelAgent"
        super().__init__(**kwargs)


class Gravity(GeniusNegotiator):
    """Gravity implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.gravity.Gravity"
        super().__init__(**kwargs)


class Group1BOA(GeniusNegotiator):
    """Group1BOA implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class HardDealer(GeniusNegotiator):
    """HardDealer implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.harddealer.HardDealer"
        super().__init__(**kwargs)


class KAgent(GeniusNegotiator):
    """KAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.kagent.KAgent"
        super().__init__(**kwargs)


class KakeSoba(GeniusNegotiator):
    """KakeSoba implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.kakesoba.KakeSoba"
        super().__init__(**kwargs)


class MINF(GeniusNegotiator):
    """MINF implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.minf.MINF"
        super().__init__(**kwargs)


class PodAgent(GeniusNegotiator):
    """PodAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class SACRA(GeniusNegotiator):
    """SACRA implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.sacra.SACRA"
        super().__init__(**kwargs)


class SAGA(GeniusNegotiator):
    """
    SAGA (Simulated Annealing Genetic Algorithm) negotiation agent.

    **ANAC 2019 Individual Utility Category Finalist**.

    SAGA Agent by Yuta Hosokawa applies a Genetic Algorithm approach to estimate
    its own preferences (using Spearman correlation as the fitness function)
    combined with time-based bidding and acceptance strategies.

    **Offering Strategy:**
        Uses a time-dependent target utility function:

        target(t) = target_min + (1 - target_min) * (1 - t^5)

        where target_min is derived from the utility of the first received
        bid: target_min = firstUtil + 0.6 * (1 - firstUtil).

        Bids are randomly generated above the target utility threshold.

    **Acceptance Strategy:**
        Employs a three-phase probabilistic acceptance strategy:

        - **Phase 1 (t <= 0.6)**: Probabilistic acceptance based on how much
          the offer exceeds the target. Uses power function with exponent
          that increases as time approaches 0.5.
        - **Phase 2 (0.6 < t < 0.997)**: Gradually increasing acceptance
          probability for bids below target, with linear interpolation.
        - **Phase 3 (t >= 0.997)**: Near deadline, accepts with probability
          proportional to utility squared.

        Always rejects offers below reservation value.

    **Opponent Modeling:**
        SAGA Agent was designed to use Genetic Algorithm to estimate
        preferences, though the current implementation uses actual
        preferences. The GA approach uses Spearman rank correlation as
        the fitness metric to evaluate preference estimation quality.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of the Automated
        Negotiating Agents Competition (ANAC) 2019. In Multi-Agent Systems and
        Agreement Technologies. EUMAS AT 2020. Lecture Notes in Computer Science,
        vol 12520. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.saga.SAGA"
        super().__init__(**kwargs)


class SolverAgent(GeniusNegotiator):
    """SolverAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.solveragent.SolverAgent"
        super().__init__(**kwargs)


class TheNewDeal(GeniusNegotiator):
    """TheNewDeal implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.thenewdeal.TheNewDeal"
        super().__init__(**kwargs)


class WinkyAgent(GeniusNegotiator):
    """WinkyAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.winkyagent.winkyAgent"
        super().__init__(**kwargs)
