"""Genius negotiator implementations - ANAC 2013 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentI",
    "AgentKF",
    "GAgent",
    "InoxAgent",
    "MetaAgent2013",
    "SlavaAgent",
    "TMFAgent",
    "TheFawkes",
]


class AgentI(GeniusNegotiator):
    """
    AgentI (GAgent) negotiation agent.

    **ANAC 2013**.

    Uses probability-based opponent modeling and sigmoid threshold for
    adaptive concession. Analyzes opponent bid variance to determine
    concession rate.

    **Offering Strategy:**
        - Uses sorted outcome space for efficient bid selection
        - Sigmoid function controls minimum utility threshold
        - Threshold adapts based on opponent's variance (willingness to concede)
        - Near deadline (t > 0.9988): offers from opponent's best bids
        - Avoids repeating previously offered bids

    **Acceptance Strategy:**
        - Accepts if opponent bid was previously offered by self
        - Accepts if opponent utility > current minimum threshold
        - Near deadline: accepts if opponent offer > 5th best opponent bid
        - Respects reservation value as lower bound

    **Opponent Modeling:**
        - Tracks variance in opponent bid utility changes
        - Estimates opponent's concession willingness
        - Uses probability model to predict future opponent behavior
        - Width of utility range determined by first opponent bid

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class AgentKF(GeniusNegotiator):
    """
    AgentKF negotiation agent.

    **ANAC 2013**.

    Enhanced version of AgentK with persistent learning across negotiation
    sessions. Stores opponent bid history and adjusts behavior based on
    accumulated experience.

    **Offering Strategy:**
        - Maintains sorted bid list by utility
        - Uses time-based threshold with "tremor" adjustment
        - Tremor increases near deadline for more aggressive concession
        - Adjusts based on bid similarity to opponent's history
        - Persists best bids across sessions for learning

    **Acceptance Strategy:**
        - Accepts if offered bid utility exceeds current threshold
        - Threshold computed from target utility and tremor factor
        - More accepting as deadline approaches

    **Opponent Modeling:**
        - Stores all opponent bids across sessions (persistent data)
        - Computes bid similarity scores against opponent history
        - Uses accumulated data to predict opponent preferences

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Kawaguchi, S., Fujita, K., & Ito, T. (2012). AgentK: Compromising
        strategy based on estimated maximum utility for automated negotiating
        agents. In: *New Trends in Agent-based Complex Automated Negotiations*.
        Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.AgentKF.AgentKF"
        super().__init__(**kwargs)


class GAgent(GeniusNegotiator):
    """
    GAgent negotiation agent.

    **ANAC 2013**.

    Alias for AgentI - both use the same underlying Java implementation.
    See :class:`AgentI` for full documentation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.GAgent.AgentI"
        super().__init__(**kwargs)


class InoxAgent(GeniusNegotiator):
    """
    InoxAgent negotiation agent.

    **ANAC 2013**.

    BOA framework-based agent with custom opponent model and acceptance
    strategy components.

    **Offering Strategy:**
        - Uses BestBid opponent model strategy
        - Selects bids that maximize expected opponent utility
        - Custom InoxAgent_Offering strategy component

    **Acceptance Strategy:**
        - AC_InoxAgent acceptance component
        - Considers both own utility and opponent modeling predictions

    **Opponent Modeling:**
        - InoxAgent_OM custom opponent model
        - Frequency-based preference learning
        - Updates model with each opponent bid

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.InoxAgent.InoxAgent"
        super().__init__(**kwargs)


class MetaAgent2013(GeniusNegotiator):
    """
    MetaAgent2013 negotiation agent.

    **ANAC 2013**.

    Meta-level agent that selects the best negotiation strategy from a pool
    of agents based on domain features. Uses regression-based prediction
    to choose optimal agent for each negotiation context.

    **Agent Selection:**
        - Extracts domain features: number of issues, issue sizes, utilities
        - Considers discount factor and reservation value
        - Uses AgentManager for regression-based agent selection
        - Persists results across sessions for learning

    **Strategy:**
        - Delegates all negotiation to selected sub-agent
        - Selection made at negotiation start based on domain analysis
        - Stores performance data for future agent selection improvement

    **Learning:**
        - Maintains persistent data across negotiation sessions
        - Updates regression model with outcomes
        - Improves agent selection over time

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Mbarki, M., & Larson, K. (2013). An adaptive meta-agent for negotiation.
        In: *AAMAS Workshop on Automated Negotiating Agents*.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.MetaAgent.MetaAgent2013"
        super().__init__(**kwargs)


class SlavaAgent(GeniusNegotiator):
    """
    SlavaAgent negotiation agent.

    **ANAC 2013**.

    Two-phase negotiation strategy with exploration and exploitation phases.
    Maintains persistent storage of best offers across sessions.

    **Offering Strategy:**
        - Exploration phase (t < 0.95): offers from high-utility bids (> 0.95)
        - 50% chance to offer maximum utility bid vs random good bid
        - Exploitation phase (t >= 0.95): more aggressive concession
        - Stores and reuses best bids across sessions

    **Acceptance Strategy:**
        - Accepts if utility > 0.9 (high threshold)
        - Also accepts if utility > 0.7 AND >= best opponent offer seen
        - Tracks best opponent offer across sessions

    **Learning:**
        - Persists best opponent offers between sessions
        - Uses session data to inform acceptance decisions
        - Conservative strategy with high utility requirements

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.SlavaAgent.SlavaAgent"
        super().__init__(**kwargs)


class TMFAgent(GeniusNegotiator):
    """
    TMFAgent negotiation agent.

    **ANAC 2013**.

    Time-aware agent with Round Trip Time (RTT) estimation for timeout
    prevention. Builds opponent utility model and optimizes for joint gains.

    **Offering Strategy:**
        - Sorts all possible bids by own utility
        - Searches for bids with best combined utility (own + estimated opponent)
        - Uses "hardness" parameter based on time and discount factor
        - Becomes more concessive as deadline approaches

    **Acceptance Strategy:**
        - Accepts if opponent bid utility exceeds computed threshold
        - Threshold decreases over time based on hardness parameter
        - Considers discount factor in threshold computation

    **Opponent Modeling:**
        - Time-weighted frequency model for opponent preferences
        - Estimates opponent utility for each possible bid
        - Updates model incrementally with each opponent bid

    **Time Management:**
        - Estimates RTT to prevent timeout near deadline
        - Adjusts offer generation speed based on remaining time
        - Conservative time handling for reliability

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.TMFAgent.TMFAgent"
        super().__init__(**kwargs)


class TheFawkes(GeniusNegotiator):
    """
    TheFawkes negotiation agent.

    **ANAC 2013**.

    BOA framework-based agent with custom components for opponent modeling,
    offer generation, and acceptance. Named after Guy Fawkes.

    **Offering Strategy:**
        - Fawkes_Offering bidding strategy component
        - Uses opponent model to find mutually beneficial bids
        - Time-dependent concession behavior

    **Acceptance Strategy:**
        - AC_TheFawkes acceptance component
        - Considers predicted opponent utility in decisions
        - Balances own utility with negotiation progress

    **Opponent Modeling:**
        - TheFawkes_OM opponent model component
        - TheFawkes_OMS opponent model strategy
        - Learns opponent preferences from bid history

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2013.TheFawkes.TheFawkes"
        super().__init__(**kwargs)
