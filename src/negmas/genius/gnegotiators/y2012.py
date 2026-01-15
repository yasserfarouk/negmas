"""Genius negotiator implementations - ANAC 2012 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentLG",
    "AgentMR",
    "CUHKAgent",
    "IAMhaggler2012",
    "MetaAgent",
    "MetaAgent2012",
    "OMACagent",
    "TheNegotiatorReloaded",
]


class AgentLG(GeniusNegotiator):
    """
    AgentLG negotiation agent.

    **ANAC 2012**.

    A time-dependent agent that is stubborn early but compromises near the
    deadline. Uses opponent bid history to learn preferences and select
    mutually beneficial offers.

    **Offering Strategy:**
        - Phase 1 (t < 0.6): Offers top 25% utility bids incrementally,
          learns opponent preferences
        - Phase 2 (0.6 <= t < 0.9995): Starts compromising, chooses bids
          better for opponent while maintaining own threshold
        - Final phase (t >= 0.9995): Offers opponent's max utility bid for self
        - Minimum utility threshold starts at 75% of max

    **Acceptance Strategy:**
        - Accepts if opponent offer >= 99% of own last offer utility
        - Accepts if time > 0.999 and offer >= 90% of own utility
        - Accepts if offer >= minimum bid utility threshold
        - Accepts if previously offered same bid

    **Opponent Modeling:**
        - Tracks all opponent bids and their utilities
        - Maintains estimate of opponent's best bid for self
        - Uses opponent history to select compromise bids

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
        kwargs["java_class_name"] = "agents.anac.y2012.AgentLG.AgentLG"
        super().__init__(**kwargs)


class AgentMR(GeniusNegotiator):
    """
    AgentMR negotiation agent.

    **ANAC 2012**.

    Uses sigmoid-based concession with adaptive parameters based on opponent
    behavior forecasting. Adjusts strategy based on discount factor presence.

    **Offering Strategy:**
        - Maintains sorted bid ranking by utility
        - Generates bid variations from opponent offers meeting threshold
        - Sigmoid function controls minimum acceptable utility over time
        - Parameters adapt based on opponent concession forecasting at t=0.5
        - Near deadline (t > 0.985): offers opponent's best bid if acceptable

    **Acceptance Strategy:**
        - Probabilistic acceptance: P = f(utility, time³)
        - Accepts if P > 0.965 or utility > threshold
        - Accepts if bid was previously in own bid ranking
        - Time-cubic function makes deadline acceptance more likely

    **Opponent Modeling:**
        - Tracks opponent's maximum offered utility over time
        - Forecasts opponent concession at t=0.5 to adjust sigmoid parameters
        - Updates sigmoid gain and percent based on concession rate
        - Different parameters for discounted vs non-discounted domains

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
        kwargs["java_class_name"] = "agents.anac.y2012.AgentMR.AgentMR"
        super().__init__(**kwargs)


class CUHKAgent(GeniusNegotiator):
    """
    CUHKAgent negotiation agent.

    **ANAC 2012 Winner**.

    CUHKAgent (developed at Chinese University of Hong Kong by Jianye Hao)
    is an adaptive negotiation agent that adjusts its strategy based on
    opponent behavior and time pressure.

    **Offering Strategy:**
        - Time-dependent concession with adaptive threshold adjustment
        - Concession rate adapts based on opponent's toughness degree
        - In large domains: focuses on high-utility bid range
        - Near deadline: considers opponent's best offer as fallback
        - Uses opponent model to select bids favorable to opponent among
          candidates

    **Acceptance Strategy:**
        - Accepts if offer exceeds current utility threshold
        - Accepts if offer exceeds the utility of planned counter-offer
        - Near deadline: more lenient acceptance based on opponent's best offer
        - Adapts acceptance based on predicted maximum achievable utility

    **Opponent Modeling:**
        - Tracks opponent's bidding history to estimate preferences
        - Calculates opponent's concession degree to adapt own strategy
        - Identifies opponent's maximum offered bid for reference
        - Uses opponent model to choose mutually beneficial bids

    References:
        Hao, J., & Leung, H. (2014). CUHKAgent: An Adaptive Negotiation Strategy
        for Bilateral Negotiations over Multiple Items. In Novel Insights in
        Agent-based Complex Automated Negotiation. Studies in Computational
        Intelligence, vol 535. Springer, Tokyo.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.CUHKAgent.CUHKAgent"
        super().__init__(**kwargs)


class IAMhaggler2012(GeniusNegotiator):
    """
    IAMhaggler2012 negotiation agent.

    **ANAC 2012**.

    An enhanced version of IAMhaggler2011 with reservation value support
    and improved utility space handling. Developed by the University of
    Southampton team.

    **Offering Strategy:**
        - Inherits GP regression-based targeting from IAMhaggler2011
        - Uses SouthamptonUtilitySpace for improved bid generation
        - Ensures offers never fall below reservation value
        - Falls back to initial bid if proposed bid is below reservation

    **Acceptance Strategy:**
        - Same as IAMhaggler2011 (multiplier-based acceptance)
        - Additional check against reservation value

    **Opponent Modeling:**
        - Same Gaussian Process regression as IAMhaggler2011
        - Time-slotted sampling of opponent behavior
        - Bayesian Monte Carlo for regression updates

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Williams, C.R., Robu, V., Gerding, E.H., & Jennings, N.R. (2012).
        IAMhaggler: A negotiation agent for complex environments. In
        New Trends in Agent-based Complex Automated Negotiations.
        Studies in Computational Intelligence, vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.IAMhaggler2012.IAMhaggler2012"
        super().__init__(**kwargs)


class MetaAgent(GeniusNegotiator):
    """
    MetaAgent negotiation agent.

    **ANAC 2012**.

    A meta-level agent that dynamically selects from a pool of existing
    negotiation agents based on domain characteristics. Uses regression
    models to predict which agent will perform best.

    **Agent Selection:**
        - Analyzes domain: issues, discount factor, domain size,
          expected utility, standard deviations
        - Uses pre-trained regression coefficients for 18 candidate agents
        - Applies Quantal Response Equilibrium (QRE) for probabilistic selection
        - Candidate pool includes winners from ANAC 2010-2011

    **Offering Strategy:**
        - First bid: always offers maximum utility bid
        - Subsequent bids: delegated to selected agent
        - Handles reservation value (ends negotiation if offer below)

    **Candidate Agents:**
        HardHeaded, AgentK2, TheNegotiator, ValueModelAgent, Gahboninho,
        BRAMAgent, AgentSmith, and several custom agents (Chameleon,
        WinnerAgent, GYRL, DNAgent, etc.)

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
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class MetaAgent2012(GeniusNegotiator):
    """
    MetaAgent2012 negotiation agent.

    **ANAC 2012**.

    Identical to MetaAgent - this is an alias wrapper pointing to the same
    Java implementation (agents.anac.y2012.MetaAgent.MetaAgent).

    See MetaAgent for full documentation of the strategy.

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
        kwargs["java_class_name"] = "agents.anac.y2012.MetaAgent.MetaAgent"
        super().__init__(**kwargs)


class OMACagent(GeniusNegotiator):
    """
    OMACagent negotiation agent.

    **ANAC 2012**.

    Opponent Modeling and Adaptive Concession agent developed at Maastricht
    University. Uses time-series forecasting to predict opponent behavior
    and adapt concession strategy accordingly.

    **Offering Strategy:**
        - Early phase (t <= 0.02): offers maximum utility bid
        - Uses exponential moving average (EMA) forecasting of opponent bids
        - Generates random bids within utility range [target±1%]
        - Adjusts target based on forecast vs original concession curve
        - Different parameters for high vs low discount factor domains

    **Acceptance Strategy:**
        - Accepts if opponent offer >= own planned offer utility
        - Accepts if bid was previously in own bid history
        - Near deadline: accepts best opponent bid if above minimum (0.59)
        - Ends negotiation if no acceptable bid found and reservation > 0

    **Opponent Modeling:**
        Time-series prediction using:

        - Moving average of opponent's max utilities per time block
        - Residual analysis with standard deviation
        - Forecasts next time slot's expected opponent utility
        - Adapts concession curve to match/beat forecast

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Chen, S., & Weiss, G. (2012). An Efficient and Adaptive Approach to
        Negotiation in Complex Environments. In ECAI 2012.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2012.OMACagent.OMACagent"
        super().__init__(**kwargs)


class TheNegotiatorReloaded(GeniusNegotiator):
    """
    TheNegotiatorReloaded negotiation agent.

    **ANAC 2012**.

    An enhanced BOA-framework version of TheNegotiator using modular
    components for opponent modeling, offering strategy, and acceptance.

    **Architecture (BOA Framework):**
        - Opponent Model: IAMhagglerBayesianModel (if domain < 200K bids)
          or NoModel for large domains
        - OM Strategy: NullStrategy with threshold 0.35
        - Offering: TheNegotiatorReloaded_Offering strategy
        - Acceptance: Custom conditions with multiple parameters

    **Acceptance Strategy:**
        Parameterized acceptance with:

        - Alpha=1, Beta=0: basic threshold multipliers
        - Gamma=1.05: acceptance utility multiplier
        - Delta=0: time-based adjustment
        - Threshold bounds: [0.98, 0.99]

    **Opponent Modeling:**
        - Bayesian model (IAMhaggler variant) for small domains
        - Disabled for large domains (> 200K possible bids)
        - Used to estimate opponent preferences

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
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2012.TheNegotiatorReloaded.TheNegotiatorReloaded"
        super().__init__(**kwargs)
