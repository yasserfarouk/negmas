"""Genius negotiator implementations - ANAC 2016 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentHP2",
    "AgentLight",
    "AgentSmith2016",
    "Atlas32016",
    "Caduceus",
    "ClockworkAgent",
    "Farma",
    "GrandmaAgent",
    "MaxOops",
    "MyAgent",
    "Ngent",
    "ParsAgent2",
    "ParsCat",
    "ParsCat2",
    "SYAgent",
    "Terra",
    "YXAgent",
]


class AgentHP2(GeniusNegotiator):
    """AgentHP2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agenthp2.AgentHP2_main"
        super().__init__(**kwargs)


class AgentLight(GeniusNegotiator):
    """AgentLight implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agentlight.AgentLight"
        super().__init__(**kwargs)


class AgentSmith2016(GeniusNegotiator):
    """AgentSmith2016 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agentsmith.AgentSmith2016"
        super().__init__(**kwargs)


class Atlas32016(GeniusNegotiator):
    """Atlas32016 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.atlas3.Atlas32016"
        super().__init__(**kwargs)


class Caduceus(GeniusNegotiator):
    """
    Caduceus negotiation agent.

    **ANAC 2016 Winner** (Individual Utility category).

    Caduceus (developed by Taha Gunes) combines multiple negotiation experts
    using ideas from algorithm portfolios, mixture of experts, and genetic
    algorithms to make collective decisions.

    **Offering Strategy:**
        - Portfolio of 5 expert agents: ParsAgent, RandomDance, Kawaii,
          Atlas3, and Caduceus2015
        - Early phase (t < 0.83): offers the best possible bid
        - Crossover strategy: each expert suggests a bid, then majority
          voting on each issue value determines final bid content
        - Experts are weighted by expertise scores (100, 10, 5, 3, 1)
        - Stochastic selection based on expertise levels

    **Acceptance Strategy:**
        - Weighted voting among expert agents
        - Accepts if weighted score of "accept" votes exceeds "bid" votes
        - Each expert's vote weighted by its expertise score

    **Opponent Modeling:**
        Delegated to individual expert agents in the portfolio:

        - ParsAgent, Atlas3, Kawaii each have their own opponent models
        - Collective decision benefits from diverse modeling approaches

    References:
        Gunes, T.D., Arditi, E., & Aydogan, R. (2017). Collective Voice of
        Experts in Multilateral Negotiation. In PRIMA 2017: Principles and
        Practice of Multi-Agent Systems. Lecture Notes in Computer Science,
        vol 10621. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.caduceus.Caduceus"
        super().__init__(**kwargs)


class ClockworkAgent(GeniusNegotiator):
    """ClockworkAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.clockworkagent.ClockworkAgent"
        super().__init__(**kwargs)


class Farma(GeniusNegotiator):
    """Farma implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.farma.Farma"
        super().__init__(**kwargs)


class GrandmaAgent(GeniusNegotiator):
    """GrandmaAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.grandma.GrandmaAgent"
        super().__init__(**kwargs)


class MaxOops(GeniusNegotiator):
    """MaxOops implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.maxoops.MaxOops"
        super().__init__(**kwargs)


class MyAgent(GeniusNegotiator):
    """MyAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.myagent.MyAgent"
        super().__init__(**kwargs)


class Ngent(GeniusNegotiator):
    """Ngent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.ngent.Ngent"
        super().__init__(**kwargs)


class ParsAgent2(GeniusNegotiator):
    """ParsAgent2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.pars2.ParsAgent2"
        super().__init__(**kwargs)


class ParsCat(GeniusNegotiator):
    """
    ParsCat negotiation agent.

    **ANAC 2016 Individual Utility Category Runner-up**.

    ParsCatAgent is developed by Amirkabir University of Technology and uses
    a time-dependent bidding strategy with a complex piecewise acceptance
    function. The agent maintains a history of opponent bids and uses
    time-based thresholds that vary across different negotiation phases.

    **Offering Strategy:**
        The agent generates random bids within a narrow utility window around
        a time-varying threshold. The threshold decreases over time but with
        different rates across negotiation phases:

        - t < 0.5: threshold = 1.0 - t/4 (slow concession)
        - 0.5 <= t < 0.8: threshold = 0.9 - t/5
        - 0.8 <= t < 0.9: threshold = 0.7 + t/5 (strategic increase)
        - 0.9 <= t < 0.95: threshold = 0.8 + t/5
        - t >= 0.95: threshold = 1.0 - t/4 - 0.01

        The search window around the threshold is typically +/- 0.01 to 0.02.
        If the best opponent bid has higher utility than the generated bid
        (in bilateral negotiations), it returns the opponent's best bid.

    **Acceptance Strategy:**
        Uses a complex piecewise function based on negotiation time with 10
        distinct phases, creating an oscillating acceptance threshold:

        - Starts high (1.0), drops to ~0.9 by t=0.25
        - Oscillates between 0.7-1.0 through mid-game
        - Ends around 0.5-0.7 in the final phase

        This non-monotonic pattern makes the agent's behavior harder to
        predict and exploit.

    **Opponent Modeling:**
        Maintains a history of opponent bids with utilities and timestamps.
        Uses the best bid from opponent history as a fallback offer when
        the generated bid has lower utility.

    References:
        Aydogan, R., Fujita, K., Baarslag, T., Jonker, C. M., & Ito, T. (2021).
        ANAC 2017: Repeated multilateral negotiation league.
        In Advances in Automated Negotiations (pp. 101-115). Springer Singapore.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class ParsCat2(GeniusNegotiator):
    """ParsCat2 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class SYAgent(GeniusNegotiator):
    """SYAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.syagent.SYAgent"
        super().__init__(**kwargs)


class Terra(GeniusNegotiator):
    """Terra implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.terra.Terra"
        super().__init__(**kwargs)


class YXAgent(GeniusNegotiator):
    """
    YXAgent negotiation agent.

    **ANAC 2016 Individual Utility Category Runner-up**.

    YXAgent employs a frequency-based opponent modeling approach combined with
    threshold-based bidding and acceptance strategies. The agent maintains
    separate models for issue weights and value frequencies for each opponent,
    identifying the "toughest" opponent to inform its acceptance decisions.

    **Offering Strategy:**
        The agent generates random bids above a threshold that is calculated
        based on the number of opponents (minimum 0.7). In early rounds, it
        offers bids with utility above a temporary threshold. After 10 rounds
        and before 90% of the time has elapsed, it uses the opponent model to
        calculate a more nuanced threshold based on the estimated utility of
        the opponent's last bid according to the toughest opponent's model.

    **Acceptance Strategy:**
        - In early rounds (first 10) or late game (>90% time): accepts if the
          opponent's offer utility exceeds the temporary threshold.
        - During mid-game: accepts if the opponent's offer utility exceeds a
          calculated threshold that accounts for the opponent model's
          evaluation of the bid.

    **Opponent Modeling:**
        YXAgent builds frequency-based models for each opponent:

        - **Issue weights**: Updated when the opponent keeps the same value
          for an issue between consecutive bids, using a time-decaying formula.
        - **Value frequencies**: Tracks how often each value is offered,
          normalized by the maximum frequency.
        - Identifies the "hardest" (toughest) opponent based on their behavior.

    References:
        Aydogan, R., Fujita, K., Baarslag, T., Jonker, C. M., & Ito, T. (2021).
        ANAC 2017: Repeated multilateral negotiation league.
        In Advances in Automated Negotiations (pp. 101-115). Springer Singapore.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.yxagent.YXAgent"
        super().__init__(**kwargs)
