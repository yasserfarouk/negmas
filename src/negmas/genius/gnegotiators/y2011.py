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
    """
    AgentK2 negotiation agent.

    **ANAC 2011**.

    An enhanced version of AgentK with improved discount factor handling.
    Uses statistical opponent modeling to estimate expected utility and
    probabilistic acceptance decisions.

    **Offering Strategy:**
        - Maintains a map of previously offered bids with utilities
        - Randomly selects from bids above current target utility
        - If no suitable cached bid exists, generates random bids meeting threshold
        - Target utility decreases over time using cubic time function: t³

    **Acceptance Strategy:**
        - Tracks opponent bid statistics (mean, variance) over all rounds
        - Calculates acceptance probability based on:
          - Estimated maximum opponent offer (mean + deviation adjustment)
          - Dynamic target utility that decreases with time
          - Discount factor ratio applied to targets
        - Accepts probabilistically when offer exceeds satisfaction threshold

    **Opponent Modeling:**
        Statistical approach tracking:

        - Running mean and variance of opponent utilities
        - Estimated deviation using √(variance × 12)
        - Used to predict expected maximum offer from opponent

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Kawaguchi, S., Fujita, K., & Ito, T. (2012). AgentK: Compromising
        strategy based on estimated maximum utility for automated negotiating
        agents. In New Trends in Agent-based Complex Automated Negotiations.
        Studies in Computational Intelligence, vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.AgentK2.Agent_K2"
        super().__init__(**kwargs)


class BramAgent(GeniusNegotiator):
    """
    BRAMAgent negotiation agent.

    **ANAC 2011**.

    A time-dependent negotiator that uses frequency-based opponent modeling
    and maintains a sorted array of candidate bids.

    **Offering Strategy:**
        - Pre-generates and sorts all candidate bids by own utility (descending)
        - Creates offers using opponent modeling (sampling from value frequencies)
        - Falls back to sorted bid array when modeled bids don't meet threshold
        - Avoids proposing the same bid too frequently (max 20% frequency)

    **Acceptance Strategy:**
        - Time-phased threshold with increasing flexibility:
          - 0-33%: threshold = max - 7% of range
          - 33-83%: threshold = max - 15% of range
          - 83-97%: threshold = max - 30% of range
          - 97-100%: threshold = max - 80% of range
        - Accepts if opponent offer meets threshold or exceeds planned counter-offer

    **Opponent Modeling:**
        Frequency-based learning from last 10 opponent bids:

        - Tracks value frequency per issue (discrete, real, integer)
        - Uses frequencies as probability distribution for bid generation
        - Rolling window: oldest bid removed when new one arrives

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Brzostowski, J., & Kowalczyk, R. (2006). Predicting partner's behaviour
        in agent negotiation. In Proceedings of AAMAS 2006.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class BramAgent2(GeniusNegotiator):
    """
    BRAMAgent2 negotiation agent.

    **ANAC 2011**.

    Identical to BramAgent - this is an alias wrapper pointing to the same
    Java implementation (agents.anac.y2011.BramAgent.BRAMAgent).

    See BramAgent for full documentation of the strategy.

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
        kwargs["java_class_name"] = "agents.anac.y2011.BramAgent.BRAMAgent"
        super().__init__(**kwargs)


class Gahboninho(GeniusNegotiator):
    """
    Gahboninho negotiation agent.

    **ANAC 2011**.

    A "bully" strategy agent that maintains high demands throughout most of
    the negotiation and only concedes significantly near the deadline. The
    agent exploits cooperative opponents by staying selfish when detecting
    niceness.

    **Offering Strategy:**
        - First 40 bids: gradually decreases from max utility to 0.925
          (allows opponent profiling)
        - After profiling: generates bids at recommended threshold
        - Uses opponent "noise" (niceness estimate) to adjust selfishness
        - Nicer opponents receive less generous offers
        - In "frenzy" mode (near deadline): offers best opponent bid seen

    **Acceptance Strategy:**
        - First 40 rounds: accepts only if utility > 0.95
        - Normal phase: accepts based on dynamic minimum threshold
        - Threshold stays high while opponent appears cooperative
        - Near deadline: accepts best opponent bid if reasonable

    **Opponent Modeling:**
        - Tracks opponent bid history and importance weights
        - Estimates "noise" as a niceness indicator
        - Filters bids late in negotiation based on opponent preferences
        - More selfish behavior toward nicer opponents

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
    """
    IAMhaggler2011 negotiation agent.

    **ANAC 2011**.

    A sophisticated agent using Gaussian Process regression to predict
    opponent behavior and optimize expected utility over time. Developed
    by the University of Southampton team.

    **Offering Strategy:**
        - Uses Bayesian Monte Carlo GP regression to model opponent concession
        - Computes expected utility surface over (time, utility) space
        - Factors in discount factor and risk parameter (default 3.0)
        - Targets utility that maximizes expected agreement value
        - Generates random bids within ±0.025 of target utility

    **Acceptance Strategy:**
        - Accepts if opponent utility × 1.02 >= own last bid utility
        - Accepts if opponent utility × 1.02 >= 0.9 (MAXIMUM_ASPIRATION)
        - Accepts if opponent utility × 1.02 >= planned counter-offer

    **Opponent Modeling:**
        Gaussian Process regression with:

        - Matern 3/2 covariance function + noise
        - Time-slotted sampling (36 slots) of opponent max utility
        - Predicts probability distribution of future opponent offers
        - Updates regression when time slot changes

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
    """
    TheNegotiator agent.

    **ANAC 2011**.

    A phase-based negotiation agent that divides the negotiation into
    distinct phases with different strategies. Uses time management and
    a bid generator to create offers.

    **Offering Strategy:**
        - Maintains a collection of all possible bids sorted by utility
        - Generates offers based on current phase and threshold
        - BidGenerator selects appropriate bids meeting phase requirements
        - Threshold varies by phase and time remaining

    **Acceptance Strategy:**
        - Phase-dependent acceptance thresholds
        - Phase 3 (endgame): considers moves left before deadline
        - Acceptor component evaluates opponent bids against threshold
        - More lenient acceptance as deadline approaches

    **Time Management:**
        - Divides negotiation into 3 phases
        - Tracks elapsed time and estimates moves remaining
        - Adjusts threshold based on discount factor
        - Phase transitions trigger strategy changes

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Dirkzwager, A., Hendrikx, M., & de Ruiter, J. (2013). TheNegotiator:
        A dynamic strategy for bilateral automated negotiation. In Complex
        Automated Negotiations: Theories, Models, and Software Competitions.
        Studies in Computational Intelligence, vol 435. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2011.TheNegotiator.TheNegotiator"
        super().__init__(**kwargs)


class ValueModelAgent(GeniusNegotiator):
    """
    ValueModelAgent negotiation agent.

    **ANAC 2011**.

    Uses temporal difference reinforcement learning to model opponent's
    utility space by learning utility loss per issue value. Slows
    concession rate until fairness requires compromise.

    **Offering Strategy:**
        - Maintains approved bids above current threshold (starts at 0.98)
        - Sorts approved bids by estimated opponent utility
        - Alternates between "best" scan (highest opponent utility) and
          "explore" scan (trying new value combinations)
        - Endgame "chicken game": waits, then offers opponent's best bid

    **Acceptance Strategy:**
        - Accepts if opponent utility > threshold and discounted, or > 0.975
        - Time-phased acceptance in final 10%:
          - 90-96%: accepts at lowered threshold
          - 96-99%: progressively more accepting
          - 99%+: accepts opponent's best if utility > 0.55
        - Adjusts threshold based on opponent concession

    **Opponent Modeling:**
        Temporal Difference learning approach:

        - Models utility loss per value in each issue
        - Uses reliability and standard deviation to split learning updates
        - Tracks opponent bid history and concession patterns
        - ValueSeperatedBids structure organizes bids by issue values

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
        kwargs["java_class_name"] = "agents.anac.y2011.ValueModelAgent.ValueModelAgent"
        super().__init__(**kwargs)
