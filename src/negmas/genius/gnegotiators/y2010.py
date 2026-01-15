"""Genius negotiator implementations - ANAC 2010 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentFSEGA",
    "AgentK",
    "AgentSmith",
    "IAMcrazyHaggler",
    "IAMhaggler",
    "Nozomi",
    "SouthamptonAgent",
    "Yushu",
]


class AgentFSEGA(GeniusNegotiator):
    """
    AgentFSEGA negotiation agent.

    **ANAC 2010 Finalist**.

    AgentFSEGA (developed by West University of Timisoara, Romania) uses Bayesian
    opponent modeling and time-dependent concession with three behavioral phases.

    **Offering Strategy:**
        - Starts with maximum utility bid
        - Maintains sorted list of bids above a minimum utility threshold (0.5)
        - Time-dependent behavior with three phases:
            - Phase 0 (t < 0.85): Conservative, stays within small sigma of best bid
            - Phase 1 (0.85 ≤ t < 0.95): Moderate concession
            - Phase 2 (t ≥ 0.95): More aggressive concession toward deadline
        - Selects bids that maximize estimated opponent utility within acceptable range

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility × 1.03 ≥ own last bid's utility
        - Accepts if opponent's bid utility > planned next bid's utility
        - Accepts opponent's first bid if it equals maximum possible utility

    **Opponent Modeling:**
        Bayesian learning approach:

        - Uses Bayesian opponent model to estimate opponent preferences
        - Updates beliefs based on opponent's bid history
        - Selects bids that are good for opponent among acceptable options

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Baarslag, T., Hindriks, K., Hendrikx, M., Dirkzwager, A., & Jonker, C.M. (2014).
        Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies.
        In Novel Insights in Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 535. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentFSEGA.AgentFSEGA"
        super().__init__(**kwargs)


class AgentK(GeniusNegotiator):
    """
    AgentK negotiation agent.

    **ANAC 2010 Winner**.

    AgentK uses adaptive concession based on statistical analysis of opponent
    behavior, with probabilistic acceptance decisions.

    **Offering Strategy:**
        - Maintains a target utility that decreases over time using cubic time function
        - Estimates opponent's maximum likely offer using mean + deviation
        - Adds randomness ("tremor") to bidding and acceptance to avoid predictability
        - Searches randomly for bids above the dynamic target utility
        - Caches and reuses previously generated bids above target

    **Acceptance Strategy:**
        - Uses probabilistic acceptance based on multiple factors:
            - Utility evaluation: how good the offer is compared to estimated max
            - Satisfaction: how close the offer is to current target
            - Time pressure: acceptance probability increases with t³
        - Never accepts with probability 1.0; always has stochastic element

    **Opponent Modeling:**
        Statistical approach:

        - Tracks running mean and variance of opponent's offers
        - Estimates maximum expected opponent utility as mean + sqrt(12 × variance)
        - Uses statistics to set adaptive target utilities
        - No explicit preference learning, relies on aggregate statistics

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Kawaguchi, S., Fujita, K., & Ito, T. (2011). AgentK: Compromising strategy
        based on estimated maximum utility for automated negotiating agents.
        In Complex Automated Negotiations: Theories, Models, and Software
        Competitions. Studies in Computational Intelligence, vol 435. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentK.Agent_K"
        super().__init__(**kwargs)


class AgentSmith(GeniusNegotiator):
    """
    AgentSmith negotiation agent.

    **ANAC 2010 Finalist**.

    AgentSmith (developed by Delft University of Technology) uses bid space
    sampling with opponent preference estimation to find near-Pareto bids.

    **Offering Strategy:**
        - Starts with maximum utility bid
        - Samples bid space and filters bids above utility threshold (0.7)
        - Sorts bids by combined utility (own + estimated opponent)
        - Iterates through sorted bids sequentially
        - Near deadline (>110s of assumed 180s session), offers best opponent bid

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility > 0.9 (margin threshold)
        - Accepts if opponent's bid utility ≥ own last bid's utility
        - Near deadline: accepts if opponent utility ≥ 0.7

    **Opponent Modeling:**
        Frequency-based preference learning:

        - Tracks opponent bid history
        - Estimates opponent's preference profile from bid patterns
        - Uses estimated preferences to sort bids by combined utility
        - Aims for Pareto-efficient outcomes

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Baarslag, T., Hindriks, K., Hendrikx, M., Dirkzwager, A., & Jonker, C.M. (2014).
        Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies.
        In Novel Insights in Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 535. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.AgentSmith.AgentSmith"
        super().__init__(**kwargs)


class IAMcrazyHaggler(GeniusNegotiator):
    """
    IAMcrazyHaggler negotiation agent.

    **ANAC 2010 Finalist** (University of Southampton).

    IAMcrazyHaggler is a simple but effective hardball strategy that randomly
    samples high-utility bids without any concession over time.

    **Offering Strategy:**
        - Generates random bids from the outcome space
        - Only offers bids with utility > 0.9 (or 0.95 in discounted domains)
        - No time-dependent concession - maintains hard position throughout
        - Completely ignores opponent's preferences and offers

    **Acceptance Strategy:**
        - Accepts if opponent's bid × 1.02 ≥ own last bid's utility
        - Accepts if opponent's bid × 1.02 ≥ 0.85 (maximum aspiration)
        - Adjusts thresholds slightly for discounted utility spaces

    **Opponent Modeling:**
        None - this agent does not model the opponent at all.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Williams, C.R., Robu, V., Gerding, E.H., & Jennings, N.R. (2011).
        An Overview of the Results and Insights from the First Automated
        Negotiating Agents Competition (ANAC 2010). In New Trends in Agent-Based
        Complex Automated Negotiations. Studies in Computational Intelligence,
        vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMcrazyHaggler"
        super().__init__(**kwargs)


class IAMhaggler(GeniusNegotiator):
    """
    IAMhaggler negotiation agent.

    **ANAC 2012 Winner** (Nash Product category).

    IAMhaggler (developed by Colin R. Williams at University of Southampton)
    uses Gaussian Process regression to predict opponent behavior and
    optimize concession timing.

    **Offering Strategy:**
        - Uses Gaussian Process to predict when opponent will make maximum
          concession
        - Calculates expected utility surface over time and utility dimensions
        - Target utility based on interpolation toward predicted best agreement
        - Limits concession based on observed opponent bidding range
        - Risk-aware utility function with configurable risk parameter

    **Acceptance Strategy:**
        - Accepts if opponent's offer * multiplier >= target utility
        - Accepts if opponent's offer * multiplier >= maximum aspiration (0.9)
        - Accepts if opponent's offer >= planned bid utility
        - Multiple acceptance thresholds for robustness

    **Opponent Modeling:**
        Gaussian Process regression approach:

        - Tracks opponent utilities over time slots
        - Fits GP model to predict future opponent concessions
        - Calculates probability distribution of opponent's future offers
        - Uses predictions to optimize concession timing
        - Incorporates discounting factor for time-sensitive domains

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Williams, C.R., Robu, V., Gerding, E.H., & Jennings, N.R. (2012).
        IAMhaggler: A Negotiation Agent for Complex Environments.
        In New Trends in Agent-Based Complex Automated Negotiations.
        Studies in Computational Intelligence, vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.IAMhaggler"
        super().__init__(**kwargs)


class Nozomi(GeniusNegotiator):
    """
    Nozomi negotiation agent.

    **ANAC 2010 Finalist**.

    Nozomi uses a sophisticated multi-strategy approach with adaptive behavior
    based on opponent responsiveness and negotiation progress.

    **Offering Strategy:**
        - Starts with maximum utility bid
        - Uses four bid types selected probabilistically:
            - COMPROMISE: Incrementally concede on one issue toward opponent's position
            - KEEP: Repeat previous bid to signal firmness
            - APPROACH: Move closer to opponent on multiple issues
            - RESTORE: Return to a saved "restore bid" (best bid close to opponent)
        - Maintains dynamic minimum utility threshold (maxCompromiseUtility)
        - Adapts concession based on opponent's reciprocity

    **Acceptance Strategy:**
        - Accepts if utility > 95% of maximum utility
        - Accepts if opponent's offer ≥ own previous bid's utility
        - Time-dependent acceptance with multiple phases:
            - t < 0.5: Accept good offers with strict utility threshold
            - 0.5 ≤ t < 0.8: Slightly more lenient acceptance
            - t ≥ 0.8: Accept if close to restore bid or previous position
        - Uses "evaluation gap" to measure bid similarity

    **Opponent Modeling:**
        Behavioral tracking approach:

        - Tracks opponent's best offer and updates maximum aspiration accordingly
        - Monitors opponent's concession patterns (continuous compromise tracking)
        - Detects if opponent is compromising and adjusts own strategy
        - Updates "restore bid" to track bids closest to opponent's best offer

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Baarslag, T., Hindriks, K., Hendrikx, M., Dirkzwager, A., & Jonker, C.M. (2014).
        Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies.
        In Novel Insights in Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 535. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Nozomi.Nozomi"
        super().__init__(**kwargs)


class SouthamptonAgent(GeniusNegotiator):
    """
    SouthamptonAgent negotiation agent.

    **ANAC 2010 Finalist** (University of Southampton).

    SouthamptonAgent is an abstract base class for Southampton agents that provides
    common infrastructure for opponent modeling and bid space analysis.

    **Offering Strategy:**
        - Abstract methods for initial and subsequent bids (implemented by subclasses)
        - Tracks own bidding history
        - Maintains bid space representation for analysis

    **Acceptance Strategy:**
        - Accepts if opponent's bid × 1.02 ≥ own last bid's utility
        - Accepts if opponent's bid × 1.02 ≥ maximum aspiration (0.9)
        - Accepts if opponent's bid × 1.02 ≥ planned next bid's utility
        - Detects if opponent is a "hardhead" (no significant concession)

    **Opponent Modeling:**
        General framework:

        - Maintains opponent model infrastructure
        - Tracks all opponent bids for analysis
        - Provides utilities for bid space analysis
        - Subclasses implement specific modeling strategies

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Williams, C.R., Robu, V., Gerding, E.H., & Jennings, N.R. (2011).
        An Overview of the Results and Insights from the First Automated
        Negotiating Agents Competition (ANAC 2010). In New Trends in Agent-Based
        Complex Automated Negotiations. Studies in Computational Intelligence,
        vol 383. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Southampton.SouthamptonAgent"
        super().__init__(**kwargs)


class Yushu(GeniusNegotiator):
    """
    Yushu negotiation agent.

    **ANAC 2010 Finalist**.

    Yushu uses time-aware concession with opponent bid tracking to adaptively
    adjust targets based on remaining negotiation rounds.

    **Offering Strategy:**
        - Starts with maximum utility bid
        - Calculates target utility using time-based concession: target = max - (max - min) × t^eagerness
        - Dynamically adjusts minimum acceptable utility based on:
            - Estimated remaining rounds (based on response time tracking)
            - Average opponent utility (domain competitiveness indicator)
        - Iterates through bid space to find bids near target utility
        - Near deadline, may suggest opponent's best historical offer

    **Acceptance Strategy:**
        - Accepts if opponent's utility ≥ target utility
        - Accepts if opponent's utility ≥ acceptable threshold (time-dependent)
        - More lenient when fewer rounds remain (< 8 rounds: accept good offers)
        - May accept opponent's best historical bid near deadline

    **Opponent Modeling:**
        Best-offer tracking:

        - Maintains sorted list of top 10 opponent bids by utility
        - Tracks average opponent utility to assess domain competitiveness
        - Uses opponent's best offers as fallback suggestions near deadline
        - Adjusts minimum acceptable utility based on opponent behavior

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Please refer to the original source code
        and papers for authoritative information.

    References:
        Baarslag, T., Hindriks, K., Hendrikx, M., Dirkzwager, A., & Jonker, C.M. (2014).
        Decoupling Negotiating Agents to Explore the Space of Negotiation Strategies.
        In Novel Insights in Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 535. Springer.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2010.Yushu.Yushu"
        super().__init__(**kwargs)
