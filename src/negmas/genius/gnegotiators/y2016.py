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
    """
    AgentHP2 negotiation agent.

    **ANAC 2016**.

    Evolution of AgentHP from ANAC 2015 by Hiroyuki Shinohara. Uses AHP
    (Analytic Hierarchy Process) for bid evaluation with utility space
    prediction for opponents.

    **Offering Strategy:**
        - Generates multiple bids within a utility range around threshold
        - Selects bid with highest AHP evaluation score
        - Utility threshold decreases over time based on opponent concession
        - Maximum concession degree limited to 0.35

    **Acceptance Strategy:**
        - Probabilistic acceptance based on AHP evaluation
        - Accepts if utility >= 0.90 or time >= 0.99
        - Accepts if utility >= current threshold
        - Considers discount factor in probability calculation

    **Opponent Modeling:**
        - Tracks opponent bids and updates utility space predictions
        - Uses AHP evaluation from opponent's perspective
        - Estimates opponent's maximum AHP evaluation for concession
        - Computes variance-based predictions of opponent behavior

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agenthp2.AgentHP2_main"
        super().__init__(**kwargs)


class AgentLight(GeniusNegotiator):
    """
    AgentLight negotiation agent.

    **ANAC 2016**.

    Uses learning phases with opponent statistics tracking. Identifies
    the "worthy opponent" for strategic targeting and employs tit-for-tat
    inspired concession.

    **Offering Strategy:**
        - Learning phase (first 10 rounds): offers maximum utility bids
        - Negotiation phase: uses tit-for-tat based concession
        - Tracks opponent statistics (mean, standard deviation) to guide offers
        - Selects bids that maximize estimated opponent utility above threshold

    **Acceptance Strategy:**
        - Accepts if opponent's offer utility exceeds time-dependent threshold
        - Near deadline (remaining rounds <= 3): uses discounted threshold
        - Prefers offers from "potential" bid list (accepted by others)

    **Opponent Modeling:**
        Per-opponent tracking with:

        - Bid history and value frequency tracking
        - Mean and standard deviation of offered utilities
        - Identification of "worthy opponent" (toughest/most cooperative)
        - Value weights based on frequency analysis

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agentlight.AgentLight"
        super().__init__(**kwargs)


class AgentSmith2016(GeniusNegotiator):
    """
    AgentSmith2016 negotiation agent.

    **ANAC 2016**.

    Modular agent with separate negotiationInfo, bidSearch, and
    negotiationStrategy components. Uses simulated annealing for bid search.

    **Offering Strategy:**
        - Uses random bid as seed for simulated annealing search
        - Searches for bids above time-dependent threshold
        - Threshold decreases gradually over negotiation

    **Acceptance Strategy:**
        - Accepts if opponent's bid exceeds current threshold
        - May end negotiation early based on reservation value

    **Opponent Modeling:**
        Frequency-based opponent modeling tracking issue value preferences
        to guide bid search toward mutually beneficial outcomes.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.agentsmith.AgentSmith2016"
        super().__init__(**kwargs)


class Atlas32016(GeniusNegotiator):
    """
    Atlas32016 negotiation agent.

    **ANAC 2016**.

    Updated version of ANAC 2015 winner Atlas3. Uses concession list
    strategy for final phase and tracks supporter counts for multilateral.

    **Offering Strategy:**
        - Time-dependent threshold with simulated annealing bid search
        - Final phase: cycles through "CList" (Concession List) of bids
          supported by other parties
        - Tracks time scale to determine when to enter final phase

    **Acceptance Strategy:**
        - Accepts if opponent bid exceeds threshold
        - Final phase: accepts bids above reservation value
        - May end negotiation if utility drops below reservation

    **Opponent Modeling:**
        - Tracks supporter count for each bid
        - Maintains PBList (Popular Bid List) of bids accepted by n-1 parties
        - Frequency-based preference estimation

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

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
    """
    ClockworkAgent negotiation agent.

    **ANAC 2016**.

    Time-phased strategy using cosine-based utility oscillation for bid
    generation and time-dependent acceptance thresholds.

    **Offering Strategy:**
        - Early phase (t < 0.3): offers maximum utility bids
        - Mid phase (0.3 <= t < 0.95): oscillating utility target using
          cosine function (rad increments by π/4 each round)
        - Late phase (0.95 <= t < 0.97): offers from candidate array of
          previously accepted bids
        - Final phase (t > 0.97): returns to maximum utility bids
        - Uses random search (10,000 iterations) to find bids above target

    **Acceptance Strategy:**
        - Time-phased thresholds with discount factor consideration
        - Early (t <= 0.95): accepts if utility > discounted 0.95
        - Mid (0.95 < t <= 0.97): accepts if utility > average of
          discounted value and bottom limit
        - Late (t > 0.97): accepts if utility > discounted value
        - Adjusts thresholds based on opponent concession detection

    **Opponent Modeling:**
        - Tracks opponent bid history
        - Detects concession by comparing recent bids (h1 > h2 or h1 > h3)
        - Stores bids accepted by others as candidates for late-phase offers

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.clockworkagent.ClockworkAgent"
        super().__init__(**kwargs)


class Farma(GeniusNegotiator):
    """
    Farma negotiation agent.

    **ANAC 2016**.

    Modular agent with negotiationInfo, bidSearch, and negotiationStrategy
    components. Uses shift-based bid search and tracks opponent behavior
    per sender.

    **Offering Strategy:**
        - Uses time-dependent threshold from negotiationStrategy
        - Generates bids via bidSearch with random seed and threshold
        - Shift-based search adjusts bids toward opponent preferences
        - Tracks own bid history and negotiating statistics

    **Acceptance Strategy:**
        - Time-dependent threshold-based acceptance
        - May terminate negotiation based on reservation value
        - Updates round counter each action

    **Opponent Modeling:**
        Per-opponent tracking including:

        - Bid accept counts per sender (CntBySender)
        - Offered value frequencies per issue
        - Separate initialization for new opponents
        - Round-based statistics updates

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.farma.Farma"
        super().__init__(**kwargs)


class GrandmaAgent(GeniusNegotiator):
    """
    GrandmaAgent negotiation agent.

    **ANAC 2016**.

    Proximity-based bidding strategy that generates random bids above a
    threshold and selects the one closest to opponents' "mean" preferences.
    Developed by Teo Cherici, Maarten de Vries, and Tim Resink.

    **Offering Strategy:**
        - Generates N random bids (default 15) above lower bound utility
        - Calculates proximity of each bid to opponents' normalized mean
        - Selects bid with highest proximity score
        - Lower bound utility decreases over time using exponential factor
        - Supports both discrete and integer issue types

    **Acceptance Strategy:**
        - Accepts if offered utility > lower bound utility AND
          offered utility > reservation value (MAU)
        - Lower bound starts at 0.95 and decreases based on discounted
          mean utility and time factor

    **Opponent Modeling:**
        Frequency-based tracking for all parties:

        - Counts how often each issue value is offered
        - Normalizes frequencies to compute mean preferences
        - For integer issues: tracks above/below median counts
        - Computes discounted mean utility with configurable sensitivity
        - Time-exponential factor adjusts convergence speed

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.grandma.GrandmaAgent"
        super().__init__(**kwargs)


class MaxOops(GeniusNegotiator):
    """
    MaxOops negotiation agent.

    **ANAC 2016**.

    Component-based agent by Max W. Y. Lam with TFComponent (threshold
    function), DMComponent (decision making), and OPTComponent modules.
    Uses comprehensive bid statistics and hash-based bid organization.

    **Offering Strategy:**
        - First offer: second-maximum utility bid (if gap is small, max bid)
        - Uses DMComponent.bidProposal() for subsequent offers
        - Organizes all possible bids in hash structure by utility (0-100)
        - Falls back to second-max bid if proposal fails

    **Acceptance Strategy:**
        - Immediately accepts max or second-max utility bids
        - DMComponent.termination() decides if negotiation should end
        - DMComponent.acceptance() decides if offer should be accepted
        - Considers reservation value when both accept and terminate trigger

    **Opponent Modeling:**
        Per-opponent tracking with extensive statistics:

        - BidHistory for each opponent with distinct bids and accepts
        - Tracks min/max distinct bids and accepts across opponents
        - Updates opponent payoff weights via OPTComponent
        - Computes domain statistics: mean, std, median, quartiles

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.maxoops.MaxOops"
        super().__init__(**kwargs)


class MyAgent(GeniusNegotiator):
    """
    MyAgent negotiation agent.

    **ANAC 2016**.

    Modular agent similar to AgentSmith2016 with negotiationInfo, bidSearch,
    and negotiationStrategy components. Uses slant analysis for opponent
    modeling and tracks issue weights per opponent.

    **Offering Strategy:**
        - Uses time-dependent threshold from negotiationStrategy
        - Generates bids via bidSearch with random seed
        - Tracks own bid history for consistency

    **Acceptance Strategy:**
        - Time-dependent threshold-based acceptance
        - May terminate negotiation based on time conditions
        - Updates "last" flag to track recent actions

    **Opponent Modeling:**
        Per-opponent tracking including:

        - Issue weight estimation updated on each offer
        - Accept list tracking which bids each opponent accepts
        - Slant analysis for predicting opponent behavior
        - Recent weight and max weight calculations
        - Maximum value tracking per opponent

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.myagent.MyAgent"
        super().__init__(**kwargs)


class Ngent(GeniusNegotiator):
    """
    Ngent negotiation agent.

    **ANAC 2016**.

    Developed by Tom and Tommy. Uses 3D space bid search to find Nash-like
    outcomes by minimizing distance to all parties' preferences. Features
    adaptive minimum utility thresholds and time-interval based concession.

    **Offering Strategy:**
        - First round: offers maximum utility bid
        - Searches bids in 3D space (own utility, opponent1 score, opponent2 score)
        - Selects bid minimizing squared distance to (1,1,1) point
        - Limits search to max 7000 bids for large domains
        - Uses log frequency for domains > 1000 bids, otherwise raw frequency

    **Acceptance Strategy:**
        - Accepts if utility >= average threshold or >= next round's utility
        - Compares discounted current maximum to predicted maximum
        - In final rounds (t > 0.99): accepts if utility > average threshold
        - May concede to opponent's best historical bid if conditions met

    **Opponent Modeling:**
        Separate AgentData for each opponent with:

        - Issue scoring using exponential decay (e^-t for additions)
        - Tracks utility maximums across three time sessions
        - Concession degree updates based on time intervals
        - Adaptive minimum utility calculation (different for small/large DF)
        - k parameter scales with domain size (1-4)

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.ngent.Ngent"
        super().__init__(**kwargs)


class ParsAgent2(GeniusNegotiator):
    """
    ParsAgent2 negotiation agent.

    **ANAC 2016**.

    Evolution of ParsAgent using k-means clustering (k=3) for opponent modeling.
    Extends AbstractTimeDependentNegotiationParty with adaptive concession
    rate based on cluster analysis.

    **Offering Strategy:**
        - Time-dependent target utility with adaptive concession rate
        - Identifies mutual issues between opponents for bid construction
        - Uses getNNBid to find nearest neighbor bid from accepted history
        - In final rounds: may offer best mutual cluster center

    **Acceptance Strategy:**
        - Accepts if offered utility >= target utility
        - Target utility never drops below ConstantUtility (default 0.8)
        - Near deadline: adjusts constant utility based on remaining rounds
        - Uses probability function to evaluate bid acceptance likelihood

    **Opponent Modeling:**
        K-means clustering (k=3) per opponent:

        - Clusters opponent bids every N rounds (slotNum=100)
        - Tracks cluster history with center movements
        - Calculates distribution factor (deviation from uniform)
        - Computes concession rate from cluster analysis:
          w1*maxCenterDistance + w2*meanDistance - w3*meanDistribution
        - Identifies mutual issues across both opponents

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

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
    """
    ParsCat2 negotiation agent.

    **ANAC 2016**.

    Alias for ParsCat agent - uses the same underlying Java implementation.
    See ParsCat for full documentation of the strategy.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.parscat.ParsCat"
        super().__init__(**kwargs)


class SYAgent(GeniusNegotiator):
    """
    SYAgent negotiation agent.

    **ANAC 2016**.

    Modular agent with negotiationInfo, bidSearch, and negotiationStrategy
    components. Tracks accepted bids per opponent and uses round counting
    for time management.

    **Offering Strategy:**
        - Uses time-dependent threshold from negotiationStrategy
        - Generates bids via bidSearch with random seed
        - Tracks own bid history

    **Acceptance Strategy:**
        - Time-dependent threshold-based acceptance
        - May terminate negotiation based on time conditions
        - Considers discount factor in utility calculations

    **Opponent Modeling:**
        Per-opponent tracking including:

        - Records which bids were accepted by whom (updateAcceptedBid)
        - Standard negotiationInfo for bid history
        - Round counter for timing
        - Separate initialization for new opponents

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Aydogan, R., et al. (2017). The Seventh International Automated Negotiating
        Agents Competition (ANAC 2016). Studies in Computational Intelligence.
        Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2016.syagent.SYAgent"
        super().__init__(**kwargs)


class Terra(GeniusNegotiator):
    """
    Terra negotiation agent.

    **ANAC 2016**.

    Terra uses a modular architecture with separate components for information
    tracking, bid search, and negotiation strategy. It maintains per-opponent
    lists of agreed and disagreed values to inform its bidding decisions.

    **Offering Strategy:**
        - Uses `bidSearch.getBid()` with a random seed and time-dependent threshold
        - Threshold obtained via `negotiationStrategy.getThreshold(time)`
        - Can end negotiation early via `selectEndNegotiation(time)` check

    **Acceptance Strategy:**
        - Accepts offers above the current threshold from negotiationStrategy
        - May terminate negotiation early based on strategic assessment

    **Opponent Modeling:**
        - Tracks agreed values list per opponent (values they accepted)
        - Tracks disagreed values list per opponent (values they rejected)
        - Tracks all offered bids per opponent
        - Updates opponent information on each received message

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2016). ANAC 2016 Individual Utility category analysis.
        In Proceedings of IJCAI Workshop on Autonomous Agents.
    """

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
