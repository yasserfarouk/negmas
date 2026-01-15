"""Genius negotiator implementations - ANAC 2015 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentBuyog",
    "AgentH",
    "AgentHP",
    "AgentNeo",
    "AgentW",
    "AgentX",
    "AresParty",
    "Atlas3",
    "CUHKAgent2015",
    "DrageKnight",
    "Group2",
    "JonnyBlack",
    "Kawaii",
    "MeanBot",
    "Mercury",
    "PNegotiator",
    "ParsAgent",
    "PhoenixParty",
    "PokerFace",
    "RandomDance",
    "SENGOKU",
    "TUDMixedStrategyAgent",
    "XianFaAgent",
    "Y2015Group2",
]


class AgentBuyog(GeniusNegotiator):
    """
    AgentBuyog negotiation agent.

    **ANAC 2015 Individual Utility Category Runner-up**.

    AgentBuyog estimates the opponent's concession function using regression
    analysis and uses this to determine optimal acceptance thresholds. It
    also estimates opponent preferences to find bids near the Kalai point
    (social welfare maximum).

    **Offering Strategy:**
        Selects bids based on domain competitiveness and opponent difficulty:

        1. Calculates acceptance threshold based on estimated opponent
           difficulty and time-based concession:

           threshold = minPoint + (1 - minPoint) * (1 - t^1.8)

        2. Searches for bids at or above threshold that are closest to
           the estimated Kalai point (maximizing social welfare)
        3. If common bids exist (offered by multiple opponents), prefers
           those with highest utility

        Near deadline (remaining rounds <= 3), threshold is halved.

    **Acceptance Strategy:**
        Multi-criteria acceptance based on:

        - Most recent offer utility vs. threshold
        - Best agreeable bid utility (common to both opponents)
        - Generated bid utility

        Accepts if opponent's offer exceeds all other options and the
        acceptance threshold, especially when near deadline.

    **Opponent Modeling:**
        Sophisticated multi-component model:

        - **Concession estimation**: Uses weighted regression to fit
          exponential concession function exp(alpha) * t^beta to opponent
          bid utilities over time
        - **Leniency calculation**: Derived from slope of estimated
          concession curve, adjusted by a leniency factor
        - **Preference estimation**: Frequency-based issue weight and
          value estimation, normalized after each update
        - **Kalai point estimation**: Finds social welfare maximum
          using estimated opponent preferences
        - **Agent difficulty**: Combined metric of leniency and domain
          competitiveness to assess how hard to negotiate with

    References:
        Fujita, K., Aydogan, R., Baarslag, T., Hindriks, K., Ito, T., &
        Jonker, C. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). In Fujita, K., et al. Modern Approaches to Agent-based
        Complex Automated Negotiation. Studies in Computational Intelligence,
        vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.agentBuyogV2.AgentBuyogMain"
        super().__init__(**kwargs)


class AgentH(GeniusNegotiator):
    """
    AgentH negotiation agent.

    **ANAC 2015**.

    Uses relative utility search and bid history analysis.

    **Offering Strategy:**
        - Relative utility search based on time progression
        - Falls back to history-based bid generation
        - Concedes gradually as time progresses

    **Acceptance Strategy:**
        - Time-dependent acceptance: accepts if utility * time < 0.45
        - Simple threshold-based decision

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.agenth.AgentH"
        super().__init__(**kwargs)


class AgentHP(GeniusNegotiator):
    """
    AgentHP negotiation agent.

    **ANAC 2015**.

    Uses Analytic Hierarchy Process (AHP) for bid evaluation with pairwise
    comparison of issues. Tracks opponent behavior using frequency analysis
    and geometric mean calculations.

    **Offering Strategy:**
        - Generates bids within a utility range based on time-dependent threshold
        - Selects bids that maximize AHP evaluation score
        - AHP weights derived from pairwise comparison matrix using geometric mean

    **Acceptance Strategy:**
        - Acceptance probability based on opponent concession rate and AHP scores
        - Considers discounted utility when time pressure increases

    **Opponent Modeling:**
        Frequency-based tracking per opponent:

        - Tracks issue value frequencies across opponent bids
        - Computes issue weights using geometric mean of frequencies
        - Updates AHP comparison matrix based on observed preferences

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.AgentHP.AgentHP"
        super().__init__(**kwargs)


class AgentNeo(GeniusNegotiator):
    """
    AgentNeo negotiation agent.

    **ANAC 2015**.

    Discount-aware agent that precomputes bids organized by utility ranges.
    Uses similarity-based bid selection with multiple choice methods.

    **Offering Strategy:**
        - Starts with maximum utility bids for first 7 rounds
        - Precomputes all possible bids and organizes them by utility ranges
        - Uses three bid selection methods (ChooseBid1/2/3) based on similarity
          to opponent's previous offers
        - Considers discount factor when calculating target utility

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility >= current threshold
        - Accepts if opponent's bid utility >= utility of next planned offer
        - Threshold decreases over time accounting for discount factor

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.AgentNeo.Groupn"
        super().__init__(**kwargs)


class AgentW(GeniusNegotiator):
    """
    AgentW negotiation agent.

    **ANAC 2015**.

    Modular agent with separate components for negotiation info tracking,
    bid searching, and strategy. Uses opponent's last bid as seed for search.

    **Offering Strategy:**
        - Modular design with negotiatingInfo, bidSearch, strategy components
        - Time-dependent utility threshold that decreases over negotiation
        - Uses opponent's last bid as seed for bid search
        - Searches for bids above threshold that may appeal to opponent

    **Acceptance Strategy:**
        - Time-dependent threshold comparison
        - Accepts if opponent's offer exceeds current threshold

    **Opponent Modeling:**
        Frequency-based model tracking opponent bidding patterns to guide
        bid generation toward mutually acceptable outcomes.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.AgentW.AgentW"
        super().__init__(**kwargs)


class AgentX(GeniusNegotiator):
    """
    AgentX negotiation agent.

    **ANAC 2015**.

    Modular agent similar to AgentW but uses simulated annealing for bid
    search with random bid seeding.

    **Offering Strategy:**
        - Modular design with negotiatingInfo, bidSearch, strategy components
        - Simulated annealing bid search for finding near-optimal bids
        - Uses random bid as initial seed (unlike AgentW's opponent-based seed)
        - Time-dependent concession following threshold curve

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility exceeds current threshold
        - Threshold decreases over time based on negotiation progress

    **Opponent Modeling:**
        Frequency-based opponent model that tracks issue value frequencies
        to estimate opponent preferences and guide bid search.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.AgentX.AgentX"
        super().__init__(**kwargs)


class AresParty(GeniusNegotiator):
    """
    AresParty negotiation agent.

    **ANAC 2015**.

    Discount-aware agent with time estimation for remaining rounds and
    adaptive concession strategy. Uses toughness parameter for concession control.

    **Offering Strategy:**
        - Estimates remaining negotiation rounds based on elapsed time
        - Precomputes bids organized by utility ranges for fast selection
        - Concession controlled by alpha1=2 toughness parameter
        - Discount-aware: adjusts target utility based on discount factor

    **Acceptance Strategy:**
        - Terminates if reservation value with discount > predicted maximum
        - Accepts bids above time-dependent threshold
        - Threshold calculation considers discount factor and remaining time

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.AresParty.AresParty"
        super().__init__(**kwargs)


class Atlas3(GeniusNegotiator):
    """
    Atlas3 negotiation agent.

    **ANAC 2015 Winner** (Individual Utility & Nash Product categories).

    Atlas3 is a sophisticated negotiation agent developed by Akiyuki Mori that
    uses adaptive strategies based on opponent behavior analysis and game-theoretic
    concepts.

    **Offering Strategy:**
        - Uses appropriate bid searching based on relative utility for linear
          utility spaces
        - Applies replacement method based on frequency analysis of opponent's
          bidding history
        - Concession function derived from Evolutionary Stable Strategy (ESS)
          expected utility analysis
        - Near deadline: cycles through promising bids from opponent's history

    **Acceptance Strategy:**
        - Accepts if the offer utility exceeds the current threshold calculated
          from ESS-based concession function
        - Near deadline: accepts bids above reservation value from candidate list

    **Opponent Modeling:**
        Frequency-based model that tracks opponent's bidding patterns to:
        - Estimate opponent preferences
        - Identify promising bids that might be acceptable to both parties
        - Guide bid search towards mutually beneficial outcomes

    References:
        Mori, A., Ito, T. (2017). Atlas3: A Negotiating Agent Based on Expecting
        Lower Limit of Concession Function. In: Modern Approaches to Agent-based
        Complex Automated Negotiation. Studies in Computational Intelligence,
        vol 674. Springer, Cham. https://doi.org/10.1007/978-3-319-51563-2_11
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.Atlas3.Atlas3"
        super().__init__(**kwargs)


class CUHKAgent2015(GeniusNegotiator):
    """
    CUHKAgent2015 negotiation agent.

    **ANAC 2015**.

    Sophisticated agent from Chinese University of Hong Kong with dual-opponent
    modeling. Tracks both opponents separately with statistical analysis over
    multiple time intervals.

    **Offering Strategy:**
        - Distance-based bid selection to balance self and opponent utilities
        - Gaussian probability distribution for utility target selection
        - Adapts strategy based on time intervals (First/Second/ThirdTimeInterval)
        - Considers both opponents' estimated preferences in bid selection

    **Acceptance Strategy:**
        - Multi-criteria acceptance considering both opponents' behavior
        - Statistical analysis of opponent concession patterns over intervals
        - More lenient acceptance near deadline

    **Opponent Modeling:**
        Dual-opponent tracking with sophisticated statistical analysis:

        - Separate models for each opponent in multilateral negotiation
        - Divides negotiation into time intervals for trend analysis
        - Tracks concession patterns, mean utilities, and variance
        - Uses statistical measures to predict opponent behavior

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.cuhkagent2015.CUHKAgent2015"
        super().__init__(**kwargs)


class DrageKnight(GeniusNegotiator):
    """
    DrageKnight negotiation agent.

    **ANAC 2015**.

    Modular agent similar to AgentW/X family, using simulated annealing
    for bid search with frequency-based opponent modeling.

    **Offering Strategy:**
        - Modular design with strategy, bidSearch, negotiatingInfo components
        - Simulated annealing search for bid generation
        - Uses getThreshold2 method for threshold calculation
        - Time-dependent concession strategy

    **Acceptance Strategy:**
        - Threshold-based acceptance using getThreshold2
        - Accepts if opponent bid exceeds calculated threshold

    **Opponent Modeling:**
        Frequency-based model tracking opponent issue value preferences
        to guide bid search toward mutually beneficial outcomes.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.DrageKnight.DrageKnight"
        super().__init__(**kwargs)


class Group2(GeniusNegotiator):
    """
    Group2 negotiation agent.

    **ANAC 2015**.

    Uses Pareto-optimal bid search with per-party opponent modeling.
    Estimates remaining rounds to adapt concession timing.

    **Offering Strategy:**
        - Pareto-optimal bid search using G2ParetoFinder class
        - Searches Pareto frontier for bids above minimum utility threshold
        - Time-dependent minimum utility with round estimation
        - Adapts bid selection based on negotiation progress

    **Acceptance Strategy:**
        - Accepts if opponent bid exceeds time-dependent threshold
        - Threshold considers estimated remaining rounds
        - More lenient as deadline approaches

    **Opponent Modeling:**
        Per-party opponent modeling tracking each opponent's preferences
        separately to find Pareto-optimal bids acceptable to all parties.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.group2.Group2"
        super().__init__(**kwargs)


class JonnyBlack(GeniusNegotiator):
    """
    JonnyBlack negotiation agent.

    **ANAC 2015**.

    Enumerates feasible bids and uses opponent preference prediction with
    round-robin fairness in multilateral scenarios.

    **Offering Strategy:**
        - Enumerates all feasible bids with utility > finalStopVal (0.6)
        - Selects bids based on estimated opponent preferences
        - Round-robin opponent favoring: cycles agentToFavor to ensure
          fairness in multilateral negotiations
        - Balances own utility with opponent satisfaction

    **Acceptance Strategy:**
        - Accepts bids above the minimum utility threshold (0.6)
        - Considers opponent preferences in acceptance decision

    **Opponent Modeling:**
        Frequency-based preference prediction:

        - Counts value frequencies across opponent bids
        - Estimates opponent utility function from frequency counts
        - Uses predictions to favor bids likely acceptable to opponents

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.JonnyBlack.JonnyBlack"
        super().__init__(**kwargs)


class Kawaii(GeniusNegotiator):
    """
    Kawaii negotiation agent.

    **ANAC 2015 Individual Utility Category Runner-up**.

    Kawaii is a negotiation agent that uses Simulated Annealing for bid
    search and a time-dependent conceding strategy. It adapts its
    acceptance threshold based on the number of accepting opponents in
    multilateral negotiations.

    **Offering Strategy:**
        Uses Simulated Annealing to search for bids near the target utility:

        1. First attempts relative utility search (for linear utility spaces)
           by selecting values that sum to the target concession amount
        2. Falls back to Simulated Annealing with parameters:
           - Start temperature: 1.0
           - End temperature: 0.0001
           - Cooling rate: 0.999

        The search minimizes the distance to target utility while staying
        above it. Returns the maximum utility bid if no suitable bid found.

    **Acceptance Strategy:**
        Time-dependent threshold with conceder behavior (exponent = 2):

        threshold(t) = 1 - (1 - a) * t^2

        where a = 0.8 is the minimum threshold.

        In multilateral scenarios, the threshold is reduced based on how
        many opponents have already accepted:

        threshold -= (threshold - minThreshold) * (acceptCount / numOpponents)

        This encourages acceptance when close to agreement.

    **Opponent Modeling:**
        Tracks which opponents have made accepting moves (offered bids
        close to previous offers). This information adjusts the acceptance
        threshold to facilitate agreement when multiple parties are close
        to consensus.

    References:
        Baarslag, T., Aydogan, R., Hindriks, K. V., Fujita, K., Ito, T., &
        Jonker, C. M. (2015). The Automated Negotiating Agents Competition,
        2010-2015. AI Magazine, 36(4), 115-118.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.fairy.kawaii"
        super().__init__(**kwargs)


class MeanBot(GeniusNegotiator):
    """
    MeanBot negotiation agent.

    **ANAC 2015**.

    Extremely simple hardball agent. Always offers maximum utility bid
    and only accepts near deadline. No opponent modeling.

    **Offering Strategy:**
        - Always offers the maximum utility bid
        - No concession throughout negotiation
        - Pure hardball/take-it-or-leave-it approach

    **Acceptance Strategy:**
        - Only considers acceptance after t >= 0.95 (95% of time elapsed)
        - Accepts only if opponent's bid utility > 0.5
        - Very restrictive acceptance criteria

    **Opponent Modeling:**
        None. This agent ignores opponent behavior entirely.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.meanBot.MeanBot"
        super().__init__(**kwargs)


class Mercury(GeniusNegotiator):
    """
    Mercury negotiation agent.

    **ANAC 2015**.

    Extended version of AresParty with enhanced multi-party support.
    Tracks acceptance signals and adapts offers accordingly.

    **Offering Strategy:**
        - Inherits AresParty's discount-aware bid generation
        - Tracks party order in multilateral negotiations
        - When one opponent accepts (halfSucc flag), offers "nice bids"
          in a lower utility range to close the deal
        - Adapts strategy based on acceptance signals from opponents

    **Acceptance Strategy:**
        - Similar to AresParty with discount-aware thresholds
        - More willing to accept when close to agreement (halfSucc=true)

    **Opponent Modeling:**
        Tracks acceptance signals per opponent:

        - Monitors which opponents have shown willingness to accept
        - Uses "halfSucc" flag to indicate partial agreement state
        - Adjusts offers to facilitate final agreement

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.Mercury.Mercury"
        super().__init__(**kwargs)


class PNegotiator(GeniusNegotiator):
    """
    PNegotiator negotiation agent.

    **ANAC 2015**.

    Two-state agent (HARDLINER/CONCEDER) with Bayesian opponent modeling
    using the BayesLogic class for preference estimation.

    **Offering Strategy:**
        Two distinct phases:

        - HARDLINER (t < 0.2): Offers maximum utility bids, no concession
        - CONCEDER (t >= 0.2): Gradual concession using formula:
          utility = (maxUtil - Cj/numIssues)² where Cj = maxUtil * t²

    **Acceptance Strategy:**
        - In HARDLINER phase: only accepts very high utility bids
        - In CONCEDER phase: accepts based on calculated concession utility

    **Opponent Modeling:**
        Bayesian opponent modeling via BayesLogic class:

        - Maintains probability distributions over opponent preferences
        - Updates beliefs based on observed bids
        - Uses Bayesian inference for preference estimation

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.pnegotiator.PNegotiator"
        super().__init__(**kwargs)


class ParsAgent(GeniusNegotiator):
    """
    ParsAgent negotiation agent.

    **ANAC 2015 Individual Utility Category Finalist**.

    ParsAgent by Zahra Khosravimehr from Amirkabir University of Technology
    uses a hybrid bidding strategy combining time-dependent, random, and
    frequency-based approaches to propose high-utility offers close to
    opponent preferences, increasing the likelihood of early agreement.

    **Offering Strategy:**
        Employs a multi-step bid generation process:

        1. First checks if there's a mutually beneficial bid from the
           intersection of both opponents' preferences (in multilateral).
        2. If not found, constructs a bid using:
           - Mutual issue values (agreed by both opponents based on frequency)
           - Own best values for non-mutual issues
        3. Falls back to modifying the best bid on the worst-weighted issue

        Target utility follows Boulware-style concession:

        target(t) = 1 - t^(1/e)

        where e = 0.15 (or 0.2 with discount factor). Minimum threshold is 0.7.

    **Acceptance Strategy:**
        Simple time-dependent acceptance: accepts if opponent's offer
        utility exceeds the target utility at current time. The target
        decreases from 1.0 towards 0.7 following the Boulware curve.

    **Opponent Modeling:**
        Frequency-based modeling for each opponent:

        - Tracks repeated values for each issue across opponent bids
        - Identifies mutual preferences between opponents (values both
          opponents frequently request)
        - Maintains sorted list of opponent bids by utility
        - Searches for Nash-like outcomes using common preferences

    References:
        Khosravimehr, Z., & Nassiri-Mofakham, F. (2017). Pars Agent: Hybrid
        Time-Dependent, Random and Frequency-Based Bidding and Acceptance
        Strategies in Multilateral Negotiations. In Fujita, K., et al.
        Modern Approaches to Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.ParsAgent.ParsAgent"
        super().__init__(**kwargs)


class PhoenixParty(GeniusNegotiator):
    """
    PhoenixParty negotiation agent.

    **ANAC 2015**.

    Advanced agent using Gaussian Process regression for opponent modeling
    to predict opponent concession behavior. Features complex parameter tuning.

    **Offering Strategy:**
        - Uses Gaussian Process predictions for opponent concession timing
        - Complex parameter set (alpha, beta, omega, epsilon, xi, gamma)
          for fine-tuned strategy control
        - Frequency-based heuristic ranking of bids
        - Discount-aware with reservation value checks

    **Acceptance Strategy:**
        - Accepts based on predicted opponent behavior from GP model
        - Checks against reservation value considering discount factor
        - Uses multiple parameters to balance acceptance criteria

    **Opponent Modeling:**
        Gaussian Process (GP) regression - unique among ANAC 2015 agents:

        - Models opponent concession as a function over time
        - Predicts future opponent utility targets
        - Uncertainty quantification in predictions
        - Combined with frequency-based heuristic ranking

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.Phoenix.PhoenixParty"
        super().__init__(**kwargs)


class PokerFace(GeniusNegotiator):
    """
    PokerFace negotiation agent.

    **ANAC 2015**.

    Two-phase strategy: random walking among high-utility bids early,
    then concedes based on opponent bid frequency analysis.

    **Offering Strategy:**
        Two phases:

        - Early phase (t < 0.6): Random walker selecting among high-utility
          bids (utility > 0.85) to avoid revealing preferences
        - Late phase (t >= 0.6): Concedes using opponent bid frequency
          analysis with binary concede bid generation

    **Acceptance Strategy:**
        - Early phase: strict, only accepts very high utility
        - Late phase: uses moving average for time estimation
        - More lenient acceptance as deadline approaches

    **Opponent Modeling:**
        Frequency analysis of opponent bids:

        - Tracks bid frequencies over negotiation history
        - Uses moving average to estimate time per round
        - Analyzes opponent patterns to guide concession

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.pokerface.PokerFace"
        super().__init__(**kwargs)


class RandomDance(GeniusNegotiator):
    """
    RandomDance negotiation agent.

    **ANAC 2015 Individual Utility Category Finalist**.

    RandomDance Agent by Shinji Kakimoto proposes an opponent modeling
    approach using multiple weighted utility estimation functions. The
    agent randomly selects among different weighting schemes, making it
    unpredictable while still being responsive to opponent behavior.

    **Offering Strategy:**
        Searches for bids that balance self-utility with estimated opponent
        utility using weighted combination:

        1. For each issue, selects values that maximize weighted sum of
           estimated utilities across all parties
        2. Adjusts own weight iteratively (0 to 10) until finding a bid
           above the target utility
        3. Target utility is adaptive: starts at estimated Nash point and
           decreases following t^discount curve

        Falls back to best bid if no suitable bid found.

    **Acceptance Strategy:**
        Accepts based on target utility comparison with safety margin:

        - Tracks time per round to estimate remaining rounds
        - If remaining rounds <= 5, accepts to avoid negotiation failure
        - Otherwise accepts if opponent's offer exceeds target utility

    **Opponent Modeling:**
        Uses a library of multiple PlayerData models with different
        learning rates (delta = 1.0, 1.05, 0.55):

        - Each model tracks value frequencies with exponential weighting
        - Issue weights derived from maximum value frequencies
        - Randomly selects which model to use for each decision
        - Tracks Nash-optimal opponent (whose bids maximize product of
          estimated utilities) for weighting decisions

        Three weighting strategies randomly selected:
        1. Nash-based: weight by Nash optimality history
        2. Equal: all opponents weighted equally
        3. Alternating: alternate between opponents

    References:
        Kakimoto, S., & Fujita, K. (2017). RandomDance: Compromising Strategy
        Considering Interdependencies of Issues with Randomness. In Fujita, K.,
        et al. Modern Approaches to Agent-based Complex Automated Negotiation.
        Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.RandomDance.RandomDance"
        super().__init__(**kwargs)


class SENGOKU(GeniusNegotiator):
    """
    SENGOKU negotiation agent.

    **ANAC 2015**.

    Japanese-style modular agent with strategy/bidSearch/negotiatingInfo
    components. Features a "last action" phase for final concession attempts.

    **Offering Strategy:**
        - Modular design similar to AgentW/X family
        - Shift-based bid search for exploring outcome space
        - Has special "last action" phase near deadline for final
          concession to avoid negotiation failure
        - Tracks acceptance rate to adapt strategy

    **Acceptance Strategy:**
        - Time-dependent threshold with acceptance rate tracking
        - Special handling in "last action" phase
        - More aggressive acceptance near deadline

    **Opponent Modeling:**
        Tracks negotiation information including acceptance rates
        to adapt bidding and acceptance strategies dynamically.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.SENGOKU.SENGOKU"
        super().__init__(**kwargs)


class TUDMixedStrategyAgent(GeniusNegotiator):
    """
    TUDMixedStrategyAgent negotiation agent.

    **ANAC 2015**.

    From TU Delft. Uses mixed strategy with opponent utility function
    modeling. May re-offer previously received bids strategically.

    **Offering Strategy:**
        - Mixed strategy combining multiple approaches
        - Uses AgentUtils class for opponent utility estimation
        - May re-offer bids previously received from opponents
        - Strategy class determines next bid utility target

    **Acceptance Strategy:**
        - Strategy class determines acceptance conditions
        - Considers estimated opponent utility in decisions
        - Adaptive threshold based on negotiation progress

    **Opponent Modeling:**
        AgentUtils-based utility function modeling:

        - Estimates opponent's utility function from bid history
        - Uses model to evaluate bid quality for opponent
        - Informs both offering and acceptance decisions

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2015.TUDMixedStrategyAgent.TUDMixedStrategyAgent"
        super().__init__(**kwargs)


class XianFaAgent(GeniusNegotiator):
    """
    XianFaAgent negotiation agent.

    **ANAC 2015**.

    Uses tree-based statistical opponent modeling with custom Tree, Node,
    and Statistician classes in the xianfa package.

    **Offering Strategy:**
        - Tree-based bid organization for efficient search
        - Uses Statistician class to analyze opponent patterns
        - Adapts bids based on statistical analysis of opponent behavior

    **Acceptance Strategy:**
        - Statistical threshold based on opponent modeling
        - Uses tree structure to evaluate bid quality

    **Opponent Modeling:**
        Tree-based statistician model (xianfa package):

        - Tree and Node classes for hierarchical bid organization
        - Statistician class for opponent behavior analysis
        - Statistical inference on opponent preferences

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.xianfa.XianFaAgent"
        super().__init__(**kwargs)


class Y2015Group2(GeniusNegotiator):
    """
    Y2015Group2 negotiation agent.

    **ANAC 2015**.

    Alias for Group2. Uses Pareto-optimal bid search with per-party
    opponent modeling. See Group2 for full documentation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code and papers
        for authoritative information.

    References:
        Fujita, K., et al. (2017). The Sixth Automated Negotiating Agents Competition
        (ANAC 2015). Studies in Computational Intelligence, vol 674. Springer, Cham.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2015.group2.Group2"
        super().__init__(**kwargs)
