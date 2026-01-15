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
    """
    AgentGP (Gaussian Process Agent) negotiation agent.

    **ANAC 2019 competitor.**

    Modular agent with NegotiationInfo, BidSearch, and NegotiationStrategy
    components. Tracks bids accepted by multiple parties (PBList) and uses
    them in final phase. Custom utility space estimation using closest
    known bid approach.

    **Offering Strategy:**
        - Uses BidSearch with threshold from NegotiationStrategy
        - Final phase: offers bids from PBList (bids accepted by others)
        - Updates time scale to track negotiation progress

    **Acceptance Strategy:**
        - Based on NegotiationStrategy threshold
        - May end negotiation if conditions warrant

    **Opponent Modeling:**
        Tracks opponent bids and updates negotiation info per sender.
        Records bids supported by all other parties.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.agentgp.AgentGP"
        super().__init__(**kwargs)


class AgentLarry(GeniusNegotiator):
    """
    AgentLarry negotiation agent.

    **ANAC 2019 competitor.**

    Uses logistic regression for opponent modeling. Trains separate models
    per opponent on bid accept/reject history. Retrains models each round
    for better performance. Uses persistent data for history initialization.

    **Offering Strategy:**
        - Evaluates each bid by: own utility + product of opponent
          acceptance probabilities
        - Iterates through all bids to find best evaluation
        - Selects bid maximizing combined score

    **Acceptance Strategy:**
        - Accepts if opponent's bid rank exceeds next bid rank
          (accounting for discount factor)

    **Opponent Modeling:**
        Logistic regression per opponent, trained on:
        - Bids they offered (labeled as would-accept)
        - Bids they rejected (labeled as reject)

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.agentlarry.AgentLarry"
        super().__init__(**kwargs)


class DandikAgent(GeniusNegotiator):
    """
    DandikAgent negotiation agent.

    **ANAC 2019 competitor.**

    Uses OLS (Ordinary Least Squares) regression for utility estimation
    from bid rankings. One-hot encodes bids for regression. Handles large
    and small domain sizes differently.

    **Offering Strategy:**
        - Time-dependent threshold: stricter early, relaxes over time
        - t < 900: offers from top 0.5% of bids
        - t > 990: more concessions allowed
        - Final rounds: may offer best opponent bid

    **Acceptance Strategy:**
        - Accepts if bid utility > 93% of max (varies by time)
        - Near deadline: accepts based on opponent's max bid utility

    **Opponent Modeling:**
        Tracks opponent's best bid for fallback offers near deadline.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.dandikagent.dandikAgent"
        super().__init__(**kwargs)


class EAgent(GeniusNegotiator):
    """
    EAgent negotiation agent.

    **ANAC 2019 competitor.**

    Designed for preference uncertainty. Estimates utility space using
    bid ranking with score-based value estimation and standard deviation
    for issue weights.

    **Offering Strategy:**
        - Always offers max utility bid from estimated utility space

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility > 0.8

    **Preference Estimation:**
        - Values scored by position in bid ranking
        - Normalized by occurrence frequency
        - Issue weights based on inverse of value standard deviation

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.eagent.EAgent"
        super().__init__(**kwargs)


class FSEGA2019(GeniusNegotiator):
    """
    FSEGA2019 negotiation agent.

    **ANAC 2019 competitor.**

    Uses Bayesian opponent modeling with time-phased strategy (SMART,
    SERIAL, RESPONSIVE, RANDOM, TIT_FOR_TAT). Maintains sorted bid list
    for efficient bid generation.

    **Offering Strategy:**
        - Initial: offers max utility bid
        - Time phases: t<0.85 (case 0), t<0.95 (case 1), else (case 2)
        - Makes concessions in final phase (case 2)
        - Minimum utility threshold: 0.5

    **Acceptance Strategy:**
        - Accepts if opponent utility >= 1.03 * own last bid utility
        - Or if opponent utility > next planned bid utility

    **Opponent Modeling:**
        Bayesian opponent model (MyBayesianOpponentModel) updated with
        each opponent bid to estimate expected utility.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.fsega2019.agent.FSEGA2019"
        super().__init__(**kwargs)


class GaravelAgent(GeniusNegotiator):
    """
    GaravelAgent negotiation agent.

    **ANAC 2019 competitor.**

    Uses OLS regression for utility estimation and frequency-based
    opponent modeling. Generates optimal bids considering both own
    and opponent preferences.

    **Offering Strategy:**
        - Early: offers max utility bid
        - After round 200: selects from optimal bids (considering
          opponent model)
        - Final rounds: strategic bid selection

    **Acceptance Strategy:**
        - Accepts if estimated utility >= 0.92
        - Final round: accepts if utility >= 0.84

    **Opponent Modeling:**
        Frequency-based tracking of opponent value preferences.
        Updates issue and value weights from opponent bid history.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.garavelagent.GaravelAgent"
        super().__init__(**kwargs)


class Gravity(GeniusNegotiator):
    """
    Gravity negotiation agent.

    **ANAC 2019 competitor.**

    Uses Copeland matrices for preference learning from bid rankings.
    Pairwise comparison of bids fills frequency matrices used to estimate
    value and issue utilities.

    **Offering Strategy:**
        - First half of time: offers best bid
        - Second half: offers bids fixing opponent's most important
          issue at their preferred value
        - Occasionally sends best bid to confuse opponent

    **Acceptance Strategy:**
        - Accepts if: next bid utility <= received bid utility AND
          last offer utility <= received bid utility AND
          opponent utility decreased AND utility > reservation

    **Opponent Modeling:**
        Frequency arrays track opponent value preferences. Issue utilities
        estimated via sum of squared errors from frequency distribution.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.gravity.Gravity"
        super().__init__(**kwargs)


class Group1BOA(GeniusNegotiator):
    """
    Group1BOA (PodAgent) negotiation agent.

    **ANAC 2019 competitor.**

    BOA framework agent with custom components: Group1_AS (acceptance),
    Group1_BS (bidding), Group1_OM (opponent model). Uses exponential
    utility distribution for preference estimation.

    **Offering Strategy:**
        - Group1_BS component generates offers
        - Considers opponent model for bid selection

    **Acceptance Strategy:**
        - Group1_AS determines acceptance based on utility thresholds

    **Preference Estimation:**
        - Augments bid ranking with random bids placed by distance
        - Exponential utility function: low + (high-low) * (i/n)^2
        - Issue weights based on variance of value occurrences

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class HardDealer(GeniusNegotiator):
    """
    HardDealer negotiation agent.

    **ANAC 2019 competitor.**

    BOA framework agent with Boulware-like behavior. Uses linear
    programming (Simplex solver) for utility estimation and
    variance/spread-based issue weight estimation.

    **Offering Strategy:**
        - Time-dependent with concession parameter e = 1.8/deadline
        - Boulware-like: concedes slowly, more at end

    **Acceptance Strategy:**
        - HardDealer_AS component with hardheaded behavior

    **Preference Estimation:**
        - Linear programming minimizes slack variables for bid ranking
        - Issue weights from variance and spread of value positions
        - Blend variable adjusts ratio based on bid ranking size

    **Opponent Modeling:**
        HardDealer_OM with HardDealer_OMS strategy for utilizing
        opponent preference estimates.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.harddealer.HardDealer"
        super().__init__(**kwargs)


class KAgent(GeniusNegotiator):
    """
    KAgent negotiation agent.

    **ANAC 2019 competitor.**

    BOA framework agent with preference uncertainty handling. Uses
    random utility space estimation and time-dependent offering with
    Boulware-like concession (e=0.01).

    **Offering Strategy:**
        - TimeDependent_Offering with very low concession (e=0.01)
        - Extremely Boulware-like behavior

    **Acceptance Strategy:**
        - AC_Uncertain_Kindly: accepts with some flexibility

    **Opponent Modeling:**
        HardHeadedFrequencyModel with BestBid OM strategy.

    **Preference Estimation:**
        Randomized utility space (weights and values from random).

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.kagent.KAgent"
        super().__init__(**kwargs)


class KakeSoba(GeniusNegotiator):
    """
    KakeSoba negotiation agent.

    **ANAC 2019 competitor.**

    Uses Tabu Search for utility space estimation with Spearman rank
    correlation as fitness metric. Tracks bid error to minimize
    deviation from estimated preferences.

    **Offering Strategy:**
        - First round: offers max utility bid
        - Later: generates bid minimizing error with utility bounds
        - Lower bound: 0.85, upper bound: 1.0

    **Acceptance Strategy:**
        - Accepts if bid within acceptable utility bounds (>= 0.85)
        - Final round: accepts if better than reservation value

    **Preference Estimation:**
        Tabu Search with 5000 movements:
        - Random initial utility space
        - Neighbors generated by modifying weights/values
        - Best solution tracked via Spearman correlation score

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.kakesoba.KakeSoba"
        super().__init__(**kwargs)


class MINF(GeniusNegotiator):
    """
    MINF negotiation agent.

    **ANAC 2019 competitor.**

    BOA framework agent with Linear Programming for utility estimation.
    Uses HardHeadedFrequencyModel for opponent modeling and time-dependent
    offering with AC_Next acceptance.

    **Offering Strategy:**
        - TimeDependent_Offering with CT (concession threshold) = 0.998
        - Updates own bid info and opponent model info

    **Acceptance Strategy:**
        - AC_Next: accepts if opponent's next bid would be worse

    **Opponent Modeling:**
        HardHeadedFrequencyModel updated with opponent bids.
        BestBid OM strategy for utilizing opponent model.

    **Preference Estimation:**
        LP_Estimation using Linear Programming on bid ranking.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.minf.MINF"
        super().__init__(**kwargs)


class PodAgent(GeniusNegotiator):
    """
    PodAgent negotiation agent (alias for Group1BOA).

    **ANAC 2019 competitor.**

    See :class:`Group1BOA` for full documentation.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.podagent.Group1_BOA"
        super().__init__(**kwargs)


class SACRA(GeniusNegotiator):
    """
    SACRA (Simulated Annealing-based Concession Rate controlling Agent).

    **ANAC 2019 competitor.**

    Uses Simulated Annealing for utility space estimation with Spearman
    rank correlation as fitness metric. Concession rate controlled by
    opponent behavior.

    **Offering Strategy:**
        - Generates 20000 candidate bids sorted by utility
        - Concession rate based on: (lastUtil - firstUtil) / maxUtil * 0.7
        - Target utility: maxUtil - concessionRate
        - Selects random bid above target utility

    **Acceptance Strategy:**
        - Probabilistic: acceptProb = (receivedUtil - target) / (max - target)
        - Accepts with probability proportional to how much bid exceeds target

    **Preference Estimation:**
        Simulated Annealing (10000 iterations):
        - Energy = negative Spearman correlation score
        - Temperature decay: alpha^(iteration/total)
        - Neighbor generation: modify one weight or value

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

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
    """
    SolverAgent negotiation agent.

    **ANAC 2019 competitor.**

    Sophisticated preference estimation using pairwise comparison from
    bid rankings. Uses Copeland-style analysis and OLS regression. Nash
    bargaining solution search near deadline.

    **Offering Strategy:**
        - Phase 1 (t < 30%): random bids above threshold (0.55-0.75)
        - Phase 2 (30-97%): cycles through high-utility bids
        - Phase 3 (>97%): offers Nash bid or max opponent bid

    **Acceptance Strategy:**
        - Accepts if own planned bid utility <= received bid utility
        - Or if received bid above phase threshold

    **Opponent Modeling:**
        OLS regression on opponent bid history to estimate opponent
        utilities. Nash product maximization for win-win bids.

    **Preference Estimation:**
        Complex pairwise analysis of bid rankings with issue and
        value ordering inference.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.solveragent.SolverAgent"
        super().__init__(**kwargs)


class TheNewDeal(GeniusNegotiator):
    """
    TheNewDeal negotiation agent.

    **ANAC 2019 competitor.**

    Estimates utilities using equal-difference distribution from bid
    ranking. Calculates value weights using linear equation solving
    and issue weights from consecutive value matches.

    **Offering Strategy:**
        - Offers bids in decreasing utility order
        - Ratio-based timing: offers at intervals proportional to
          domain size / timeline
        - Cycles back to start after reaching midpoint

    **Acceptance Strategy:**
        - Accepts if opponent bid utility >= own next bid utility
        - Final round: always accepts (if not first agent)

    **Preference Estimation:**
        - Equal-difference utility distribution across bid ranking
        - Value weights via linear equation solving per value
        - Issue weights from consecutive match frequency

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.thenewdeal.TheNewDeal"
        super().__init__(**kwargs)


class WinkyAgent(GeniusNegotiator):
    """
    WinkyAgent negotiation agent.

    **ANAC 2019 competitor.**

    Uses Batch Gradient Descent (BGD) for utility estimation from bid
    rankings. One-hot encodes bids and trains value parameters. Time-phased
    strategy with opponent bid tracking.

    **Offering Strategy:**
        - t < 70%: random bids above threshold (0.7-0.82 based on max)
        - t < 98%: similar with slightly lower threshold
        - t < 99%: offers from sorted received bids (top 3%)
        - Final: accepts or offers best received bid

    **Acceptance Strategy:**
        - Time-dependent thresholds relative to highest received utility
        - t < 99%: accepts if utility > highestReceived - 0.03
        - Later: progressively relaxes threshold

    **Preference Estimation:**
        BGD training:
        - One-hot encoded bid features
        - Learning rate = initial value utility / 10
        - Iterations = ranklistSize * valueSum

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Aydogan, R. et al. (2020). Challenges and Main Results of ANAC 2019.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2019.winkyagent.winkyAgent"
        super().__init__(**kwargs)
