"""Genius negotiator implementations - ANAC 2017 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "AgentF",
    "AgentKN",
    "CaduceusDC16",
    "Farma17",
    "Farma2017",
    "GeneKing",
    "Gin",
    "Group3",
    "Imitator",
    "MadAgent",
    "Mamenchis",
    "Mosa",
    "ParsAgent3",
    "PonPokoAgent",
    "Rubick",
    "ShahAgent",
    "SimpleAgent2017",
    "TaxiBox",
    "TucAgent",
]


class AgentF(GeniusNegotiator):
    """
    AgentF negotiation agent.

    **ANAC 2017**.

    AgentF uses a modular architecture with negotiationInfo, bidSearch, and
    negotiationStrategy components. It leverages persistent data from past
    negotiations to build a "PBList" (popular bids from previous agreements).

    **Offering Strategy:**
        - Uses time-scaled threshold via negotiationStrategy.getThreshold(time)
        - Searches for bids above the threshold, preferring those in PBList
        - In final phase, traverses a "CList" of candidate bids with concession
        - Tracks supporter count in multilateral settings

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility exceeds current threshold
        - More aggressive acceptance in final negotiation phase

    **Opponent Modeling:**
        - Builds PBList from past agreements across negotiations
        - Uses persistent data to remember successful bids

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.agentf.AgentF"
        super().__init__(**kwargs)


class AgentKN(GeniusNegotiator):
    """
    AgentKN negotiation agent.

    **ANAC 2017 Nash Product Category Finalist**.

    AgentKN by Keita Nakamura uses Simulated Annealing for bid search and
    a sophisticated opponent modeling approach to estimate the maximum
    utility the opponent might offer. It balances self-utility maximization
    with opponent value frequency analysis.

    **Offering Strategy:**
        Uses Simulated Annealing to search for 10 bids that maximize utility
        while starting from a random initial bid. The bids are then scored
        using a combined metric:

        score = utility + 0.1^(log10(frequency)+1) * frequency * utility

        where frequency is the sum of opponent-offered value frequencies.
        This encourages bids that are both high-utility and contain values
        the opponent has frequently requested.

    **Acceptance Strategy:**
        Accepts when the opponent's bid exceeds a dynamic threshold:

        threshold(t) = 1 - (1 - e_max(t)) * t^alpha

        where alpha > 1 controls concession rate and e_max(t) is the
        estimated maximum utility the opponent might offer, calculated as:

        e_max(t) = mu(t) + (1 - mu(t)) * d(t)
        d(t) = sqrt(3) * sigma(t) / sqrt(mu(t) * (1 - mu(t)))

        where mu(t) is the mean and sigma(t) is the standard deviation of
        utilities from opponent offers.

    **Opponent Modeling:**
        Tracks value frequencies for each issue across opponent bids:

        - Updates issue weights when consecutive bids have the same value
        - Maintains normalized value frequency counts per issue
        - Uses statistical analysis (mean, std) of opponent bid utilities
          to estimate the opponent's bidding range

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
        kwargs["java_class_name"] = "agents.anac.y2017.agentkn.AgentKN"
        super().__init__(**kwargs)


class CaduceusDC16(GeniusNegotiator):
    """
    CaduceusDC16 negotiation agent.

    **ANAC 2017**.

    CaduceusDC16 is a meta-agent that combines 5 sub-agents (YXAgent, ParsCat,
    Farma, MyAgent, Atlas32016) using a weighted voting mechanism. Each
    sub-agent contributes to bid selection and acceptance decisions.

    **Offering Strategy:**
        - Each sub-agent proposes a bid and votes on others' proposals
        - Weighted scoring: 5 (YXAgent), 4 (ParsCat), 3 (Farma), 2 (MyAgent), 1 (Atlas32016)
        - Uses getMostProposedBidWithWeight() for weighted value voting per issue
        - Offers best bid early (before 83% * discount factor of time)

    **Acceptance Strategy:**
        - Voting mechanism: accepts if accept_score > bid_score
        - Each sub-agent votes to accept or reject based on its own criteria
        - Weighted votes determine final decision

    **Opponent Modeling:**
        Inherits opponent modeling from each sub-agent, which are combined
        through the voting mechanism.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.caduceusdc16.CaduceusDC16"
        super().__init__(**kwargs)


class Farma17(GeniusNegotiator):
    """
    Farma17 negotiation agent.

    **ANAC 2017**.

    Farma17 uses a modular design with NegoStats, BidSearch, NegoHistory, and
    NegoStrategy components. It tracks rejected and agreed values per opponent
    to inform its bidding strategy.

    **Offering Strategy:**
        - Time-dependent threshold controls minimum acceptable utility
        - BidSearch component finds bids above threshold
        - Considers opponent preferences when selecting among valid bids

    **Acceptance Strategy:**
        - Accepts bids above the time-dependent threshold
        - Updates opponent info when receiving offers or accepts

    **Opponent Modeling:**
        - Tracks rejected and agreed values for each opponent
        - NegoStats maintains statistics on opponent behavior
        - NegoHistory stores past negotiation data for analysis

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class Farma2017(GeniusNegotiator):
    """
    Farma2017 negotiation agent.

    **ANAC 2017**.

    Alias for Farma17. Uses the same Java class (agents.anac.y2017.farma.Farma17).
    See :class:`Farma17` for full documentation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class GeneKing(GeniusNegotiator):
    """
    GeneKing negotiation agent.

    **ANAC 2017**.

    GeneKing uses a genetic algorithm approach to bid generation. It maintains
    a gene pool of bids and evolves them using crossover and mutation operators
    informed by opponent preferences.

    **Offering Strategy:**
        - Maintains a gene pool of candidate bids
        - Crossover with frequency weighting based on opponent value preferences
        - Mutation with 1/70 probability per issue
        - Bid evaluation: util*utilWeight + sim*simWeight + exUtil1 + exUtil2 - diff
        - Uses negotiation history for initial population

    **Acceptance Strategy:**
        - Accepts bids that meet utility threshold criteria
        - Threshold adapts based on time and opponent behavior

    **Opponent Modeling:**
        - Tracks frequency per issue/value for each opponent
        - Uses frequency data to weight crossover operations
        - Higher frequency values more likely selected during evolution

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.geneking.GeneKing"
        super().__init__(**kwargs)


class Gin(GeniusNegotiator):
    """
    Gin negotiation agent.

    **ANAC 2017**.

    Gin builds a frequency table per issue across all opponents and uses
    negotiation history to adapt its concession behavior based on past
    success/failure rates.

    **Offering Strategy:**
        - Builds frequency table of opponent-preferred values per issue
        - Selects bids sorted by alignment with frequency table
        - History-aware: adjusts fitToOpponent (0/1/2) based on failure rate
        - More aggressive if failure rate > 40%, moderate if > 20%

    **Acceptance Strategy:**
        - Three-tier threshold system: 0.9 / 0.88 / 0.80
        - Thresholds adjust more aggressively if history shows failures
        - Accepts when opponent bid exceeds current threshold

    **Opponent Modeling:**
        - Aggregates value frequencies across all opponents
        - Uses frequency data to select mutually beneficial bids

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.gin.Gin"
        super().__init__(**kwargs)


class Group3(GeniusNegotiator):
    """
    Group3 negotiation agent.

    **ANAC 2017**.

    Group3 tracks assumed values per opponent and uses history-based minimum
    utility adjustments. It merges opponent preferences to generate bids that
    are likely acceptable to multiple parties.

    **Offering Strategy:**
        - Tracks assumed values per opponent (increments when same value repeated)
        - Merges opponent preferences to generate consensus bids
        - History-based minimum utility adjustment

    **Acceptance Strategy:**
        - Time-based acceptance thresholds:
          - 0.85 when time < 0.5
          - 0.75 when time < 0.8
          - 0.70 when time < 0.9
          - Accepts any offer when time > 0.95

    **Opponent Modeling:**
        - Increments value weights when opponents repeat the same value
        - Merges preferences across opponents to find common ground

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.group3.Group3"
        super().__init__(**kwargs)


class Imitator(GeniusNegotiator):
    """
    Imitator negotiation agent.

    **ANAC 2017**.

    Imitator uses frequency-based opponent modeling to generate bids that
    match the most frequent values offered by opponents. It aims to find
    agreements by imitating opponent preferences.

    **Offering Strategy:**
        - Builds frequency table per issue/value from opponent offers
        - Generates bids using the most frequent values from each opponent
        - Compares frequencies across opponents to select optimal values

    **Acceptance Strategy:**
        - Time-based acceptance with increasing acceptance rate
        - More likely to accept as deadline approaches

    **Opponent Modeling:**
        - Tracks value frequencies for each issue per opponent
        - Compares frequencies across opponents to identify common preferences

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.limitator.Imitator"
        super().__init__(**kwargs)


class MadAgent(GeniusNegotiator):
    """
    MadAgent negotiation agent.

    **ANAC 2017**.

    MadAgent uses risk-based fake bid generation and multiple opponent models.
    It employs different "madness" phases as the deadline approaches.

    **Offering Strategy:**
        - Generates fake bids periodically (every 100000/2^5 rounds) for risk assessment
        - Maintains preferred bids list from opponent modeling
        - Offers second-best bid in first 5% of negotiation
        - Three opponent models: two individual + one combined

    **Acceptance Strategy:**
        - Threshold starts at 0.8 * 1.125 = 0.9
        - Adapts threshold based on opponent model predictions
        - Time phases: "almost mad" at 50%, "mad" at 80%

    **Opponent Modeling:**
        - Maintains three separate opponent models
        - Tracks preferred bids per opponent
        - Combines models for final decision making

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.madagent.MadAgent"
        super().__init__(**kwargs)


class Mamenchis(GeniusNegotiator):
    """
    Mamenchis negotiation agent.

    **ANAC 2017**.

    Mamenchis uses complex time-phase bidding with social welfare optimization.
    It aims to maximize both sum and product of utilities across parties.

    **Offering Strategy:**
        - Time phases: time1=0.9, time2=0.99, time3=0.995
        - History-aware parameter initialization
        - Concession formula: upper - (upper - lower) * t^exponent
        - Generates candidate bids by merging top bids from parties

    **Acceptance Strategy:**
        - Phase-dependent acceptance thresholds
        - Social welfare considerations (max sum and product of utilities)
        - More aggressive near deadline

    **Opponent Modeling:**
        - Preference modeling per opponent
        - Estimates opponent utility for social welfare calculation
        - Merges top bids to find mutually beneficial outcomes

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.mamenchis.Mamenchis"
        super().__init__(**kwargs)


class Mosa(GeniusNegotiator):
    """
    Mosa negotiation agent.

    **ANAC 2017**.

    Mosa is very similar to Mamenchis (same team) with slightly different
    parameters. Uses time-phase bidding with social welfare optimization.

    **Offering Strategy:**
        - Time phases: 0.9, 0.99, 0.996
        - Initial upper=0.95, exponent=50
        - Same social welfare approach as Mamenchis
        - Slightly different tuning parameters

    **Acceptance Strategy:**
        - Phase-dependent acceptance with social welfare consideration
        - Similar to Mamenchis with parameter variations

    **Opponent Modeling:**
        - Same preference modeling approach as Mamenchis
        - Estimates opponent utilities for welfare calculations

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.mosateam.Mosa"
        super().__init__(**kwargs)


class ParsAgent3(GeniusNegotiator):
    """
    ParsAgent3 negotiation agent.

    **ANAC 2017**.

    Alias for ShahAgent. Uses the same Java class (agents.anac.y2017.parsagent3.ShahAgent).
    See :class:`ShahAgent` for full documentation.

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.parsagent3.ShahAgent"
        super().__init__(**kwargs)


class PonPokoAgent(GeniusNegotiator):
    """
    PonPokoAgent negotiation agent.

    **ANAC 2017 Individual Utility Category Winner**.

    PonPokoAgent by Takaki Matsune employs a randomized multi-strategy approach
    that makes it difficult for opponents to predict its behavior. At the start
    of each negotiation session, it randomly selects one of 5 different bidding
    patterns, each with distinct concession characteristics.

    **Offering Strategy:**
        The agent randomly selects one of 5 bidding patterns at initialization:

        - **Pattern 0**: Sinusoidal oscillation with slow linear decline.
          High/Low thresholds follow sin(40t) pattern.
        - **Pattern 1**: Linear concession from 1.0, slow decline to 0.78.
        - **Pattern 2**: Sinusoidal with larger amplitude (sin(20t)).
        - **Pattern 3**: Very conservative, minimal concession (0.95-1.0)
          until deadline when it drops to 0.7.
        - **Pattern 4**: Sinusoidal pattern tied to time (sin(20t) * t).

        Bids are selected from the pre-sorted bid space within the
        [threshold_low, threshold_high] utility range.

    **Acceptance Strategy:**
        Accepts if the opponent's offer has utility greater than the current
        threshold_low value. This creates a simple but effective acceptance
        criterion that varies with the selected bidding pattern.

    **Opponent Modeling:**
        PonPokoAgent does not employ explicit opponent modeling. Its strength
        lies in the unpredictability of its randomly selected strategy, making
        it resistant to exploitation by adaptive opponents.

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
        kwargs["java_class_name"] = "agents.anac.y2017.ponpokoagent.PonPokoAgent"
        super().__init__(**kwargs)


class Rubick(GeniusNegotiator):
    """
    Rubick negotiation agent.

    **ANAC 2017 Individual Utility Category Finalist**.

    Rubick Agent by Okan Tunali is a complex time-based conceder enriched
    with frequency-based opponent modeling. It maintains the highest utility
    ever received as a lower bound and uses randomized Boulware-style
    concession with increasing variance over time.

    **Offering Strategy:**
        Generates bids above a target utility using opponent model insights:

        - Searches for bids that maximize intersection with opponent's
          most frequently offered values (frequency-based search)
        - If no suitable bid found via opponent model, falls back to
          nearest bid to target utility
        - Near deadline (t > 0.995), offers from a cached list of
          previously accepted bids

        Target utility follows randomized Boulware with power parameter
        randomly selected based on max received utility.

    **Acceptance Strategy:**
        Time-based with randomness to prevent exploitation:

        target = 1 - t^power * |N(0, 1/3)|

        where power is randomly 2, 3, or 10 based on opponent behavior.
        The target is bounded by the maximum received utility.
        Accepts if opponent offer exceeds target or time > 0.999.

    **Opponent Modeling:**
        Employs frequency-based opponent modeling:

        - Tracks value frequencies for each issue per opponent
        - Extracts "bags" of preferred values (above median frequency)
        - Scores bids by counting intersections with opponent preferences
        - Maintains separate models for multilateral scenarios

        Also keeps a sorted list of bids that were accepted by opponents
        in previous negotiations for use near the deadline.

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
        kwargs["java_class_name"] = "agents.anac.y2017.rubick.Rubick"
        super().__init__(**kwargs)


class ShahAgent(GeniusNegotiator):
    """
    ShahAgent negotiation agent.

    **ANAC 2017**.

    ShahAgent uses K-means clustering (20 clusters) to analyze opponent bids
    and estimate concession rates. It adapts its bidding based on cluster
    analysis and negotiation history.

    **Offering Strategy:**
        - Time-dependent utility: (pmin + (1-pmin) * (1-f(t))) * discount^t
        - Clusters opponent bids using K-means (20 clusters)
        - Tracks cluster history with distribution factor and max distance
        - History-aware minimum limit adjustment

    **Acceptance Strategy:**
        - Accepts bids above dynamic threshold based on cluster analysis
        - Concession rate estimated from opponent cluster transitions

    **Opponent Modeling:**
        - K-means clustering of opponent bid patterns
        - Tracks cluster distribution and transitions over time
        - Estimates opponent concession rate from cluster analysis

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.parsagent3.ShahAgent"
        super().__init__(**kwargs)


class SimpleAgent2017(GeniusNegotiator):
    """
    SimpleAgent2017 negotiation agent.

    **ANAC 2017**.

    SimpleAgent2017 uses a Predictor class for bid generation and has
    history-aware threshold adjustment. It delegates most logic to the
    Predictor component.

    **Offering Strategy:**
        - Delegates to Predictor class for bid generation
        - History-aware threshold via setHistoryAndUpdateThreshold()

    **Acceptance Strategy:**
        - Threshold-based acceptance adjusted by negotiation history
        - Simple decision logic delegating to Predictor

    **Opponent Modeling:**
        - Handled by Predictor component
        - Uses history data for prediction

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.simpleagent.SimpleAgent"
        super().__init__(**kwargs)


class TaxiBox(GeniusNegotiator):
    """
    TaxiBox negotiation agent.

    **ANAC 2017**.

    TaxiBox uses adaptive target utility with emax estimation based on
    opponent behavior tracking. It adjusts its target dynamically based on
    the difference between expected and actual opponent utilities.

    **Offering Strategy:**
        - Estimates emax: avg + (1-avg) * |diff|
        - Target formula: 1 - (1-emax) * (0.8t)^(3 + 2*shiwang - 0.8t)
        - Tracks opponent utility average and changes
        - Adaptive concession based on opponent responsiveness

    **Acceptance Strategy:**
        - Accepts bids above current target utility
        - Target adapts based on opponent utility patterns

    **Opponent Modeling:**
        - Tracks average utility of opponent offers
        - Monitors utility changes to estimate opponent concession

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.tangxun.taxibox"
        super().__init__(**kwargs)


class TucAgent(GeniusNegotiator):
    """
    TucAgent negotiation agent.

    **ANAC 2017**.

    TucAgent uses Bayesian opponent modeling with separate models for each
    opponent in multilateral settings. It employs an IssueManager for
    threshold calculation and bid generation.

    **Offering Strategy:**
        - IssueManager handles bid generation and threshold calculation
        - Uses Bayesian models to predict opponent preferences
        - Selects bids based on combined opponent model predictions

    **Acceptance Strategy:**
        - Immediately accepts bids with utility > 0.95
        - Otherwise uses dynamic threshold from IssueManager
        - Bayesian models inform acceptance decisions

    **Opponent Modeling:**
        - Bayesian opponent model with separate models per opponent
        - Updates beliefs based on observed opponent offers
        - IssueManager integrates opponent models for decisions

    Note:
        This description is AI-generated based on the original Java implementation
        and may not be fully accurate. Refer to the original source code for
        authoritative information.

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
        kwargs["java_class_name"] = "agents.anac.y2017.tucagent.TucAgent"
        super().__init__(**kwargs)
