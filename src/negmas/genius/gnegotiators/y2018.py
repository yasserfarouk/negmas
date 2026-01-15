"""Genius negotiator implementations - ANAC 2018 agents."""

from __future__ import annotations

from ..negotiator import GeniusNegotiator

__all__ = [
    "Agent33",
    "Agent36",
    "AgentHerb",
    "AgentNP1",
    "AgreeableAgent2018",
    "AteamAgent",
    "Ateamagent",
    "BetaOne",
    "Betaone",
    "ConDAgent",
    "ExpRubick",
    "FullAgent",
    "GroupY",
    "IQSun2018",
    "Lancelot",
    "Libra",
    "MengWan",
    "PonPokoRampage",
    "SMACAgent",
    "Seto",
    "Shiboy",
    "Sontag",
    "Yeela",
]


class Agent33(GeniusNegotiator):
    """
    Agent33 negotiation agent.

    **ANAC 2018 competitor.**

    Modular agent with separate components for statistics, bid search, history
    tracking, and strategy. Tracks rejected and agreed values per opponent.

    **Offering Strategy:**
        - Uses time-dependent threshold from NegoStrategy component
        - BidSearch generates bids above threshold considering opponent preferences

    **Acceptance Strategy:**
        - Accepts if opponent's bid utility exceeds current threshold
        - Threshold decreases over time

    **Opponent Modeling:**
        Tracks frequency of values rejected/agreed by each opponent via NegoStats.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agent33.Agent33"
        super().__init__(**kwargs)


class Agent36(GeniusNegotiator):
    """
    Agent36 (MengWan) negotiation agent.

    **ANAC 2018 competitor.**

    Tracks opponent preferences via frequency maps and finds mutual issues
    between opponents. Stores commonly accepted bids across sessions.

    **Offering Strategy:**
        - Generates bids above time-dependent threshold
        - Threshold: 0.8 initially, 0.75 at t>0.95, 0.7 at t>0.98

    **Acceptance Strategy:**
        - Accepts if utility exceeds current threshold
        - More permissive in final negotiation stages

    **Opponent Modeling:**
        Frequency-based tracking of opponent value preferences to identify
        mutually beneficial issues.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class AgentHerb(GeniusNegotiator):
    """
    AgentHerb negotiation agent.

    **ANAC 2018 competitor.**

    Uses logistic regression for opponent modeling. Trains separate models per
    opponent on bid accept/reject history. Uses persistent data for initial
    training in repeated negotiations.

    **Offering Strategy:**
        - Evaluates bids by: own utility + product of opponent acceptance
          probabilities
        - Selects bids maximizing combined score

    **Acceptance Strategy:**
        - Accepts if bid utility is acceptable given opponent model predictions

    **Opponent Modeling:**
        Logistic regression trained on opponent accept/reject decisions for
        each bid offered.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agentherb.AgentHerb"
        super().__init__(**kwargs)


class AgentNP1(GeniusNegotiator):
    """
    AgentNP1 negotiation agent.

    **ANAC 2018 competitor.**

    Frequency-based opponent modeling with issue weight estimation. Tracks
    the "hardest opponent" based on value frequency differences.

    **Offering Strategy:**
        - Generates bids above time-phased thresholds
        - Threshold: 0.81 at t<0.9, 0.78 at t<0.99

    **Acceptance Strategy:**
        - Accepts if utility exceeds current phase threshold

    **Opponent Modeling:**
        Estimates opponent issue weights via value frequency analysis.
        Identifies hardest opponent for strategic adaptation.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.agentnp1.AgentNP1"
        super().__init__(**kwargs)


class AgreeableAgent2018(GeniusNegotiator):
    """
    AgreeableAgent2018 negotiation agent.

    **ANAC 2018 competitor.**

    Frequency-based opponent modeling with roulette wheel bid selection.
    Domain-size aware timing for concession adjustments.

    **Offering Strategy:**
        - Time-dependent concession: pMin + (pMax-pMin)*(1-f(t))
        - Roulette wheel selection among bids in utility range
        - Selection weighted by opponent model scores

    **Acceptance Strategy:**
        - Accepts bids within acceptable utility range based on opponent model

    **Opponent Modeling:**
        Frequency-based analysis of opponent bid history to estimate
        preferences.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs[
            "java_class_name"
        ] = "agents.anac.y2018.agreeableagent2018.AgreeableAgent2018"
        super().__init__(**kwargs)


class AteamAgent(GeniusNegotiator):
    """
    AteamAgent negotiation agent.

    **ANAC 2018 competitor.**

    Generates all possible bids sorted by utility. Estimates opponent utility
    via bid history frequency analysis. Looks for common issues with opponent.

    **Offering Strategy:**
        - First 50% of time: offers maximum utility bid
        - Later: searches for bids above threshold (0.8 * max utility)
        - Considers estimated opponent utility for bid selection

    **Acceptance Strategy:**
        - Accepts if utility exceeds threshold based on negotiation progress

    **Opponent Modeling:**
        Frequency analysis of opponent bid history to estimate their utility
        function and find common ground.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)


class Ateamagent(GeniusNegotiator):
    """
    Ateamagent negotiation agent (alias for AteamAgent).

    **ANAC 2018 competitor.**

    See :class:`AteamAgent` for full documentation.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ateamagent.ATeamAgent"
        super().__init__(**kwargs)


class BetaOne(GeniusNegotiator):
    """
    BetaOne negotiation agent.

    **ANAC 2018 competitor.**

    Uses linear regression to detect opponent "betrayal" via slope analysis.
    Features AntiAnalysis component for threshold boxes and history-aware
    selfish ratio adjustment.

    **Offering Strategy:**
        - Generates bids based on threshold adjusted by opponent behavior
        - AntiAnalysis component provides threshold boundaries

    **Acceptance Strategy:**
        - Accepts based on utility threshold adjusted by betrayal detection

    **Opponent Modeling:**
        Linear regression on opponent concession patterns to detect
        cooperative vs. competitive behavior (betrayal detection).

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class Betaone(GeniusNegotiator):
    """
    Betaone negotiation agent (alias for BetaOne).

    **ANAC 2018 competitor.**

    See :class:`BetaOne` for full documentation.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.beta_one.Group2"
        super().__init__(**kwargs)


class ConDAgent(GeniusNegotiator):
    """
    ConDAgent negotiation agent.

    **ANAC 2018 competitor.**

    Uses Bayesian opponent modeling for each opponent. IssueManager handles
    threshold calculation and bid generation.

    **Offering Strategy:**
        - Generates bids via IssueManager above dynamic threshold
        - Considers Bayesian model predictions for bid selection

    **Acceptance Strategy:**
        - Accepts immediately if utility > 0.95
        - Otherwise uses dynamic threshold from IssueManager

    **Opponent Modeling:**
        Bayesian model per opponent updated with observed bids to estimate
        opponent preferences probabilistically.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.condagent.ConDAgent"
        super().__init__(**kwargs)


class ExpRubick(GeniusNegotiator):
    """
    ExpRubick negotiation agent.

    **ANAC 2018 competitor.**

    Adaptive target utility with emax estimation. Uses frequency-based
    opponent modeling with "bags" of preferred values and history analysis.

    **Offering Strategy:**
        - Target utility: 1 - (1-emax) * (0.8t)^(3 + 2*concession - 0.8t)
        - Generates bids near target considering opponent preferences

    **Acceptance Strategy:**
        - Accepts if utility exceeds adaptive target

    **Opponent Modeling:**
        Frequency-based with "bags" collecting preferred values per issue
        for each opponent.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.exp_rubick.Exp_Rubick"
        super().__init__(**kwargs)


class FullAgent(GeniusNegotiator):
    """
    FullAgent negotiation agent.

    **ANAC 2018 competitor.**

    BOA framework agent with decoupled components: OpponentModel_lgsmi,
    OMStrategy_lgsmi, OfferingStrategy_lgsmi, AcceptanceStrategy_lgsmi.
    BidsManager tracks opponent bids and acceptances.

    **Offering Strategy:**
        - OfferingStrategy_lgsmi component generates offers
        - Uses opponent model for bid selection optimization

    **Acceptance Strategy:**
        - AcceptanceStrategy_lgsmi determines acceptance
        - Considers opponent model predictions

    **Opponent Modeling:**
        OpponentModel_lgsmi with OMStrategy_lgsmi for strategic use of
        opponent preference estimates.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.fullagent.FullAgent"
        super().__init__(**kwargs)


class GroupY(GeniusNegotiator):
    """
    GroupY negotiation agent.

    **ANAC 2018 competitor.**

    Opponent model per agent tracking value frequencies. History-aware
    initialization from previous sessions.

    **Offering Strategy:**
        - First 3 rounds: offers best bid
        - Later: uses opponent model scoring for bid selection
        - Time-based phases at 0.5 and 0.2 remaining time

    **Acceptance Strategy:**
        - Time-based utility thresholds with phase transitions

    **Opponent Modeling:**
        Tracks value frequency per opponent to estimate preferences.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.groupy.GroupY"
        super().__init__(**kwargs)


class IQSun2018(GeniusNegotiator):
    """
    IQSun2018 negotiation agent.

    **ANAC 2018 competitor.**

    Weighted average of multiple factors for utility calculation. Domain-size
    dependent weights with sinusoidal threshold variations.

    **Offering Strategy:**
        - Combines: time-based utility, session history average, and
          "helping bid" with sinusoidal threshold
        - Concession factor: 0.1-0.3, minimum utility: 0.5
        - Weights adjusted by domain size

    **Acceptance Strategy:**
        - Accepts based on weighted combination of factors

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.iqson.IQSun2018"
        super().__init__(**kwargs)


class Lancelot(GeniusNegotiator):
    """
    Lancelot negotiation agent.

    **ANAC 2018 competitor.**

    Modular design with separate strategy and bidSearch components.
    Time-phased behavior with opponent evaluation influencing decisions.

    **Offering Strategy:**
        - t < 0.2: random bids above threshold
        - 0.2 <= t < 0.98: positive bids considering opponent evaluation
        - t >= 0.98: threshold-based offers

    **Acceptance Strategy:**
        - Opponent evaluation influences acceptance threshold
        - More permissive near deadline

    **Opponent Modeling:**
        Evaluates opponent behavior to adjust strategy parameters.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.lancelot.Lancelot"
        super().__init__(**kwargs)


class Libra(GeniusNegotiator):
    """
    Libra negotiation agent.

    **ANAC 2018 competitor.**

    Meta-agent with 19 sub-agents including Boulware, Conceder, ParsAgent,
    Atlas3, YXAgent, Farma, PonPokoAgent, Rubick, etc. Uses weighted voting
    on offer/accept/end decisions.

    **Offering Strategy:**
        - Weighted voting among sub-agents' proposed offers
        - Weights adjusted based on opponent responses to each sub-agent

    **Acceptance Strategy:**
        - Weighted voting among sub-agents' accept/reject decisions

    **Opponent Modeling:**
        Indirect through sub-agents. Weights reflect which sub-agent
        strategies are most effective against current opponent.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.libra.Libra"
        super().__init__(**kwargs)


class MengWan(GeniusNegotiator):
    """
    MengWan negotiation agent (alias for Agent36).

    **ANAC 2018 competitor.**

    See :class:`Agent36` for full documentation.

    Note:
        AI-generated summary. May not be fully accurate.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.meng_wan.Agent36"
        super().__init__(**kwargs)


class PonPokoRampage(GeniusNegotiator):
    """
    PonPokoRampage negotiation agent.

    **ANAC 2018 competitor.**

    Extension of PonPokoAgent with 5 random threshold patterns. Detects
    "hardliner" opponents and adjusts strategy accordingly.

    **Offering Strategy:**
        - Sinusoidal threshold variations with random pattern selection
        - Adjusts threshold +0.05 if hardliner opponent detected

    **Acceptance Strategy:**
        - Uses threshold modified by opponent hardliner detection

    **Opponent Modeling:**
        Detects hardliner opponents by counting unique bids offered.
        Few unique bids indicates a hardliner.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.ponpokorampage.PonPokoRampage"
        super().__init__(**kwargs)


class SMACAgent(GeniusNegotiator):
    """
    SMACAgent (Sophisticated Machine-learning Agent for Negotiations).

    **ANAC 2018 competitor.**

    Pre-optimized parameter arrays (15 configurations) selected by domain
    size, reservation value, and discount factor. Uses sigmoid utility curves
    and frequency-based opponent modeling.

    **Offering Strategy:**
        - Sigmoid utility curves parameterized by pre-optimized config
        - Parameters selected based on domain characteristics

    **Acceptance Strategy:**
        - Uses selected parameter configuration for threshold

    **Opponent Modeling:**
        Frequency-based with Chebyshev/Euclidean distance metrics for
        similarity computation.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.smac_agent.SMAC_Agent"
        super().__init__(**kwargs)


class Seto(GeniusNegotiator):
    """
    Seto negotiation agent.

    **ANAC 2018 competitor.**

    Similar architecture to Agent33/Shiboy with NegoStats, BidSearch,
    NegoHistory, and NegoStrategy components. Alpha parameter adjusted
    based on first received bid.

    **Offering Strategy:**
        - Uses NegoStrategy with alpha = 4*sqrt(util) + 0.5
        - Alpha computed from first received bid utility

    **Acceptance Strategy:**
        - Time-dependent threshold from NegoStrategy

    **Opponent Modeling:**
        Tracks rejected/agreed values via NegoStats component.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.seto.Seto"
        super().__init__(**kwargs)


class Shiboy(GeniusNegotiator):
    """
    Shiboy negotiation agent.

    **ANAC 2018 competitor.**

    Modular design identical to Agent33/Seto with NegoStats, BidSearch,
    NegoHistory, and NegoStrategy components.

    **Offering Strategy:**
        - BidSearch generates bids above time-dependent threshold
        - Considers opponent preferences from NegoStats

    **Acceptance Strategy:**
        - Time-dependent threshold from NegoStrategy

    **Opponent Modeling:**
        Tracks rejected and agreed values per opponent via NegoStats.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.shiboy.Shiboy"
        super().__init__(**kwargs)


class Sontag(GeniusNegotiator):
    """
    Sontag negotiation agent.

    **ANAC 2018 competitor.**

    Simple time-based agent with persistent history. Uses logarithmic
    lower bound formula for concession.

    **Offering Strategy:**
        - Lower bound: t/2.5 - log10(t/2 + 0.1)
        - Generates random bids within utility bounds

    **Acceptance Strategy:**
        - Accepts if utility exceeds lower bound

    **Opponent Modeling:**
        Uses persistent history from previous sessions but no explicit
        opponent model.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.sontag.Sontag"
        super().__init__(**kwargs)


class Yeela(GeniusNegotiator):
    """
    Yeela negotiation agent.

    **ANAC 2018 competitor.**

    Uses Learner class with genetic algorithm style optimization. Tracks
    best received offer and may offer it if better than generated bids.

    **Offering Strategy:**
        - Generates bids via Learner optimization
        - May offer best received bid if it's better than generated
        - Compares new offers against all previous bids

    **Acceptance Strategy:**
        - Gives up negotiation at t=0.75 if no acceptable agreement

    **Opponent Modeling:**
        Tracks best received offer from opponents.

    Note:
        AI-generated summary based on Java source. May not be fully accurate.
        Refer to original implementation for authoritative details.

    References:
        Baarslag, T., et al. (2019). Proceedings of the ANAC 2018 Multilateral
        Negotiation League.
    """

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2018.yeela.Yeela"
        super().__init__(**kwargs)
