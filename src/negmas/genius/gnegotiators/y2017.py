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
    """AgentF implementation."""

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
    """CaduceusDC16 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.caduceusdc16.CaduceusDC16"
        super().__init__(**kwargs)


class Farma17(GeniusNegotiator):
    """Farma17 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class Farma2017(GeniusNegotiator):
    """Farma2017 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.farma.Farma17"
        super().__init__(**kwargs)


class GeneKing(GeniusNegotiator):
    """GeneKing implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.geneking.GeneKing"
        super().__init__(**kwargs)


class Gin(GeniusNegotiator):
    """Gin implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.gin.Gin"
        super().__init__(**kwargs)


class Group3(GeniusNegotiator):
    """Group3 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.group3.Group3"
        super().__init__(**kwargs)


class Imitator(GeniusNegotiator):
    """Imitator implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.limitator.Imitator"
        super().__init__(**kwargs)


class MadAgent(GeniusNegotiator):
    """MadAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.madagent.MadAgent"
        super().__init__(**kwargs)


class Mamenchis(GeniusNegotiator):
    """Mamenchis implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.mamenchis.Mamenchis"
        super().__init__(**kwargs)


class Mosa(GeniusNegotiator):
    """Mosa implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.mosateam.Mosa"
        super().__init__(**kwargs)


class ParsAgent3(GeniusNegotiator):
    """ParsAgent3 implementation."""

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
    """ShahAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.parsagent3.ShahAgent"
        super().__init__(**kwargs)


class SimpleAgent2017(GeniusNegotiator):
    """SimpleAgent2017 implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.simpleagent.SimpleAgent"
        super().__init__(**kwargs)


class TaxiBox(GeniusNegotiator):
    """TaxiBox implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.tangxun.taxibox"
        super().__init__(**kwargs)


class TucAgent(GeniusNegotiator):
    """TucAgent implementation."""

    def __init__(self, **kwargs):
        """Initialize the instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        kwargs["java_class_name"] = "agents.anac.y2017.tucagent.TucAgent"
        super().__init__(**kwargs)
