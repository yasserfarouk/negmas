"""Python-native Genius BOA Negotiator implementations.

This module contains Python implementations of classic Genius agents as BOANegotiator
subclasses. These negotiators use the transcompiled Genius BOA components and do NOT
require the Java Genius bridge.

The naming convention uses a 'G' prefix to distinguish these Python-native implementations
from the Java-bridge versions in gnegotiators.py.

References:
    - ANAC (Automated Negotiating Agents Competition): https://ii.tudelft.nl/ANAC/
    - Genius: https://ii.tudelft.nl/genius/
"""

from __future__ import annotations

from negmas.gb.negotiators.modular.boa import BOANegotiator
from negmas.gb.components.genius import (
    # Acceptance policies
    GACNext,
    GACCombi,
    GACConst,
    GACPrevious,
    GACTrue,
    GACCombiMax,
    GACCombiMaxInWindow,
    # Offering policies
    GTimeDependentOffering,
    GBoulwareOffering,
    GConcederOffering,
    GLinearOffering,
    GHardlinerOffering,
    GRandomOffering,
    # Opponent models
    GDefaultModel,
    GHardHeadedFrequencyModel,
    GSmithFrequencyModel,
    GAgentXFrequencyModel,
)

__all__ = [
    # Classic time-dependent agents
    "GBoulware",
    "GConceder",
    "GLinear",
    "GHardliner",
    # ANAC competition winners/notable agents
    "GHardHeaded",
    "GAgentK",
    "GAgentSmith",
    "GNozomi",
    "GFSEGA",
    "GCUHKAgent",
    "GAgentLG",
    "GAgentX",
    # Utility agents
    "GRandom",
]


# =============================================================================
# Component construction helper
# =============================================================================

_PARAM_DOC = """
        offering: A ready offering policy to use instead of the default one
            (``offering_params`` is then ignored).
        acceptance: A ready acceptance policy to use instead of the default one
            (``acceptance_params`` is then ignored).
        model: A ready opponent model to use instead of the default one
            (``model_params`` is then ignored).
        offering_params: Extra keyword arguments forwarded to the default
            offering policy's constructor. Every hyperparameter of the
            transcompiled Genius strategy is reachable this way (see the policy
            class for the full list).
        acceptance_params: Extra keyword arguments forwarded to the default
            acceptance policy's constructor.
        model_params: Extra keyword arguments forwarded to the default opponent
            model's constructor.
"""


def _make_components(
    kwargs: dict,
    offering_type,
    acceptance_type,
    model_type=None,
    offering_kwargs: dict | None = None,
    acceptance_kwargs: dict | None = None,
):
    """Builds the (offering, acceptance, model) triple for a ``G*`` negotiator.

    Any of the three components may be supplied ready-made via ``kwargs``
    (``offering=`` / ``acceptance=`` / ``model=``); otherwise it is constructed
    from its default type merged with the corresponding ``*_params`` dict popped
    from ``kwargs``. ``acceptance_type`` is called with ``offering_policy=`` when
    it accepts that argument.

    Returns:
        A ``(offering, acceptance, model)`` tuple. ``model`` is ``None`` when
        ``model_type`` is ``None`` and none was supplied.
    """
    offering_params = dict(kwargs.pop("offering_params", None) or {})
    acceptance_params = dict(kwargs.pop("acceptance_params", None) or {})
    model_params = dict(kwargs.pop("model_params", None) or {})

    offering = kwargs.pop("offering", None)
    if offering is None:
        offering = offering_type(**{**(offering_kwargs or {}), **offering_params})

    acceptance = kwargs.pop("acceptance", None)
    if acceptance is None:
        base = dict(acceptance_kwargs or {})
        try:
            acceptance = acceptance_type(
                offering_policy=offering, **{**base, **acceptance_params}
            )
        except TypeError:
            acceptance = acceptance_type(**{**base, **acceptance_params})

    model = kwargs.pop("model", None)
    if model is None and model_type is not None:
        model = model_type(**model_params)

    return offering, acceptance, model


# =============================================================================
# Classic Time-Dependent Agents
# =============================================================================


class GBoulware(BOANegotiator):
    """
    Python-native Boulware negotiator.

    A time-dependent agent with e < 1 (typically e=0.2), which concedes slowly
    and only makes significant concessions near the deadline. This is a conservative
    negotiation strategy.

    The Boulware strategy is named after the labor negotiator Lemuel Boulware,
    known for making "take it or leave it" offers.

    Uses:
        - Offering: GBoulwareOffering (e=0.2)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        """Initialize the Boulware negotiator.

        Args:
            **kwargs: Additional arguments passed to `BOANegotiator`, plus the
                component overrides documented in ``_PARAM_DOC``.
        """
        offering, acceptance, _ = _make_components(kwargs, GBoulwareOffering, GACNext)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GConceder(BOANegotiator):
    """
    Python-native Conceder negotiator.

    A time-dependent agent with e > 1 (typically e=2.0), which concedes quickly
    early in the negotiation. This is an accommodating negotiation strategy.

    Uses:
        - Offering: GConcederOffering (e=2.0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        """Initialize the Conceder negotiator.

        Args:
            **kwargs: Additional arguments passed to `BOANegotiator`, plus the
                component overrides documented in ``_PARAM_DOC``.
        """
        offering, acceptance, _ = _make_components(kwargs, GConcederOffering, GACNext)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GLinear(BOANegotiator):
    """
    Python-native Linear negotiator.

    A time-dependent agent with e = 1, which concedes at a constant rate
    throughout the negotiation.

    Uses:
        - Offering: GLinearOffering (e=1.0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        """Initialize the Linear negotiator.

        Args:
            **kwargs: Additional arguments passed to `BOANegotiator`, plus the
                component overrides documented in ``_PARAM_DOC``.
        """
        offering, acceptance, _ = _make_components(kwargs, GLinearOffering, GACNext)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


class GHardliner(BOANegotiator):
    """
    Python-native Hardliner negotiator.

    A time-dependent agent with e = 0, which never concedes and always offers
    its best outcome. This is the most aggressive negotiation strategy.

    Uses:
        - Offering: GHardlinerOffering (e=0)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        """Initialize the Hardliner negotiator.

        Args:
            **kwargs: Additional arguments passed to `BOANegotiator`, plus the
                component overrides documented in ``_PARAM_DOC``.
        """
        offering, acceptance, _ = _make_components(kwargs, GHardlinerOffering, GACNext)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)


# =============================================================================
# ANAC Competition Winners and Notable Agents
# =============================================================================


class GHardHeaded(BOANegotiator):
    """
    Python-native HardHeaded agent (ANAC 2011 Winner).

    HardHeaded (KLH) won the ANAC 2011 competition. It uses a Boulware-style
    time-dependent offering strategy combined with frequency-based opponent
    modeling to estimate the opponent's preferences.

    The agent tracks which issues remain unchanged in opponent offers to infer
    which issues are most important to the opponent.

    Uses:
        - Offering: GTimeDependentOffering (e=0.2, Boulware-style)
        - Acceptance: GACNext (accepts if opponent's offer >= next planned offer)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - van Krimpen, T.,";"; man, D.;"; Hindriks, K. (2011). "HardHeaded".
          ANAC 2011.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize HardHeaded agent.

        Args:
            e: The time-dependency exponent (default 0.2 for Boulware behavior)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACNext,
            GHardHeadedFrequencyModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=None,
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentK(BOANegotiator):
    """
    Python-native AgentK (ANAC 2010).

    AgentK was one of the top performers in ANAC 2010. It uses a time-dependent
    offering strategy with adaptive acceptance based on a combination of
    conditions (AC_Combi).

    Uses:
        - Offering: GTimeDependentOffering (e=0.2)
        - Acceptance: GACCombi (combined acceptance conditions)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Kawaguchi, S.; Fujita, K.; Ito, T. (2010). "AgentK: Compromising
          Strategy based on Estimated Maximum Utility for Automated Negotiating
          Agents". ANAC 2010.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize AgentK.

        Args:
            e: The time-dependency exponent (default 0.2)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACCombi,
            GDefaultModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=None,
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentSmith(BOANegotiator):
    """
    Python-native AgentSmith (ANAC 2010).

    AgentSmith was a notable agent in ANAC 2010, using a time-dependent strategy
    with frequency-based opponent modeling. It uses AC_Const acceptance which
    accepts any offer above a fixed utility threshold.

    Uses:
        - Offering: GTimeDependentOffering (e=0.3)
        - Acceptance: GACConst (accepts above fixed threshold)
        - Model: GSmithFrequencyModel (Smith-style frequency model)

    References:
        - de Jonge, D. (2010). "AgentSmith". ANAC 2010.
    """

    def __init__(self, e: float = 0.3, c: float = 0.8, **kwargs):
        """
        Initialize AgentSmith.

        Args:
            e: The time-dependency exponent (default 0.3)
            c: The constant acceptance threshold (default 0.8)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACConst,
            GSmithFrequencyModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=dict(c=c),
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GNozomi(BOANegotiator):
    """
    Python-native Nozomi agent (ANAC 2010).

    Nozomi was a competitive agent in ANAC 2010. It uses a Boulware-style
    time-dependent offering with AC_Previous acceptance, which accepts if
    the opponent's offer is better than their previous offer.

    Uses:
        - Offering: GTimeDependentOffering (e=0.2)
        - Acceptance: GACPrevious (accepts if better than opponent's previous offer)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Fujita, K. (2010). "Nozomi". ANAC 2010.
    """

    def __init__(self, e: float = 0.2, **kwargs):
        """
        Initialize Nozomi.

        Args:
            e: The time-dependency exponent (default 0.2)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACPrevious,
            GDefaultModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=None,
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GFSEGA(BOANegotiator):
    """
    Python-native FSEGA agent (ANAC 2010).

    AgentFSEGA (Faculty of Computer Science Agent) from ANAC 2010 uses a
    conceding strategy with constant threshold acceptance.

    Uses:
        - Offering: GConcederOffering (e=2.0)
        - Acceptance: GACConst (accepts above fixed threshold)
        - Model: GDefaultModel (basic opponent model)

    References:
        - Zaharia, G.; et al. (2010). "AgentFSEGA". ANAC 2010.
    """

    def __init__(self, c: float = 0.7, **kwargs):
        """
        Initialize FSEGA.

        Args:
            c: The constant acceptance threshold (default 0.7)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GConcederOffering,
            GACConst,
            GDefaultModel,
            offering_kwargs=None,
            acceptance_kwargs=dict(c=c),
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GCUHKAgent(BOANegotiator):
    """
    Python-native CUHKAgent (ANAC 2012 Winner).

    CUHKAgent won ANAC 2012. It uses a sophisticated time-dependent strategy
    with combined acceptance conditions. The agent is particularly good at
    handling domains with discount factors.

    Uses:
        - Offering: GTimeDependentOffering (e=0.15, more conservative)
        - Acceptance: GACCombi (combined acceptance conditions)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - Hao, J.; Leung, H. (2012). "CUHKAgent: A Strategy for Bilateral
          Multi-issue Negotiation". ANAC 2012.
    """

    def __init__(self, e: float = 0.15, **kwargs):
        """
        Initialize CUHKAgent.

        Args:
            e: The time-dependency exponent (default 0.15 for conservative behavior)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACCombi,
            GHardHeadedFrequencyModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=None,
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentLG(BOANegotiator):
    """
    Python-native AgentLG (ANAC 2012).

    AgentLG was a notable agent in ANAC 2012. It uses a time-dependent
    strategy with AC_CombiMax acceptance which considers both the opponent's
    best offer and the time remaining.

    Uses:
        - Offering: GTimeDependentOffering (e=0.25)
        - Acceptance: GACCombiMax (maximum-based combined acceptance)
        - Model: GHardHeadedFrequencyModel (frequency-based opponent modeling)

    References:
        - ANAC 2012 AgentLG.
    """

    def __init__(self, e: float = 0.25, **kwargs):
        """
        Initialize AgentLG.

        Args:
            e: The time-dependency exponent (default 0.25)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACCombiMax,
            GHardHeadedFrequencyModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=None,
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


class GAgentX(BOANegotiator):
    """
    Python-native AgentX (ANAC 2015).

    AgentX was a notable agent in ANAC 2015. It uses adaptive time-dependent
    offering with window-based acceptance and exponential smoothing opponent
    modeling.

    Uses:
        - Offering: GTimeDependentOffering (e=0.3, adaptive)
        - Acceptance: GACCombiMaxInWindow (window-based maximum acceptance)
        - Model: GAgentXFrequencyModel (exponential smoothing opponent model)

    References:
        - ANAC 2015 AgentX.
    """

    def __init__(self, e: float = 0.3, t: float = 0.98, **kwargs):
        """
        Initialize AgentX.

        Args:
            e: The time-dependency exponent (default 0.3)
            t: Time threshold after which window-based acceptance kicks in (default 0.98)
            offering_params: Keyword arguments forwarded to the default offering
                policy (every Genius hyperparameter is reachable this way).
            acceptance_params: Keyword arguments forwarded to the default
                acceptance policy.
            model_params: Keyword arguments forwarded to the default opponent
                model.
            **kwargs: Additional arguments passed to BOANegotiator. A ready
                ``offering`` / ``acceptance`` / ``model`` may also be passed.
        """
        offering, acceptance, model = _make_components(
            kwargs,
            GTimeDependentOffering,
            GACCombiMaxInWindow,
            GAgentXFrequencyModel,
            offering_kwargs=dict(e=e),
            acceptance_kwargs=dict(t=t),
        )
        super().__init__(
            offering=offering, acceptance=acceptance, model=model, **kwargs
        )


# =============================================================================
# Utility Agents
# =============================================================================


class GRandom(BOANegotiator):
    """
    Python-native Random negotiator.

    A simple agent that makes random offers and accepts any offer. Useful for
    testing and as a baseline.

    Uses:
        - Offering: GRandomOffering (random bid selection)
        - Acceptance: GACTrue (accepts any offer)
        - Model: None (no opponent modeling)
    """

    def __init__(self, **kwargs):
        """Initialize the Random negotiator.

        Args:
            **kwargs: Additional arguments passed to `BOANegotiator`, plus the
                component overrides documented in ``_PARAM_DOC``.
        """
        offering, acceptance, _ = _make_components(kwargs, GRandomOffering, GACTrue)
        super().__init__(offering=offering, acceptance=acceptance, **kwargs)
