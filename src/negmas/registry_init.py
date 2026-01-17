"""Initialization module that registers all built-in mechanisms, negotiators, and components.

This module is automatically imported when negmas.registry is imported,
ensuring all built-in classes are registered in the registries.

Note: Only concrete, usable classes are registered - not base classes.
"""

from __future__ import annotations

from pathlib import Path

from negmas.registry import (
    mechanism_registry,
    negotiator_registry,
    component_registry,
    scenario_registry,
)

__all__: list[str] = []

# Source identifier for all built-in registrations
NEGMAS_SOURCE = "negmas"


def _register_mechanisms() -> None:
    """Register all built-in mechanisms.

    Note: Only concrete mechanisms are registered, not base classes like Mechanism.

    Mechanism tags indicate their type and requirements:
    - "builtin": Built into negmas
    - "sao": Single Agreement Only protocol
    - "gb": General Bargaining protocol
    - "tau": TAU protocol (no deadline required)
    - "st": Single Text protocol
    - "requires-deadline": Requires n_steps or time_limit
    - "propose": Requires negotiators to have propose capability
    - "respond": Requires negotiators to have respond capability
    """
    from negmas.sao.mechanism import SAOMechanism
    from negmas.gb.mechanisms.base import ParallelGBMechanism, SerialGBMechanism
    from negmas.gb.mechanisms.tau import TAUMechanism
    from negmas.gb.mechanisms.tauserial import SerialTAUMechanism
    from negmas.st import HillClimbingSTMechanism, VetoSTMechanism

    # Register SAO mechanisms
    # Note: requires_deadline is now expressed via the "requires-deadline" tag
    mechanism_registry.register(
        SAOMechanism,
        short_name="SAOMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "sao", "requires-deadline", "propose", "respond"},
    )

    # Register GB mechanisms (concrete implementations)
    mechanism_registry.register(
        ParallelGBMechanism,
        short_name="ParallelGBMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "gb", "requires-deadline", "propose", "respond"},
    )
    mechanism_registry.register(
        SerialGBMechanism,
        short_name="SerialGBMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "gb", "requires-deadline", "propose", "respond"},
    )

    # Register TAU mechanisms - they don't require a deadline (no "requires-deadline" tag)
    mechanism_registry.register(
        TAUMechanism,
        short_name="TAUMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "gb", "tau", "propose", "respond"},
    )
    mechanism_registry.register(
        SerialTAUMechanism,
        short_name="SerialTAUMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "gb", "tau", "propose", "respond"},
    )

    # Register ST mechanisms
    mechanism_registry.register(
        HillClimbingSTMechanism,
        short_name="HillClimbingSTMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "st", "requires-deadline"},
    )
    mechanism_registry.register(
        VetoSTMechanism,
        short_name="VetoSTMechanism",
        source=NEGMAS_SOURCE,
        tags={"builtin", "st", "requires-deadline"},
    )


def _register_sao_negotiators() -> None:
    """Register SAO negotiators.

    Note: Base classes like SAONegotiator are not registered.

    Negotiator tags indicate their capabilities and characteristics:
    - "builtin": Built into negmas (not from Genius)
    - "sao": Works with SAO protocol
    - "propose": Can propose offers
    - "respond": Can respond to offers
    - "aspiration": Uses aspiration-based strategy
    - "random": Uses random strategy
    - "tit-for-tat": Uses tit-for-tat strategy
    - "time-based": Uses time-based concession
    - "utility-based": Uses utility-based strategy
    - "learning": Learns from interaction
    """
    from negmas.sao.negotiators import (
        AspirationNegotiator,
        ToughNegotiator,
        NiceNegotiator,
        RandomNegotiator,
        RandomAlwaysAcceptingNegotiator,
        LimitedOutcomesNegotiator,
        LimitedOutcomesAcceptor,
        NaiveTitForTatNegotiator,
        SimpleTitForTatNegotiator,
        TimeBasedNegotiator,
        TimeBasedConcedingNegotiator,
        BoulwareTBNegotiator,
        ConcederTBNegotiator,
        LinearTBNegotiator,
        UtilBasedNegotiator,
        CABNegotiator,
        CANNegotiator,
        CARNegotiator,
        WABNegotiator,
        WANNegotiator,
        WARNegotiator,
        MiCRONegotiator,
        FastMiCRONegotiator,
    )

    # Base tags for all builtin SAO negotiators
    base_tags = {"builtin", "sao", "propose", "respond"}

    # Aspiration negotiators
    negotiator_registry.register(
        AspirationNegotiator,
        short_name="AspirationNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"aspiration", "time-based"},
    )
    negotiator_registry.register(
        ToughNegotiator,
        short_name="ToughNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"aspiration"},
    )
    negotiator_registry.register(
        NiceNegotiator,
        short_name="NiceNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"aspiration"},
    )

    # Random negotiators
    negotiator_registry.register(
        RandomNegotiator,
        short_name="RandomNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"random"},
    )
    negotiator_registry.register(
        RandomAlwaysAcceptingNegotiator,
        short_name="RandomAlwaysAcceptingNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"random"},
    )

    # Limited outcomes negotiators
    negotiator_registry.register(
        LimitedOutcomesNegotiator,
        short_name="LimitedOutcomesNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"limited-outcomes"},
    )
    negotiator_registry.register(
        LimitedOutcomesAcceptor,
        short_name="LimitedOutcomesAcceptor",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"limited-outcomes"},
    )

    # Tit-for-tat negotiators
    negotiator_registry.register(
        NaiveTitForTatNegotiator,
        short_name="NaiveTitForTatNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"tit-for-tat"},
    )
    negotiator_registry.register(
        SimpleTitForTatNegotiator,
        short_name="SimpleTitForTatNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"tit-for-tat"},
    )

    # Time-based negotiators
    negotiator_registry.register(
        TimeBasedNegotiator,
        short_name="TimeBasedNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"time-based"},
    )
    negotiator_registry.register(
        TimeBasedConcedingNegotiator,
        short_name="TimeBasedConcedingNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"time-based"},
    )
    negotiator_registry.register(
        BoulwareTBNegotiator,
        short_name="BoulwareTBNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"time-based", "boulware"},
    )
    negotiator_registry.register(
        ConcederTBNegotiator,
        short_name="ConcederTBNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"time-based", "conceder"},
    )
    negotiator_registry.register(
        LinearTBNegotiator,
        short_name="LinearTBNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"time-based", "linear"},
    )

    # Utility-based negotiators
    negotiator_registry.register(
        UtilBasedNegotiator,
        short_name="UtilBasedNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"utility-based"},
    )

    # CAB family negotiators (Concede-to-Agreement-Based)
    negotiator_registry.register(
        CABNegotiator,
        short_name="CABNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"cab-family", "learning"},
    )
    negotiator_registry.register(
        CANNegotiator,
        short_name="CANNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"cab-family", "learning"},
    )
    negotiator_registry.register(
        CARNegotiator,
        short_name="CARNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"cab-family", "learning"},
    )

    # WAB family negotiators (Walk-Away-Based)
    negotiator_registry.register(
        WABNegotiator,
        short_name="WABNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"wab-family", "learning"},
    )
    negotiator_registry.register(
        WANNegotiator,
        short_name="WANNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"wab-family", "learning"},
    )
    negotiator_registry.register(
        WARNegotiator,
        short_name="WARNegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"wab-family", "learning"},
    )

    # MiCRO negotiators
    negotiator_registry.register(
        MiCRONegotiator,
        short_name="MiCRONegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"micro", "learning"},
    )
    negotiator_registry.register(
        FastMiCRONegotiator,
        short_name="FastMiCRONegotiator",
        source=NEGMAS_SOURCE,
        tags=base_tags | {"micro", "learning"},
    )


def _register_sao_components() -> None:
    """Register SAO components (acceptance policies, offering policies, models).

    Note: Base classes like AcceptancePolicy and OfferingPolicy are not registered.

    Component tags indicate their type and characteristics:
    - "builtin": Built into negmas
    - "sao": For SAO protocol
    - "acceptance": Acceptance policy
    - "offering": Offering policy
    - "model": Opponent model
    - "rational": Uses rational/utility-based decisions
    - "random": Uses random decisions
    - "time-based": Uses time-based logic
    """
    # Import specific acceptance policies from sao.components.acceptance
    from negmas.sao.components.acceptance import (
        AcceptImmediately,
        RejectAlways,
        AcceptAnyRational,
        AcceptBetterRational,
        AcceptNotWorseRational,
        AcceptAbove,
        AcceptBetween,
        AcceptAround,
        AcceptBest,
        AcceptTop,
        AcceptAfter,
        RandomAcceptancePolicy,
        EndImmediately,
        ACConst,
        ACNext,
        ACLast,
        ACTime,
        ACLastKReceived,
        ACLastFractionReceived,
    )

    # Acceptance policies with appropriate tags
    acceptance_base = {"builtin", "sao"}
    acceptance_policies = [
        (AcceptImmediately, {"simple"}),
        (RejectAlways, {"simple"}),
        (AcceptAnyRational, {"rational"}),
        (AcceptBetterRational, {"rational"}),
        (AcceptNotWorseRational, {"rational"}),
        (AcceptAbove, {"threshold"}),
        (AcceptBetween, {"threshold"}),
        (AcceptAround, {"threshold"}),
        (AcceptBest, {"optimal"}),
        (AcceptTop, {"optimal"}),
        (AcceptAfter, {"time-based"}),
        (RandomAcceptancePolicy, {"random"}),
        (EndImmediately, {"simple"}),
        (ACConst, {"threshold"}),
        (ACNext, {"adaptive"}),
        (ACLast, {"adaptive"}),
        (ACTime, {"time-based"}),
        (ACLastKReceived, {"adaptive"}),
        (ACLastFractionReceived, {"adaptive"}),
    ]
    for cls, extra_tags in acceptance_policies:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="acceptance",
            tags=acceptance_base | extra_tags,
        )

    # Import offering policies from sao.components.offering
    from negmas.sao.components.offering import (
        RandomOfferingPolicy,
        NoneOfferingPolicy,
        OfferBest,
        OfferTop,
        TimeBasedOfferingPolicy,
    )

    offering_base = {"builtin", "sao"}
    offering_policies = [
        (RandomOfferingPolicy, {"random"}),
        (NoneOfferingPolicy, {"simple"}),
        (OfferBest, {"optimal"}),
        (OfferTop, {"optimal"}),
        (TimeBasedOfferingPolicy, {"time-based"}),
    ]
    for cls, extra_tags in offering_policies:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="offering",
            tags=offering_base | extra_tags,
        )

    # Import models from sao.components.models
    from negmas.sao.components.models import (
        ZeroSumModel,
        FrequencyUFunModel,
        FrequencyLinearUFunModel,
    )

    model_base = {"builtin", "sao"}
    models = [
        (ZeroSumModel, {"zero-sum"}),
        (FrequencyUFunModel, {"frequency", "learning"}),
        (FrequencyLinearUFunModel, {"frequency", "learning", "linear"}),
    ]
    for cls, extra_tags in models:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="model",
            tags=model_base | extra_tags,
        )


def _register_genius_boa_components() -> None:
    """Register Genius BOA components (transcompiled from Java).

    These are acceptance policies, offering policies, and opponent models
    from the Genius negotiation framework.

    Component tags:
    - "genius": From Genius framework
    - "boa": Part of Genius BOA architecture
    - "ai-generated": Transcompiled with AI assistance
    - "genius-translated": Direct translation from Java source
    """
    # Import Genius acceptance policies
    from negmas.gb.components.genius.acceptance import (
        # Base acceptance strategies
        GACNext,
        GACConst,
        GACTime,
        GACPrevious,
        GACGap,
        GACCombi,
        GACCombiMaxInWindow,
        GACTrue,
        GACFalse,
        GACConstDiscounted,
        GACCombiAvg,
        GACCombiBestAvg,
        GACCombiMax,
        GACCombiV2,
        GACCombiV3,
        GACCombiV4,
        GACCombiBestAvgDiscounted,
        GACCombiMaxInWindowDiscounted,
        GACCombiProb,
        GACCombiProbDiscounted,
        # ANAC 2010 acceptance strategies
        GACABMP,
        GACAgentK,
        GACAgentFSEGA,
        GACIAMCrazyHaggler,
        GACYushu,
        GACNozomi,
        GACIAMHaggler2010,
        GACAgentSmith,
        # ANAC 2011 acceptance strategies
        GACHardHeaded,
        GACAgentK2,
        GACBRAMAgent,
        GACGahboninho,
        GACNiceTitForTat,
        GACTheNegotiator,
        GACValueModelAgent,
        GACIAMHaggler2011,
        # ANAC 2012 acceptance strategies
        GACCUHKAgent,
        GACOMACagent,
        GACAgentLG,
        GACAgentMR,
        GACBRAMAgent2,
        GACIAMHaggler2012,
        GACTheNegotiatorReloaded,
        # ANAC 2013 acceptance strategies
        GACTheFawkes,
        GACInoxAgent,
        GACInoxAgentOneIssue,
        # Other acceptance strategies
        GACUncertain,
        GACMAC,
    )

    genius_acceptance_base = {
        "genius",
        "boa",
        "gb",
        "ai-generated",
        "genius-translated",
    }
    genius_acceptance = [
        # Base acceptance strategies
        (GACNext, {"adaptive"}),
        (GACConst, {"threshold"}),
        (GACTime, {"time-based"}),
        (GACPrevious, {"adaptive"}),
        (GACGap, {"adaptive"}),
        (GACCombi, {"combined"}),
        (GACCombiMaxInWindow, {"combined", "windowed"}),
        (GACTrue, {"simple"}),
        (GACFalse, {"simple"}),
        (GACConstDiscounted, {"threshold", "discounted"}),
        (GACCombiAvg, {"combined"}),
        (GACCombiBestAvg, {"combined"}),
        (GACCombiMax, {"combined"}),
        (GACCombiV2, {"combined"}),
        (GACCombiV3, {"combined"}),
        (GACCombiV4, {"combined"}),
        (GACCombiBestAvgDiscounted, {"combined", "discounted"}),
        (GACCombiMaxInWindowDiscounted, {"combined", "windowed", "discounted"}),
        (GACCombiProb, {"combined", "probabilistic"}),
        (GACCombiProbDiscounted, {"combined", "probabilistic", "discounted"}),
        # ANAC 2010
        (GACABMP, {"anac2010"}),
        (GACAgentK, {"anac2010"}),
        (GACAgentFSEGA, {"anac2010"}),
        (GACIAMCrazyHaggler, {"anac2010"}),
        (GACYushu, {"anac2010"}),
        (GACNozomi, {"anac2010"}),
        (GACIAMHaggler2010, {"anac2010"}),
        (GACAgentSmith, {"anac2010"}),
        # ANAC 2011
        (GACHardHeaded, {"anac2011"}),
        (GACAgentK2, {"anac2011"}),
        (GACBRAMAgent, {"anac2011"}),
        (GACGahboninho, {"anac2011"}),
        (GACNiceTitForTat, {"anac2011", "tit-for-tat"}),
        (GACTheNegotiator, {"anac2011"}),
        (GACValueModelAgent, {"anac2011"}),
        (GACIAMHaggler2011, {"anac2011"}),
        # ANAC 2012
        (GACCUHKAgent, {"anac2012"}),
        (GACOMACagent, {"anac2012"}),
        (GACAgentLG, {"anac2012"}),
        (GACAgentMR, {"anac2012"}),
        (GACBRAMAgent2, {"anac2012"}),
        (GACIAMHaggler2012, {"anac2012"}),
        (GACTheNegotiatorReloaded, {"anac2012"}),
        # ANAC 2013
        (GACTheFawkes, {"anac2013"}),
        (GACInoxAgent, {"anac2013"}),
        (GACInoxAgentOneIssue, {"anac2013"}),
        # Other
        (GACUncertain, {"uncertainty"}),
        (GACMAC, {"multi-attribute"}),
    ]
    for cls, extra_tags in genius_acceptance:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="acceptance",
            tags=genius_acceptance_base | extra_tags,
        )

    # Import Genius offering policies
    from negmas.gb.components.genius.offering import (
        # Base offering strategies
        GTimeDependentOffering,
        GRandomOffering,
        GBoulwareOffering,
        GConcederOffering,
        GLinearOffering,
        GHardlinerOffering,
        GChoosingAllBids,
        # ANAC 2010 offering strategies
        GIAMCrazyHagglerOffering,
        GAgentKOffering,
        GAgentFSEGAOffering,
        GAgentSmithOffering,
        GNozomiOffering,
        GYushuOffering,
        GIAMhaggler2010Offering,
        # ANAC 2011 offering strategies
        GHardHeadedOffering,
        GAgentK2Offering,
        GBRAMAgentOffering,
        GGahboninhoOffering,
        GNiceTitForTatOffering,
        GTheNegotiatorOffering,
        GValueModelAgentOffering,
        GIAMhaggler2011Offering,
        # ANAC 2012 offering strategies
        GCUHKAgentOffering,
        GOMACagentOffering,
        GAgentLGOffering,
        GAgentMROffering,
        GBRAMAgent2Offering,
        GIAMHaggler2012Offering,
        GTheNegotiatorReloadedOffering,
        # ANAC 2013 offering strategies
        GFawkesOffering,
        GInoxAgentOffering,
    )

    genius_offering_base = {"genius", "boa", "gb", "ai-generated", "genius-translated"}
    genius_offering = [
        # Base offering strategies
        (GTimeDependentOffering, {"time-based"}),
        (GRandomOffering, {"random"}),
        (GBoulwareOffering, {"time-based", "boulware"}),
        (GConcederOffering, {"time-based", "conceder"}),
        (GLinearOffering, {"time-based", "linear"}),
        (GHardlinerOffering, {"hardliner"}),
        (GChoosingAllBids, {"exhaustive"}),
        # ANAC 2010
        (GIAMCrazyHagglerOffering, {"anac2010", "hardliner"}),
        (GAgentKOffering, {"anac2010", "adaptive"}),
        (GAgentFSEGAOffering, {"anac2010", "adaptive"}),
        (GAgentSmithOffering, {"anac2010", "time-based"}),
        (GNozomiOffering, {"anac2010", "adaptive"}),
        (GYushuOffering, {"anac2010", "sigmoid"}),
        (GIAMhaggler2010Offering, {"anac2010", "conservative"}),
        # ANAC 2011
        (GHardHeadedOffering, {"anac2011", "conservative"}),
        (GAgentK2Offering, {"anac2011", "adaptive"}),
        (GBRAMAgentOffering, {"anac2011", "opponent-modeling"}),
        (GGahboninhoOffering, {"anac2011", "adaptive"}),
        (GNiceTitForTatOffering, {"anac2011", "tit-for-tat"}),
        (GTheNegotiatorOffering, {"anac2011", "piecewise"}),
        (GValueModelAgentOffering, {"anac2011", "value-modeling"}),
        (GIAMhaggler2011Offering, {"anac2011", "conservative"}),
        # ANAC 2012
        (GCUHKAgentOffering, {"anac2012", "adaptive"}),
        (GOMACagentOffering, {"anac2012", "prediction"}),
        (GAgentLGOffering, {"anac2012", "learning"}),
        (GAgentMROffering, {"anac2012", "risk-based"}),
        (GBRAMAgent2Offering, {"anac2012", "opponent-modeling"}),
        (GIAMHaggler2012Offering, {"anac2012", "conservative"}),
        (GTheNegotiatorReloadedOffering, {"anac2012", "piecewise"}),
        # ANAC 2013
        (GFawkesOffering, {"anac2013", "prediction"}),
        (GInoxAgentOffering, {"anac2013", "adaptive"}),
    ]
    for cls, extra_tags in genius_offering:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="offering",
            tags=genius_offering_base | extra_tags,
        )

    # Import Genius opponent models
    from negmas.gb.components.genius.models import (
        # Base opponent models
        GHardHeadedFrequencyModel,
        GDefaultModel,
        GUniformModel,
        GOppositeModel,
        GSmithFrequencyModel,
        GAgentXFrequencyModel,
        GNashFrequencyModel,
        GBayesianModel,
        GScalableBayesianModel,
        # Additional opponent models
        GFSEGABayesianModel,
        GIAMhagglerBayesianModel,
        GCUHKFrequencyModel,
        GAgentLGModel,
        GTheFawkesModel,
        GInoxAgentModel,
        GWorstModel,
        GPerfectModel,
    )

    genius_model_base = {"genius", "boa", "gb", "ai-generated", "genius-translated"}
    genius_models = [
        # Base opponent models
        (GHardHeadedFrequencyModel, {"frequency", "learning"}),
        (GDefaultModel, {"simple"}),
        (GUniformModel, {"simple"}),
        (GOppositeModel, {"simple"}),
        (GSmithFrequencyModel, {"frequency", "learning"}),
        (GAgentXFrequencyModel, {"frequency", "learning"}),
        (GNashFrequencyModel, {"frequency", "nash", "learning"}),
        (GBayesianModel, {"bayesian", "learning"}),
        (GScalableBayesianModel, {"bayesian", "learning", "scalable"}),
        # Additional opponent models
        (GFSEGABayesianModel, {"bayesian", "learning", "anac2010"}),
        (GIAMhagglerBayesianModel, {"bayesian", "learning"}),
        (GCUHKFrequencyModel, {"frequency", "learning", "anac2012"}),
        (GAgentLGModel, {"learning", "anac2012"}),
        (GTheFawkesModel, {"learning", "anac2013"}),
        (GInoxAgentModel, {"learning", "anac2013"}),
        (GWorstModel, {"simple", "pessimistic"}),
        (GPerfectModel, {"oracle", "testing"}),
    ]
    for cls, extra_tags in genius_models:
        component_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            component_type="model",
            tags=genius_model_base | extra_tags,
        )


def _register_genius_negotiators() -> None:
    """Register Genius negotiators.

    Registers all Python wrappers for Genius Java negotiation agents.
    These are organized by ANAC competition year.

    Negotiator tags:
    - "genius": From Genius framework (requires Java bridge)
    - "anac": Competed in ANAC competition
    - "anac-YYYY": Year-specific tag (e.g., "anac-2019")
    - "propose": Can propose offers
    - "respond": Can respond to offers
    """
    try:
        # Import basic agents
        from negmas.genius.gnegotiators.basic import (
            ABMPAgent2,
            BayesianAgent,
            BoulwareNegotiationParty,
            ConcederNegotiationParty,
            FunctionalAcceptor,
            FuzzyAgent,
            ImmediateAcceptor,
            OptimalBidderSimple,
            RandomCounterOfferNegotiationParty,
            RandomParty,
            RandomParty2,
            SimilarityAgent,
            SimpleAgent,
            TimeDependentAgentBoulware,
            TimeDependentAgentConceder,
            TimeDependentAgentHardliner,
            TimeDependentAgentLinear,
            UtilityBasedAcceptor,
        )

        # Import others (TU Delft course agents)
        from negmas.genius.gnegotiators.others import (
            AI2014Group2,
            Group1,
            Group10,
            Group11,
            Group12,
            Group13,
            Group14,
            Group15,
            Group16,
            Group17,
            Group18,
            Group19,
            Group20,
            Group21,
            Group22,
            Group3Q2015,
            Group4,
            Group5,
            Group6,
            Group7,
            Group8,
            Group9,
            Q12015Group2,
        )

        # Import ANAC 2010 agents
        from negmas.genius.gnegotiators.y2010 import (
            AgentFSEGA,
            AgentK,
            AgentSmith,
            IAMcrazyHaggler,
            IAMhaggler,
            Nozomi,
            SouthamptonAgent,
            Yushu,
        )

        # Import ANAC 2011 agents
        from negmas.genius.gnegotiators.y2011 import (
            AgentK2,
            BramAgent,
            BramAgent2,
            Gahboninho,
            HardHeaded,
            IAMhaggler2011,
            NiceTitForTat,
            TheNegotiator,
            ValueModelAgent,
        )

        # Import ANAC 2012 agents
        from negmas.genius.gnegotiators.y2012 import (
            AgentLG,
            AgentMR,
            CUHKAgent,
            IAMhaggler2012,
            MetaAgent,
            MetaAgent2012,
            OMACagent,
            TheNegotiatorReloaded,
        )

        # Import ANAC 2013 agents
        from negmas.genius.gnegotiators.y2013 import (
            AgentI,
            AgentKF,
            GAgent,
            InoxAgent,
            MetaAgent2013,
            SlavaAgent,
            TMFAgent,
            TheFawkes,
        )

        # Import ANAC 2014 agents
        from negmas.genius.gnegotiators.y2014 import (
            AgentM,
            AgentQuest,
            AgentTD,
            AgentTRP,
            AgentYK,
            AnacSampleAgent,
            ArisawaYaki,
            Aster,
            Atlas,
            BraveCat,
            DoNA,
            E2Agent,
            Flinch,
            Gangester,
            Gangster,
            KGAgent,
            Simpatico,
            Sobut,
            TUDelftGroup2,
            WhaleAgent,
        )

        # Import ANAC 2015 agents
        from negmas.genius.gnegotiators.y2015 import (
            AgentBuyog,
            AgentH,
            AgentHP,
            AgentNeo,
            AgentW,
            AgentX,
            AresParty,
            Atlas3,
            CUHKAgent2015,
            DrageKnight,
            Group2,
            JonnyBlack,
            Kawaii,
            MeanBot,
            Mercury,
            ParsAgent,
            PhoenixParty,
            PNegotiator,
            PokerFace,
            RandomDance,
            SENGOKU,
            TUDMixedStrategyAgent,
            XianFaAgent,
            Y2015Group2,
        )

        # Import ANAC 2016 agents
        from negmas.genius.gnegotiators.y2016 import (
            AgentHP2,
            AgentLight,
            AgentSmith2016,
            Atlas32016,
            Caduceus,
            ClockworkAgent,
            Farma,
            GrandmaAgent,
            MaxOops,
            MyAgent,
            Ngent,
            ParsAgent2,
            ParsCat,
            ParsCat2,
            SYAgent,
            Terra,
            YXAgent,
        )

        # Import ANAC 2017 agents
        from negmas.genius.gnegotiators.y2017 import (
            AgentF,
            AgentKN,
            CaduceusDC16,
            Farma17,
            Farma2017,
            GeneKing,
            Gin,
            Group3,
            Imitator,
            MadAgent,
            Mamenchis,
            Mosa,
            ParsAgent3,
            PonPokoAgent,
            Rubick,
            ShahAgent,
            SimpleAgent2017,
            TaxiBox,
            TucAgent,
        )

        # Import ANAC 2018 agents
        from negmas.genius.gnegotiators.y2018 import (
            Agent33,
            Agent36,
            AgentHerb,
            AgentNP1,
            AgreeableAgent2018,
            AteamAgent,
            Ateamagent,
            BetaOne,
            Betaone,
            ConDAgent,
            ExpRubick,
            FullAgent,
            GroupY,
            IQSun2018,
            Lancelot,
            Libra,
            MengWan,
            PonPokoRampage,
            Seto,
            Shiboy,
            SMACAgent,
            Sontag,
            Yeela,
        )

        # Import ANAC 2019 agents
        from negmas.genius.gnegotiators.y2019 import (
            AgentGG,
            AgentGP,
            AgentLarry,
            DandikAgent,
            EAgent,
            FSEGA2019,
            GaravelAgent,
            Gravity,
            Group1BOA,
            HardDealer,
            KAgent,
            KakeSoba,
            MINF,
            PodAgent,
            SACRA,
            SAGA,
            SolverAgent,
            TheNewDeal,
            WinkyAgent,
        )
    except ImportError:
        # Genius support not available
        return

    # Base tags for all Genius negotiators
    genius_base = {"genius", "propose", "respond"}

    # Register basic agents (no specific ANAC year)
    basic_agents = [
        ABMPAgent2,
        BayesianAgent,
        BoulwareNegotiationParty,
        ConcederNegotiationParty,
        FunctionalAcceptor,
        FuzzyAgent,
        ImmediateAcceptor,
        OptimalBidderSimple,
        RandomCounterOfferNegotiationParty,
        RandomParty,
        RandomParty2,
        SimilarityAgent,
        SimpleAgent,
        TimeDependentAgentBoulware,
        TimeDependentAgentConceder,
        TimeDependentAgentHardliner,
        TimeDependentAgentLinear,
        UtilityBasedAcceptor,
    ]
    for cls in basic_agents:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"basic"},
        )

    # Register TU Delft course agents (no specific ANAC year)
    other_agents = [
        AI2014Group2,
        Group1,
        Group10,
        Group11,
        Group12,
        Group13,
        Group14,
        Group15,
        Group16,
        Group17,
        Group18,
        Group19,
        Group20,
        Group21,
        Group22,
        Group3Q2015,
        Group4,
        Group5,
        Group6,
        Group7,
        Group8,
        Group9,
        Q12015Group2,
    ]
    for cls in other_agents:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"tudelft"},
        )

    # Register ANAC 2010 agents
    anac_2010 = [
        AgentFSEGA,
        AgentK,
        AgentSmith,
        IAMcrazyHaggler,
        IAMhaggler,
        Nozomi,
        SouthamptonAgent,
        Yushu,
    ]
    for cls in anac_2010:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2010"},
        )

    # Register ANAC 2011 agents
    anac_2011 = [
        AgentK2,
        BramAgent,
        BramAgent2,
        Gahboninho,
        HardHeaded,
        IAMhaggler2011,
        NiceTitForTat,
        TheNegotiator,
        ValueModelAgent,
    ]
    for cls in anac_2011:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2011"},
        )

    # Register ANAC 2012 agents
    anac_2012 = [
        AgentLG,
        AgentMR,
        CUHKAgent,
        IAMhaggler2012,
        MetaAgent,
        MetaAgent2012,
        OMACagent,
        TheNegotiatorReloaded,
    ]
    for cls in anac_2012:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2012"},
        )

    # Register ANAC 2013 agents
    anac_2013 = [
        AgentI,
        AgentKF,
        GAgent,
        InoxAgent,
        MetaAgent2013,
        SlavaAgent,
        TMFAgent,
        TheFawkes,
    ]
    for cls in anac_2013:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2013"},
        )

    # Register ANAC 2014 agents
    anac_2014 = [
        AgentM,
        AgentQuest,
        AgentTD,
        AgentTRP,
        AgentYK,
        AnacSampleAgent,
        ArisawaYaki,
        Aster,
        Atlas,
        BraveCat,
        DoNA,
        E2Agent,
        Flinch,
        Gangester,
        Gangster,
        KGAgent,
        Simpatico,
        Sobut,
        TUDelftGroup2,
        WhaleAgent,
    ]
    for cls in anac_2014:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2014"},
        )

    # Register ANAC 2015 agents
    anac_2015 = [
        AgentBuyog,
        AgentH,
        AgentHP,
        AgentNeo,
        AgentW,
        AgentX,
        AresParty,
        Atlas3,
        CUHKAgent2015,
        DrageKnight,
        Group2,
        JonnyBlack,
        Kawaii,
        MeanBot,
        Mercury,
        ParsAgent,
        PhoenixParty,
        PNegotiator,
        PokerFace,
        RandomDance,
        SENGOKU,
        TUDMixedStrategyAgent,
        XianFaAgent,
        Y2015Group2,
    ]
    for cls in anac_2015:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2015"},
        )

    # Register ANAC 2016 agents
    anac_2016 = [
        AgentHP2,
        AgentLight,
        AgentSmith2016,
        Atlas32016,
        Caduceus,
        ClockworkAgent,
        Farma,
        GrandmaAgent,
        MaxOops,
        MyAgent,
        Ngent,
        ParsAgent2,
        ParsCat,
        ParsCat2,
        SYAgent,
        Terra,
        YXAgent,
    ]
    for cls in anac_2016:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2016"},
        )

    # Register ANAC 2017 agents
    anac_2017 = [
        AgentF,
        AgentKN,
        CaduceusDC16,
        Farma17,
        Farma2017,
        GeneKing,
        Gin,
        Group3,
        Imitator,
        MadAgent,
        Mamenchis,
        Mosa,
        ParsAgent3,
        PonPokoAgent,
        Rubick,
        ShahAgent,
        SimpleAgent2017,
        TaxiBox,
        TucAgent,
    ]
    for cls in anac_2017:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2017"},
        )

    # Register ANAC 2018 agents
    anac_2018 = [
        Agent33,
        Agent36,
        AgentHerb,
        AgentNP1,
        AgreeableAgent2018,
        AteamAgent,
        Ateamagent,
        BetaOne,
        Betaone,
        ConDAgent,
        ExpRubick,
        FullAgent,
        GroupY,
        IQSun2018,
        Lancelot,
        Libra,
        MengWan,
        PonPokoRampage,
        Seto,
        Shiboy,
        SMACAgent,
        Sontag,
        Yeela,
    ]
    for cls in anac_2018:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2018"},
        )

    # Register ANAC 2019 agents
    anac_2019 = [
        AgentGG,
        AgentGP,
        AgentLarry,
        DandikAgent,
        EAgent,
        FSEGA2019,
        GaravelAgent,
        Gravity,
        Group1BOA,
        HardDealer,
        KAgent,
        KakeSoba,
        MINF,
        PodAgent,
        SACRA,
        SAGA,
        SolverAgent,
        TheNewDeal,
        WinkyAgent,
    ]
    for cls in anac_2019:
        negotiator_registry.register(
            cls,
            short_name=cls.__name__,
            source=NEGMAS_SOURCE,
            tags=genius_base | {"anac", "anac-2019"},
        )


def _register_scenarios() -> None:
    """Register all built-in scenarios shipped with negmas.

    These are negotiation scenarios (domain + utility files) that are included
    in the negmas package distribution.

    Scenario tags indicate their source and characteristics:
    - "builtin": Shipped with negmas
    - "xml": XML format (Genius)
    - "json": JSON format (GeniusWeb)
    - "yaml": YAML format
    - "bilateral": 2 negotiators
    - "multilateral": 3+ negotiators
    """
    # Get the path to the scenarios directory (shipped with negmas)
    scenarios_dir = Path(__file__).parent / "scenarios"

    if not scenarios_dir.exists():
        return

    # Register each scenario folder
    for scenario_path in scenarios_dir.iterdir():
        if scenario_path.is_dir() and not scenario_path.name.startswith("_"):
            # Determine number of negotiators by counting utility/profile files
            n_negotiators = _count_ufuns_in_scenario(scenario_path)

            # Determine if bilateral or multilateral based on ufun count
            extra_tags: set[str] = set()
            if n_negotiators == 2:
                extra_tags.add("bilateral")
            elif n_negotiators is not None and n_negotiators > 2:
                extra_tags.add("multilateral")

            scenario_registry.register(
                path=scenario_path,
                name=scenario_path.name,
                source=NEGMAS_SOURCE,
                tags={"builtin"} | extra_tags,
                n_negotiators=n_negotiators,
            )


def _count_ufuns_in_scenario(scenario_path: Path) -> int | None:
    """Count the number of utility function files in a scenario folder.

    Looks for profile/utility files in XML, JSON, and YAML formats.

    Args:
        scenario_path: Path to the scenario folder.

    Returns:
        The number of utility files found, or None if detection failed.
    """
    # Try XML format (Genius style): *prof*.xml, *profile*.xml, *util*.xml
    xml_profiles = (
        list(scenario_path.glob("*prof*.xml"))
        or list(scenario_path.glob("*profile*.xml"))
        or list(scenario_path.glob("*util*.xml"))
    )
    if xml_profiles:
        return len(xml_profiles)

    # Try JSON format (GeniusWeb style): *profile*.json, *util*.json
    json_profiles = list(scenario_path.glob("*profile*.json")) or list(
        scenario_path.glob("*util*.json")
    )
    if json_profiles:
        return len(json_profiles)

    # Try YAML format: *profile*.yaml, *profile*.yml, *util*.yaml, *util*.yml
    yaml_profiles = (
        list(scenario_path.glob("*profile*.yaml"))
        or list(scenario_path.glob("*profile*.yml"))
        or list(scenario_path.glob("*util*.yaml"))
        or list(scenario_path.glob("*util*.yml"))
    )
    if yaml_profiles:
        return len(yaml_profiles)

    return None


def _register_all() -> None:
    """Register all built-in classes in the registries."""
    _register_mechanisms()
    _register_sao_negotiators()
    _register_sao_components()
    _register_genius_boa_components()
    _register_genius_negotiators()
    _register_scenarios()


# Auto-register when this module is imported
_register_all()
