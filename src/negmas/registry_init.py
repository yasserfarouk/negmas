"""Initialization module that registers all built-in mechanisms, negotiators, and components.

This module is automatically imported when negmas.registry is imported,
ensuring all built-in classes are registered in the registries.

Note: Only concrete, usable classes are registered - not base classes.
"""

from __future__ import annotations

from negmas.registry import mechanism_registry, negotiator_registry, component_registry

__all__: list[str] = []


def _register_mechanisms() -> None:
    """Register all built-in mechanisms.

    Note: Only concrete mechanisms are registered, not base classes like Mechanism.
    """
    from negmas.sao.mechanism import SAOMechanism
    from negmas.gb.mechanisms.base import ParallelGBMechanism, SerialGBMechanism
    from negmas.gb.mechanisms.tau import TAUMechanism
    from negmas.gb.mechanisms.tauserial import SerialTAUMechanism
    from negmas.st import HillClimbingSTMechanism, VetoSTMechanism

    # Register SAO mechanisms
    mechanism_registry.register(
        SAOMechanism, short_name="SAOMechanism", requires_deadline=True
    )

    # Register GB mechanisms (concrete implementations)
    mechanism_registry.register(
        ParallelGBMechanism, short_name="ParallelGBMechanism", requires_deadline=True
    )
    mechanism_registry.register(
        SerialGBMechanism, short_name="SerialGBMechanism", requires_deadline=True
    )

    # Register TAU mechanisms - they don't require a deadline
    mechanism_registry.register(
        TAUMechanism, short_name="TAUMechanism", requires_deadline=False
    )
    mechanism_registry.register(
        SerialTAUMechanism, short_name="SerialTAUMechanism", requires_deadline=False
    )

    # Register ST mechanisms
    mechanism_registry.register(
        HillClimbingSTMechanism,
        short_name="HillClimbingSTMechanism",
        requires_deadline=True,
    )
    mechanism_registry.register(
        VetoSTMechanism, short_name="VetoSTMechanism", requires_deadline=True
    )


def _register_sao_negotiators() -> None:
    """Register SAO negotiators.

    Note: Base classes like SAONegotiator are not registered.
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

    # Aspiration negotiators
    negotiator_registry.register(
        AspirationNegotiator, short_name="AspirationNegotiator"
    )
    negotiator_registry.register(ToughNegotiator, short_name="ToughNegotiator")
    negotiator_registry.register(NiceNegotiator, short_name="NiceNegotiator")

    # Random negotiators
    negotiator_registry.register(RandomNegotiator, short_name="RandomNegotiator")
    negotiator_registry.register(
        RandomAlwaysAcceptingNegotiator, short_name="RandomAlwaysAcceptingNegotiator"
    )

    # Limited outcomes negotiators
    negotiator_registry.register(
        LimitedOutcomesNegotiator, short_name="LimitedOutcomesNegotiator"
    )
    negotiator_registry.register(
        LimitedOutcomesAcceptor, short_name="LimitedOutcomesAcceptor"
    )

    # Tit-for-tat negotiators
    negotiator_registry.register(
        NaiveTitForTatNegotiator, short_name="NaiveTitForTatNegotiator"
    )
    negotiator_registry.register(
        SimpleTitForTatNegotiator, short_name="SimpleTitForTatNegotiator"
    )

    # Time-based negotiators
    negotiator_registry.register(TimeBasedNegotiator, short_name="TimeBasedNegotiator")
    negotiator_registry.register(
        TimeBasedConcedingNegotiator, short_name="TimeBasedConcedingNegotiator"
    )
    negotiator_registry.register(
        BoulwareTBNegotiator, short_name="BoulwareTBNegotiator"
    )
    negotiator_registry.register(
        ConcederTBNegotiator, short_name="ConcederTBNegotiator"
    )
    negotiator_registry.register(LinearTBNegotiator, short_name="LinearTBNegotiator")

    # Utility-based negotiators
    negotiator_registry.register(UtilBasedNegotiator, short_name="UtilBasedNegotiator")

    # CAB family negotiators (Concede-to-Agreement-Based)
    negotiator_registry.register(CABNegotiator, short_name="CABNegotiator")
    negotiator_registry.register(CANNegotiator, short_name="CANNegotiator")
    negotiator_registry.register(CARNegotiator, short_name="CARNegotiator")

    # WAB family negotiators (Walk-Away-Based)
    negotiator_registry.register(WABNegotiator, short_name="WABNegotiator")
    negotiator_registry.register(WANNegotiator, short_name="WANNegotiator")
    negotiator_registry.register(WARNegotiator, short_name="WARNegotiator")

    # MiCRO negotiators
    negotiator_registry.register(MiCRONegotiator, short_name="MiCRONegotiator")
    negotiator_registry.register(FastMiCRONegotiator, short_name="FastMiCRONegotiator")


def _register_sao_components() -> None:
    """Register SAO components (acceptance policies, offering policies, models).

    Note: Base classes like AcceptancePolicy and OfferingPolicy are not registered.
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

    for cls in [
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
    ]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="acceptance"
        )

    # Import offering policies from sao.components.offering
    from negmas.sao.components.offering import (
        RandomOfferingPolicy,
        NoneOfferingPolicy,
        OfferBest,
        OfferTop,
        TimeBasedOfferingPolicy,
    )

    for cls in [
        RandomOfferingPolicy,
        NoneOfferingPolicy,
        OfferBest,
        OfferTop,
        TimeBasedOfferingPolicy,
    ]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="offering"
        )

    # Import models from sao.components.models
    from negmas.sao.components.models import (
        ZeroSumModel,
        FrequencyUFunModel,
        FrequencyLinearUFunModel,
    )

    for cls in [ZeroSumModel, FrequencyUFunModel, FrequencyLinearUFunModel]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="model"
        )


def _register_genius_boa_components() -> None:
    """Register Genius BOA components (transcompiled from Java).

    These are acceptance policies, offering policies, and opponent models
    from the Genius negotiation framework.
    """
    # Import Genius acceptance policies
    from negmas.gb.components.genius.acceptance import (
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
    )

    for cls in [
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
    ]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="acceptance"
        )

    # Import Genius offering policies
    from negmas.gb.components.genius.offering import (
        GTimeDependentOffering,
        GRandomOffering,
        GBoulwareOffering,
        GConcederOffering,
        GLinearOffering,
        GHardlinerOffering,
        GChoosingAllBids,
    )

    for cls in [
        GTimeDependentOffering,
        GRandomOffering,
        GBoulwareOffering,
        GConcederOffering,
        GLinearOffering,
        GHardlinerOffering,
        GChoosingAllBids,
    ]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="offering"
        )

    # Import Genius opponent models
    from negmas.gb.components.genius.models import (
        GHardHeadedFrequencyModel,
        GDefaultModel,
        GUniformModel,
        GOppositeModel,
        GSmithFrequencyModel,
        GAgentXFrequencyModel,
        GNashFrequencyModel,
        GBayesianModel,
        GScalableBayesianModel,
    )

    for cls in [
        GHardHeadedFrequencyModel,
        GDefaultModel,
        GUniformModel,
        GOppositeModel,
        GSmithFrequencyModel,
        GAgentXFrequencyModel,
        GNashFrequencyModel,
        GBayesianModel,
        GScalableBayesianModel,
    ]:
        component_registry.register(
            cls, short_name=cls.__name__, component_type="model"
        )


def _register_genius_negotiators() -> None:
    """Register Genius negotiators.

    Registers all Python wrappers for Genius Java negotiation agents.
    These are organized by ANAC competition year.
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
        negotiator_registry.register(cls, short_name=cls.__name__)

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
        negotiator_registry.register(cls, short_name=cls.__name__)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2010)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2011)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2012)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2013)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2014)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2015)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2016)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2017)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2018)

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
        negotiator_registry.register(cls, short_name=cls.__name__, anac_year=2019)


def _register_all() -> None:
    """Register all built-in classes in the registries."""
    _register_mechanisms()
    _register_sao_negotiators()
    _register_sao_components()
    _register_genius_boa_components()
    _register_genius_negotiators()


# Auto-register when this module is imported
_register_all()
