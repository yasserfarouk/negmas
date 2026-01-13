ANAC Competition
================

The Automated Negotiating Agents Competition (ANAC) is an international competition
that has been running since 2010 to bring together researchers from the negotiation
community. The competition challenges participants to design autonomous negotiation
agents that can effectively negotiate with other agents in various scenarios.

NegMAS provides comprehensive support for ANAC competitions through the ``GENIUS_INFO``
(also available as ``ANAC_INFO``) dictionary, which contains metadata about all ANAC
competition years from 2010 to present.

.. note::

    The competition was originally called the **Genius Negotiation League** and ran on the
    Java-based `Genius platform <http://ii.tudelft.nl/genius/>`_. Starting in 2019, some
    leagues began using the GeniusWeb platform, and from 2024, the **Automated Negotiation
    League (ANL)** runs natively on NegMAS.


Accessing Competition Data
--------------------------

You can access information about any competition year using the ``GENIUS_INFO`` or
``ANAC_INFO`` dictionaries:

.. code-block:: python

    from negmas.genius.ginfo import GENIUS_INFO, ANAC_INFO, get_anac_agents

    # ANAC_INFO is an alias for GENIUS_INFO
    assert ANAC_INFO is GENIUS_INFO

    # Get information about a specific year
    info_2024 = GENIUS_INFO[2024]
    print(info_2024["winners"])  # List of winners
    print(info_2024["finalists"])  # List of finalists
    print(info_2024["participants"])  # All participants

    # Use helper function to get agents with specific criteria
    winners_2024 = get_anac_agents(year=2024, winners_only=True)
    bilateral_agents = get_anac_agents(bilateral=True)


Agent Availability
------------------

Agents from different competition years are available through different libraries:

**Java/Genius Agents (2010-2018)**
    These agents were written in Java for the Genius platform. To use them, you need:

    1. **GeniusBridge**: A Java bridge that must be running to communicate with Java agents.
       See the :doc:`tutorials/02.integrating_with_genius` tutorial for setup instructions.

    2. **negmas-genius-agents** (optional): A library that re-implements many of these agents
       in pure Python using AI-based conversion. This allows running them without the Java bridge.
       Install with: ``pip install negmas-genius-agents``

**GeniusWeb Agents (2019-2023)**
    These agents were written for the GeniusWeb platform (Python/Java hybrid). To use them:

    - Install the `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_:
      ``pip install negmas-geniusweb-bridge``

**ANL Agents (2024+)**
    These agents are written in pure Python for NegMAS. To use them:

    - Install the `anl-agents <https://github.com/yasserfarouk/anl-agents>`_ package:
      ``pip install anl-agents``


Competition Years
-----------------

Below is a detailed summary of each competition year, including the settings, winners,
and agent availability.

2010
^^^^

**Settings**: Bilateral, Linear utilities, Discounting

**Winners**:

1. AgentK (``agents.anac.y2010.AgentK.Agent_K``)
2. Yushu (``agents.anac.y2010.Yushu.Yushu``)
3. Nozomi (``agents.anac.y2010.Nozomi.Nozomi``)
4. IAMhaggler (``agents.anac.y2010.Southampton.IAMhaggler``)

**Participants** (7 agents): AgentFSEGA, AgentK, AgentSmith, Nozomi, IAMcrazyHaggler, IAMhaggler, Yushu

**Agent Access**: Java/Genius (requires GeniusBridge)


2011
^^^^

**Settings**: Bilateral, Linear utilities, Discounting

**Winners**:

1. HardHeaded (``agents.anac.y2011.HardHeaded.KLH``)
2. Gahboninho (``agents.anac.y2011.Gahboninho.Gahboninho``)
3. IAMhaggler2011 (``agents.anac.y2011.IAMhaggler2011.IAMhaggler2011``)

**Finalists** (8 agents): HardHeaded, Gahboninho, IAMhaggler2011, BramAgent, AgentK2, TheNegotiator, Nice Tit-for-Tat, ValueModelAgent

**Agent Access**: Java/Genius (requires GeniusBridge)


2012
^^^^

**Settings**: Bilateral, Linear utilities, Reservation value, Discounting

**Winners**:

1. CUHKAgent (``agents.anac.y2012.CUHKAgent.CUHKAgent``)
2. AgentLG (``agents.anac.y2012.AgentLG.AgentLG``)
3. OMACagent (``agents.anac.y2012.OMACagent.OMACagent``)

**Finalists** (8 agents): CUHKAgent, AgentLG, OMACagent, AgentMR, TheNegotiatorReloaded, BRAMAgent2, MetaAgent, IAMhaggler2012

**Agent Access**: Java/Genius (requires GeniusBridge)


2013
^^^^

**Settings**: Bilateral, Linear utilities, Learning, Reservation value, Discounting

**Winners**:

1. TheFawkes (``agents.anac.y2013.TheFawkes.TheFawkes``)
2. MetaAgent2013 (``agents.anac.y2013.MetaAgent.MetaAgent2013``)
3. TMFAgent (``agents.anac.y2013.TMFAgent.TMFAgent``)

**Finalists** (7 agents): TheFawkes, MetaAgent2013, TMFAgent, AgentKF, InoxAgent, Slavaagent, GAgent

**Agent Access**: Java/Genius (requires GeniusBridge)


2014
^^^^

**Settings**: Bilateral, Non-linear utilities, Reservation value

**Winners**:

1. AgentM (``agents.anac.y2014.AgentM.AgentM``)
2. DoNA (``agents.anac.y2014.DoNA.DoNA``)
3. Gangster (``agents.anac.y2014.Gangster.Gangster``)

**Finalists** (9 agents): AgentM, DoNA, Gangster, AgentYK, BraveCat, E2Agent, Atlas3, AgentTRP, WhaleAgent

**Participants** (18 agents): Full list in ``GENIUS_INFO[2014]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge)


2015
^^^^

**Settings**: Multilateral, Linear utilities, Reservation value

**Winners**:

1. Atlas3 (``agents.anac.y2015.Atlas3.Atlas3``)
2. ParsAgent (``agents.anac.y2015.ParsAgent.ParsAgent``)
3. RandomDance (``agents.anac.y2015.RandomDance.RandomDance``)

**Finalists** (8 agents): Atlas3, ParsAgent, RandomDance, AgentX, Kawaii, Mercury, PhoenixParty, PokerFace

**Participants** (24 agents): Full list in ``GENIUS_INFO[2015]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge)


2016
^^^^

**Settings**: Multilateral, Linear utilities, Reservation value

**Winners**:

1. Caduceus (``agents.anac.y2016.caduceus.Caduceus``)
2. YXAgent (``agents.anac.y2016.yxagent.YXAgent``)

**Finalists** (10 agents): Caduceus, YXAgent, ParsCat, Farma, Atlas32016, MyAgent, Ngent, GrandmaAgent, AgentHP2, Terra

**Participants** (16 agents): Full list in ``GENIUS_INFO[2016]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge)


2017
^^^^

**Settings**: Multilateral, Linear utilities, Learning, Reservation value

**Winners**:

1. PonPokoAgent (``agents.anac.y2017.ponpokoagent.PonPokoAgent``)
2. CaduceusDC16 (``agents.anac.y2017.caduceus.CaduceusDC16``)
3. Rubick (``agents.anac.y2017.rubick.Rubick``)

**Finalists** (10 agents): PonPokoAgent, CaduceusDC16, Rubick, AgentF, AgentKN, GeneKing, Mamenchis, ParsCat2, TucAgent, SimpleAgent2017

**Participants** (18 agents): Full list in ``GENIUS_INFO[2017]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge)


2018
^^^^

**Settings**: Multilateral, Linear utilities, Learning, Reservation value, Discounting

**Winners**:

1. MengWan (``agents.anac.y2018.meng_wan.Agent36``)
2. BetaOne (``agents.anac.y2018.beta_one.Group2``)
3. AgentHerb (``agents.anac.y2018.agentherb.AgentHerb``)

**Finalists** (13 agents): MengWan, BetaOne, AgentHerb, ConDAgent, ExpRubick, FullAgent, IQSun2018, Libra, PonPokoRampage, Seto, Shiboy, SMAC_Agent, Yeela

**Participants** (20 agents): Full list in ``GENIUS_INFO[2018]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge)


2019
^^^^

**Settings**: Bilateral, Linear utilities, Reservation value, Uncertainty

**Platform**: GeniusWeb

**Winners**:

1. AgentGG (``agents.anac.y2019.agentgg.AgentGG``)
2. KakeSoba (``agents.anac.y2019.kakesoba.KakeSoba``)
3. SAGA (``agents.anac.y2019.saga.SAGA``)

**Finalists** (6 agents): AgentGG, KakeSoba, SAGA, DandikAgent, GaravelAgent, MINF

**Participants** (18 agents): Full list in ``GENIUS_INFO[2019]["participants"]``

**Agent Access**: Java/Genius (requires GeniusBridge) or `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_


2020
^^^^

**Settings**: Linear utilities, Reservation value, Uncertainty, Elicitation

**Platform**: GeniusWeb (Java)

**Finalists** (5 agents, alphabetically - no explicit ranking published):

- AgentKT (Brown University, USA)
- AhBuNeAgent (Ozyegin University, Turkey)
- Angel (University of Tulsa, USA)
- HammingAgent (TUAT, Japan)
- ShineAgent (Bar-Ilan University, Israel)

**Participants** (13 agents): AgentKT, AgentP1DAMO, AgentXX, AhBuNeAgent, Anaconda, Angel, AzarAgent, BlingBling, DUOAgent, ForArisa, HammingAgent, NiceAgent, ShineAgent

**Agent Access**: `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_


2021
^^^^

**Settings**: Linear utilities, Learning, Reservation value, Uncertainty, Elicitation

**Platform**: GeniusWeb (Java)

**Winners**:

1. **AlphaBIU** (Bar-Ilan University, Israel)
2. **MatrixAlienAgent** (University of Tulsa, USA)
3. **TripleAgent** (Utrecht University, Netherlands)

**Agent Access**: `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_

.. note::

    The 2021 agents in the bridge package are not yet fully translated from Java.


2022
^^^^

**Settings**: Linear utilities, Learning, Reservation value, Elicitation

**Platform**: GeniusWeb (Python)

**Winners (Individual Utility)**:

1. **DreamTeam109Agent** (College of Management Academic Studies, Israel)
2. **ChargingBoul** (University of Tulsa, USA)

**Winners (Social Welfare)**:

1. **DreamTeam109Agent** (College of Management Academic Studies, Israel)
2. **Agent007** (Bar-Ilan University, Israel)

**Finalists** (3 agents): Agent007, ChargingBoul, DreamTeam109Agent

**Participants** (19 agents): Agent007, Agent4410, AgentFish, AgentFO2, BIU_agent, ChargingBoul, CompromisingAgent, DreamTeam109Agent, GEAAgent, LearningAgent, LuckyAgent2022, MiCROAgent, Pinar_Agent, ProcrastinAgent, RGAgent, SmartAgent, SuperAgent, ThirdAgent, Tjaronchery10Agent

**Agent Access**: `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_


2023
^^^^

**Settings**: Linear utilities, Learning, Reservation value, Elicitation

**Platform**: GeniusWeb (Python)

**Winners (Individual Utility)**:

1. **ExploitAgent** - Bram Renting (Leiden University, Delft University of Technology)
2. **MiCRO2023** - Dave de Jonge (IIIA-CSIC)

**Winners (Social Welfare)**:

1. **AntHeartAgent** - Kaiyou Lei et al. (Ant Group)
2. **SmartAgent** - Jianing Zhao et al. (Southwest University)

**Finalists** (4 agents): ExploitAgent, MiCRO2023, AntHeartAgent, SmartAgent

**Participants** (15 agents): AgentFO3, AmbitiousAgent, AntAllianceAgent, AntHeartAgent, ColmanAnacondotAgent2, ExploitAgent, GotAgent, HybridAgent2023, KBTimeDiffAgent, MiCRO2023, MSCAgent, PopularAgent, SmartAgent, SpaghettiAgent, TripleEAgent

**Agent Access**: `negmas-geniusweb-bridge <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_


2024 (ANL)
^^^^^^^^^^

**Settings**: Bilateral, Linear utilities, Reservation value

**Platform**: NegMAS (Native Python)

**Winners (Individual Advantage)**:

1. **Shochan** - Takayama, TUAT, Japan (``anl_agents.anl2024.takafam.Shochan``)
2. **UOAgent** - Hirotada Matsumoto, TUAT, Japan (``anl_agents.anl2024.team_moto.UOAgent``)
3. **AgentRenting2024** - Mick Elshout, Utrecht University, Netherlands (``anl_agents.anl2024.team_renting.AgentRenting2024``)

**Winner (Nash Optimality)**:

1. **Shochan** - Takayama, TUAT, Japan

**Finalists** (10 agents): Shochan, UOAgent, AgentRenting2024, AntiAgent, HardChaosNegotiator, KosAgent, Nayesian2, CARCAgent, BidBot, AgentNyan

**Participants** (19 agents): Full list in ``GENIUS_INFO[2024]["participants"]``

**Agent Access**: `anl-agents <https://github.com/yasserfarouk/anl-agents>`_ (``pip install anl-agents``)

**More Information**: `ANL 2024 Results <https://scml.cs.brown.edu/anl2024>`_


2025 (ANL)
^^^^^^^^^^

**Settings**: Bilateral, Linear utilities, Reservation value

**Platform**: NegMAS (Native Python)

**Winners**:

1. **RUFL** - Garrett Seo, Rutgers University, USA (``anl_agents.anl2025.team_271.RUFL``) - 250€
1. **SAC Agent** - Hossein Savari, University of Tehran, Iran (``anl_agents.anl2025.university_of_tehran.SacAgent``) - 250€ (tied for 1st)
3. **UFunAtAgent** - Fukutoku Yuma, TUAT, Japan (``anl_agents.anl2025.team_305.UfunATAgent``) - 100€

**Finalists** (12 agents, by qualification score):

1. SacAgent (University of Tehran)
2. ProbaBot (CWI, Netherlands)
3. RUFL (Rutgers University)
4. KDY (TUAT)
5. JeemNegotiator (Bar-Ilan University)
6. Astrat3m (Chongqing Jiaotong University)
7. A4E (TUAT)
8. OzUAgent (Ozyegin University)
9. SmartNegotiator (Bar-Ilan University)
10. CARC2025 (HIT Shenzhen)
11. UfunATAgent (TUAT)
12. Wagent (Chongqing Jiaotong University)

**Participants** (17 agents): Full list in ``GENIUS_INFO[2025]["participants"]``

**Agent Access**: `anl-agents <https://github.com/yasserfarouk/anl-agents>`_ (``pip install anl-agents``)

**More Information**: `ANL 2025 Results <https://scml.cs.brown.edu/anl2025>`_


Competition Settings Reference
------------------------------

The following table summarizes the settings for each competition year:

.. list-table::
   :header-rows: 1
   :widths: 10 15 10 10 10 10 10 10 15

   * - Year
     - Type
     - Linear
     - Learning
     - Reservation
     - Discount
     - Uncertainty
     - Elicitation
     - Platform
   * - 2010
     - Bilateral
     - Yes
     - No
     - No
     - Yes
     - No
     - No
     - Genius
   * - 2011
     - Bilateral
     - Yes
     - No
     - No
     - Yes
     - No
     - No
     - Genius
   * - 2012
     - Bilateral
     - Yes
     - No
     - Yes
     - Yes
     - No
     - No
     - Genius
   * - 2013
     - Bilateral
     - Yes
     - Yes
     - Yes
     - Yes
     - No
     - No
     - Genius
   * - 2014
     - Bilateral
     - No
     - No
     - Yes
     - No
     - No
     - No
     - Genius
   * - 2015
     - Multilateral
     - Yes
     - No
     - Yes
     - No
     - No
     - No
     - Genius
   * - 2016
     - Multilateral
     - Yes
     - No
     - Yes
     - No
     - No
     - No
     - Genius
   * - 2017
     - Multilateral
     - Yes
     - Yes
     - Yes
     - No
     - No
     - No
     - Genius
   * - 2018
     - Multilateral
     - Yes
     - Yes
     - Yes
     - Yes
     - No
     - No
     - Genius
   * - 2019
     - Bilateral
     - Yes
     - No
     - Yes
     - No
     - Yes
     - No
     - GeniusWeb
   * - 2020
     - Mixed
     - Yes
     - No
     - Yes
     - No
     - Yes
     - Yes
     - GeniusWeb
   * - 2021
     - Mixed
     - Yes
     - Yes
     - Yes
     - No
     - Yes
     - Yes
     - GeniusWeb
   * - 2022
     - Mixed
     - Yes
     - Yes
     - Yes
     - No
     - No
     - Yes
     - GeniusWeb
   * - 2023
     - Mixed
     - Yes
     - Yes
     - Yes
     - No
     - No
     - Yes
     - GeniusWeb
   * - 2024
     - Bilateral
     - Yes
     - No
     - Yes
     - No
     - No
     - No
     - NegMAS
   * - 2025
     - Bilateral
     - Yes
     - No
     - Yes
     - No
     - No
     - No
     - NegMAS


External Resources
------------------

- `ANAC Official Website <https://web.tuat.ac.jp/~katfuji/ANAC2024/>`_
- `ANL Competition Portal <https://scml.cs.brown.edu/anl>`_
- `Genius Platform <http://ii.tudelft.nl/genius/>`_
- `GeniusWeb Platform <https://tracinsy.ewi.tudelft.nl/pubtrac/GeniusWeb>`_
- `anl-agents Repository <https://github.com/yasserfarouk/anl-agents>`_
- `negmas-geniusweb-bridge Repository <https://github.com/yasserfarouk/negmas-geniusweb-bridge>`_
