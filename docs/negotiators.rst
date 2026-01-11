Available Negotiators
=====================

NegMAS provides a rich set of negotiation agents for different mechanisms.
This page lists all available negotiators organized by category.

.. contents:: Table of Contents
   :local:
   :depth: 2

Native SAO Negotiators
----------------------

These negotiators work with the Stacked Alternating Offers (SAO) mechanism,
which is the most common bilateral negotiation protocol.

Base Classes
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``SAONegotiator``
     - Base class for all SAO negotiators (alias for SAOPRNegotiator)
   * - ``SAOPRNegotiator``
     - Base SAO negotiator with propose/respond interface
   * - ``SAOCallNegotiator``
     - SAO negotiator using callback functions

Time-Based Negotiators
~~~~~~~~~~~~~~~~~~~~~~

Negotiators that concede over time according to various curves.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``TimeBasedNegotiator``
     - Base time-based strategy independent of received offers
   * - ``TimeBasedConcedingNegotiator``
     - Time-based with configurable starting utility
   * - ``AspirationNegotiator``
     - Alternative interface to time-based conceding
   * - ``BoulwareTBNegotiator``
     - Concedes sub-linearly (tough early, concedes late)
   * - ``LinearTBNegotiator``
     - Concedes linearly over time
   * - ``ConcederTBNegotiator``
     - Concedes super-linearly (concedes early)
   * - ``FirstOfferOrientedTBNegotiator``
     - Orients offers toward partner's first offer
   * - ``LastOfferOrientedTBNegotiator``
     - Orients offers toward partner's last offer
   * - ``BestOfferOrientedTBNegotiator``
     - Orients offers toward partner's best offer
   * - ``AdditiveParetoFollowingTBNegotiator``
     - Additive weighted sum for Pareto-following
   * - ``MultiplicativeParetoFollowingTBNegotiator``
     - Multiplicative weighted selection
   * - ``AdditiveLastOfferFollowingTBNegotiator``
     - Additive with last offer filter
   * - ``MultiplicativeLastOfferFollowingTBNegotiator``
     - Multiplicative with last offer filter
   * - ``AdditiveFirstFollowingTBNegotiator``
     - Additive with first offer filter
   * - ``MultiplicativeFirstFollowingTBNegotiator``
     - Multiplicative with first offer filter

Rational Concession Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Negotiators based on the MiCRO (Monotonic Concession with Rational Outcomes) protocol.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``MiCRONegotiator``
     - Rational concession negotiator - concedes one outcome at a time
   * - ``FastMiCRONegotiator``
     - Faster version of MiCRO that may skip outcomes
   * - ``CABNegotiator``
     - Conceding Accepting Better (optimal, complete)
   * - ``CARNegotiator``
     - Conceding Accepting Rational
   * - ``CANNegotiator``
     - Conceding Accepting Not Worse (optimal, complete)
   * - ``WABNegotiator``
     - Wasting Accepting Better
   * - ``WARNegotiator``
     - Wasting Accepting Any (equilibrium)
   * - ``WANNegotiator``
     - Wasting Accepting Not Worse (equilibrium)

Tit-for-Tat Negotiators
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``NaiveTitForTatNegotiator``
     - Naive tit-for-tat without opponent model
   * - ``SimpleTitForTatNegotiator``
     - Alias for NaiveTitForTatNegotiator

Other Native Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``RandomNegotiator``
     - Responds randomly in negotiation
   * - ``RandomAlwaysAcceptingNegotiator``
     - Random with always-accept option
   * - ``NiceNegotiator``
     - Offers and accepts anything
   * - ``ToughNegotiator``
     - Accepts and proposes only the best outcome
   * - ``TopFractionNegotiator``
     - Offers and accepts only top fraction of outcomes
   * - ``LimitedOutcomesNegotiator``
     - Uses a fixed set of outcomes
   * - ``LimitedOutcomesAcceptor``
     - Uses a fixed set of outcomes for acceptance only
   * - ``UtilBasedNegotiator``
     - Base class for utility-based decisions
   * - ``HybridNegotiator``
     - Combines time-based and behavior-based strategies
   * - ``ControlledSAONegotiator``
     - Negotiator with external control interface

Native GB Negotiators
---------------------

These negotiators work with the General Bargaining (GB) mechanisms,
which support more complex multi-party and concurrent negotiations.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GBNegotiator``
     - Base class for all GB negotiators

GB negotiators share most implementations with SAO negotiators. The same
negotiator types (time-based, MiCRO, tit-for-tat, etc.) are available
with identical interfaces.

Genius Bridge Negotiators
-------------------------

NegMAS provides Python wrappers for 196 negotiation agents from the
`Genius <http://ii.tudelft.nl/genius/>`_ negotiation platform. These agents
participated in the Automated Negotiating Agents Competition (ANAC) from 2010-2019.

.. note::
   Genius negotiators require the Genius bridge to be running. See
   :doc:`tutorials/02.integrating_with_genius` for setup instructions.

Basic/Utility Agents (18)
~~~~~~~~~~~~~~~~~~~~~~~~~

Fundamental negotiation strategies and utility-based agents.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``TimeDependentAgentBoulware``
     - Boulware time-dependent strategy
   * - ``TimeDependentAgentConceder``
     - Conceder time-dependent strategy
   * - ``TimeDependentAgentHardliner``
     - Hardliner time-dependent strategy
   * - ``TimeDependentAgentLinear``
     - Linear time-dependent strategy
   * - ``BoulwareNegotiationParty``
     - Boulware negotiation party
   * - ``ConcederNegotiationParty``
     - Conceder negotiation party
   * - ``RandomParty``
     - Random negotiation party
   * - ``RandomParty2``
     - Alternative random party
   * - ``RandomCounterOfferNegotiationParty``
     - Random counter-offer strategy
   * - ``SimpleAgent``
     - Simple baseline agent
   * - ``BayesianAgent``
     - Bayesian learning agent
   * - ``SimilarityAgent``
     - Similarity-based agent
   * - ``ABMPAgent2``
     - ABMP strategy agent
   * - ``FuzzyAgent``
     - Fuzzy logic agent
   * - ``OptimalBidderSimple``
     - Simple optimal bidding
   * - ``FunctionalAcceptor``
     - Functional acceptance strategy
   * - ``ImmediateAcceptor``
     - Immediate acceptance strategy
   * - ``UtilityBasedAcceptor``
     - Utility-based acceptance

TU Delft Course Agents (23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agents developed in TU Delft negotiation courses.

``AI2014Group2``, ``Group1``, ``Group3Q2015``, ``Group4``, ``Group5``,
``Group6``, ``Group7``, ``Group8``, ``Group9``, ``Group10``, ``Group11``,
``Group12``, ``Group13``, ``Group14``, ``Group15``, ``Group16``, ``Group17``,
``Group18``, ``Group19``, ``Group20``, ``Group21``, ``Group22``, ``Q12015Group2``

ANAC 2010 Agents (8)
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AgentK``
     - ANAC 2010 winner
   * - ``Yushu``
     - ANAC 2010 finalist
   * - ``Nozomi``
     - ANAC 2010 finalist
   * - ``IAMhaggler``
     - Southampton's IAM haggler
   * - ``IAMcrazyHaggler``
     - Aggressive IAM variant
   * - ``AgentFSEGA``
     - FSEGA agent
   * - ``AgentSmith``
     - Agent Smith
   * - ``SouthamptonAgent``
     - Southampton's agent

ANAC 2011 Agents (9)
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``HardHeaded``
     - ANAC 2011 winner
   * - ``Gahboninho``
     - ANAC 2011 finalist
   * - ``IAMhaggler2011``
     - Updated IAM haggler
   * - ``AgentK2``
     - Updated AgentK
   * - ``TheNegotiator``
     - The Negotiator agent
   * - ``NiceTitForTat``
     - Nice tit-for-tat strategy
   * - ``ValueModelAgent``
     - Value model-based agent
   * - ``BramAgent``
     - Bram agent
   * - ``BramAgent2``
     - Bram agent variant

ANAC 2012 Agents (8)
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``CUHKAgent``
     - ANAC 2012 winner (CUHK)
   * - ``AgentLG``
     - ANAC 2012 finalist
   * - ``OMACagent``
     - OMAC agent
   * - ``TheNegotiatorReloaded``
     - Updated Negotiator
   * - ``AgentMR``
     - Agent MR
   * - ``IAMhaggler2012``
     - Updated IAM haggler
   * - ``MetaAgent``
     - Meta-learning agent
   * - ``MetaAgent2012``
     - Updated meta agent

ANAC 2013 Agents (8)
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``TheFawkes``
     - ANAC 2013 finalist
   * - ``MetaAgent2013``
     - Updated meta agent
   * - ``TMFAgent``
     - TMF agent
   * - ``AgentKF``
     - Agent KF
   * - ``InoxAgent``
     - Inox agent
   * - ``SlavaAgent``
     - Slava agent
   * - ``GAgent``
     - G agent
   * - ``AgentI``
     - Agent I

ANAC 2014 Agents (20)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AgentM``
     - ANAC 2014 finalist
   * - ``DoNA``
     - DoNA agent
   * - ``Gangster``
     - Gangster agent
   * - ``Gangester``
     - Gangster variant
   * - ``WhaleAgent``
     - Whale agent
   * - ``E2Agent``
     - E2 agent
   * - ``AgentYK``
     - Agent YK
   * - ``KGAgent``
     - KG agent
   * - ``BraveCat``
     - Brave Cat agent
   * - ``Atlas``
     - Atlas agent
   * - ``AgentQuest``
     - Agent Quest
   * - ``AgentTD``
     - Agent TD
   * - ``AgentTRP``
     - Agent TRP
   * - ``AnacSampleAgent``
     - ANAC sample agent
   * - ``ArisawaYaki``
     - Arisawa Yaki
   * - ``Aster``
     - Aster agent
   * - ``Flinch``
     - Flinch agent
   * - ``Simpatico``
     - Simpatico agent
   * - ``Sobut``
     - Sobut agent
   * - ``TUDelftGroup2``
     - TU Delft Group 2

ANAC 2015 Agents (24)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``Atlas3``
     - ANAC 2015 winner
   * - ``ParsAgent``
     - ANAC 2015 finalist
   * - ``RandomDance``
     - Random Dance agent
   * - ``Kawaii``
     - Kawaii agent
   * - ``AgentX``
     - Agent X
   * - ``PhoenixParty``
     - Phoenix Party
   * - ``PokerFace``
     - Poker Face agent
   * - ``CUHKAgent2015``
     - Updated CUHK agent
   * - ``DrageKnight``
     - Drage Knight
   * - ``JonnyBlack``
     - Jonny Black
   * - ``MeanBot``
     - Mean Bot
   * - ``Mercury``
     - Mercury agent
   * - ``PNegotiator``
     - P Negotiator
   * - ``SENGOKU``
     - Sengoku agent
   * - ``TUDMixedStrategyAgent``
     - TUD Mixed Strategy
   * - ``XianFaAgent``
     - Xian Fa agent
   * - ``AgentBuyog``
     - Agent Buyog
   * - ``AgentH``
     - Agent H
   * - ``AgentHP``
     - Agent HP
   * - ``AgentNeo``
     - Agent Neo
   * - ``AgentW``
     - Agent W
   * - ``AresParty``
     - Ares Party
   * - ``Group2``
     - Group 2
   * - ``Y2015Group2``
     - 2015 Group 2

ANAC 2016 Agents (17)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``Caduceus``
     - ANAC 2016 finalist
   * - ``YXAgent``
     - YX agent
   * - ``ParsCat``
     - Pars Cat agent
   * - ``ParsCat2``
     - Pars Cat variant
   * - ``Atlas32016``
     - Updated Atlas3
   * - ``Ngent``
     - Ngent agent
   * - ``AgentHP2``
     - Updated Agent HP
   * - ``AgentLight``
     - Agent Light
   * - ``AgentSmith2016``
     - Updated Agent Smith
   * - ``ClockworkAgent``
     - Clockwork agent
   * - ``Farma``
     - Farma agent
   * - ``GrandmaAgent``
     - Grandma agent
   * - ``MaxOops``
     - Max Oops
   * - ``MyAgent``
     - My Agent
   * - ``ParsAgent2``
     - Updated Pars Agent
   * - ``SYAgent``
     - SY agent
   * - ``Terra``
     - Terra agent

ANAC 2017 Agents (19)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``PonPokoAgent``
     - ANAC 2017 winner
   * - ``CaduceusDC16``
     - Caduceus DC16
   * - ``Rubick``
     - Rubick agent
   * - ``AgentKN``
     - Agent KN
   * - ``Farma17``
     - Farma 2017
   * - ``Farma2017``
     - Farma 2017 variant
   * - ``GeneKing``
     - Gene King
   * - ``Gin``
     - Gin agent
   * - ``Group3``
     - Group 3
   * - ``Imitator``
     - Imitator agent
   * - ``MadAgent``
     - Mad Agent
   * - ``Mamenchis``
     - Mamenchis agent
   * - ``Mosa``
     - Mosa agent
   * - ``ParsAgent3``
     - Updated Pars Agent
   * - ``ShahAgent``
     - Shah agent
   * - ``SimpleAgent2017``
     - Simple Agent 2017
   * - ``TaxiBox``
     - Taxi Box
   * - ``TucAgent``
     - Tuc agent
   * - ``AgentF``
     - Agent F

ANAC 2018 Agents (23)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AgentHerb``
     - ANAC 2018 finalist
   * - ``MengWan``
     - Meng Wan agent
   * - ``Yeela``
     - Yeela agent
   * - ``Sontag``
     - Sontag agent
   * - ``PonPokoRampage``
     - PonPoko Rampage
   * - ``Lancelot``
     - Lancelot agent
   * - ``AteamAgent``
     - A-Team agent
   * - ``Ateamagent``
     - A-Team variant
   * - ``BetaOne``
     - Beta One
   * - ``Betaone``
     - Beta One variant
   * - ``ConDAgent``
     - ConD agent
   * - ``ExpRubick``
     - Exp Rubick
   * - ``FullAgent``
     - Full agent
   * - ``GroupY``
     - Group Y
   * - ``IQSun2018``
     - IQ Sun 2018
   * - ``Libra``
     - Libra agent
   * - ``Seto``
     - Seto agent
   * - ``Shiboy``
     - Shiboy agent
   * - ``SMACAgent``
     - SMAC agent
   * - ``Agent33``
     - Agent 33
   * - ``Agent36``
     - Agent 36
   * - ``AgentNP1``
     - Agent NP1
   * - ``AgreeableAgent2018``
     - Agreeable Agent

ANAC 2019 Agents (19)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AgentGG``
     - ANAC 2019 finalist
   * - ``KakeSoba``
     - Kake Soba agent
   * - ``SAGA``
     - SAGA agent
   * - ``WinkyAgent``
     - Winky agent
   * - ``AgentGP``
     - Agent GP
   * - ``AgentLarry``
     - Agent Larry
   * - ``DandikAgent``
     - Dandik agent
   * - ``EAgent``
     - E agent
   * - ``FSEGA2019``
     - FSEGA 2019
   * - ``GaravelAgent``
     - Garavel agent
   * - ``Gravity``
     - Gravity agent
   * - ``Group1BOA``
     - Group 1 BOA
   * - ``HardDealer``
     - Hard Dealer
   * - ``KAgent``
     - K agent
   * - ``MINF``
     - MINF agent
   * - ``PodAgent``
     - Pod agent
   * - ``SACRA``
     - SACRA agent
   * - ``SolverAgent``
     - Solver agent
   * - ``TheNewDeal``
     - The New Deal

Usage Examples
--------------

Using Native Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from negmas.sao import SAOMechanism
   from negmas.sao.negotiators import (
       AspirationNegotiator,
       BoulwareTBNegotiator,
       MiCRONegotiator,
   )
   from negmas.preferences import LinearAdditiveUtilityFunction as U
   from negmas.outcomes import make_issue

   # Create a simple negotiation scenario
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]

   # Create negotiators with utility functions
   negotiator1 = AspirationNegotiator(name="buyer")
   negotiator2 = BoulwareTBNegotiator(name="seller")

   # Run negotiation
   mechanism = SAOMechanism(issues=issues, n_steps=100)
   mechanism.add(negotiator1, ufun=U.random(issues))
   mechanism.add(negotiator2, ufun=U.random(issues))
   mechanism.run()

Using Genius Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from negmas.genius import GeniusBridge
   from negmas.genius.gnegotiators import Atlas3, AgentK
   from negmas.sao import SAOMechanism

   # Start the Genius bridge (required)
   GeniusBridge.start()

   # Create Genius negotiators
   negotiator1 = Atlas3()
   negotiator2 = AgentK()

   # Run negotiation (same as native negotiators)
   mechanism = SAOMechanism(issues=issues, n_steps=100)
   mechanism.add(negotiator1, ufun=ufun1)
   mechanism.add(negotiator2, ufun=ufun2)
   mechanism.run()

See Also
--------

- :doc:`tutorials/01.running_simple_negotiation` - Basic negotiation tutorial
- :doc:`tutorials/02.integrating_with_genius` - Genius integration guide
- :doc:`tutorials/03.develop_new_negotiator` - Creating custom negotiators
- :doc:`components` - Negotiation components (acceptance, offering strategies)
- :doc:`modules/sao` - SAO mechanism API reference
- :doc:`modules/gb` - GB mechanism API reference
