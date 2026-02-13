Available Negotiators
=====================

NegMAS provides a rich set of negotiation agents for different mechanisms.
This page lists all available negotiators organized by category.

Negotiation Callback Lifecycle
------------------------------

When a negotiator participates in a negotiation, the mechanism calls various
callbacks in a **guaranteed order**. Understanding this order is essential for
implementing custom negotiators.

Callback Order Flowchart
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      NEGOTIATION LIFECYCLE                              │
    └─────────────────────────────────────────────────────────────────────────┘

    1. JOINING PHASE (when negotiator.join() or mechanism.add() is called)
       ├── join(nmi, state, preferences, role)
       │   └── Returns True/False to accept/reject joining
       └── [preferences assigned internally, owner NOT set yet]

    2. NEGOTIATION START (when mechanism.step() or mechanism.run() is called)
       │
       │   ┌─────────────────────────────────────────────────────────────────┐
       │   │  IMPORTANT: The following callbacks are called ONCE per         │
       │   │  negotiation, in this EXACT order, regardless of when           │
       │   │  preferences were assigned (at constructor or at join time).    │
       │   └─────────────────────────────────────────────────────────────────┘
       │
       ├── [owner set on preferences]
       ├── on_preferences_changed([Initialization])  ◄── ALWAYS FIRST (if preferences exist)
       ├── on_negotiation_start(state)               ◄── ALWAYS SECOND
       └── on_round_start(state)                     ◄── First round starts

    3. NEGOTIATION ROUNDS (repeated until agreement, timeout, or break)
       │
       ├── propose(state) → Outcome | None
       │   └── Called when it's negotiator's turn to make an offer
       │
       ├── respond(state, offer, source) → ResponseType
       │   └── Called when evaluating an offer from another negotiator
       │
       ├── on_partner_proposal(state, partner_id, offer)
       │   └── Notification when a partner makes a proposal
       │
       ├── on_partner_response(state, partner_id, outcome, response)
       │   └── Notification when a partner responds to an offer
       │
       ├── on_round_end(state)
       │   └── Called at the end of each round
       │
       └── on_round_start(state)  [if more rounds remain]
           └── Called at the start of each new round

    4. NEGOTIATION END
       │
       ├── on_negotiation_end(state)
       │   └── Called when negotiation concludes (agreement, timeout, or break)
       │
       └── on_leave(state)
           └── Called when negotiator leaves the mechanism
           └── [owner cleared from preferences]
           └── on_preferences_changed([Dissociated])  ◄── Notifies of disconnection


Key Guarantees
~~~~~~~~~~~~~~

1. **Initialization Order**: ``on_preferences_changed([Initialization])`` is
   **always** called before ``on_negotiation_start()``, regardless of when
   preferences were set.

2. **Exactly Once**: Both ``on_preferences_changed([Initialization])`` and
   ``on_negotiation_start()`` are called **exactly once** per negotiation.

3. **Before Proposals**: These initialization callbacks always occur before
   any ``propose()`` or ``respond()`` calls.

4. **Owner Lifecycle**: The preferences ``owner`` is set just before
   ``on_preferences_changed([Initialization])`` and cleared in ``on_leave()``.

5. **Dissociation Notification**: When a negotiator leaves, it
   receive an ``on_preferences_changed([Dissociated])`` notification.

Example: Tracking Callback Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from negmas.sao import SAOMechanism, SAONegotiator, ResponseType
    from negmas.common import PreferencesChangeType


    class CallbackTracker(SAONegotiator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.callback_log = []

        def on_preferences_changed(self, changes):
            change_types = [c.type.name for c in changes]
            self.callback_log.append(f"on_preferences_changed({change_types})")
            super().on_preferences_changed(changes)

        def on_negotiation_start(self, state):
            self.callback_log.append("on_negotiation_start")
            super().on_negotiation_start(state)

        def on_round_start(self, state):
            self.callback_log.append(f"on_round_start(step={state.step})")
            super().on_round_start(state)

        def propose(self, state, dest=None):
            self.callback_log.append(f"propose(step={state.step})")
            return self.nmi.random_outcome()

        def respond(self, state, source=None):
            self.callback_log.append(f"respond(step={state.step})")
            return ResponseType.REJECT_OFFER


    # Run a negotiation and inspect the callback order
    tracker = CallbackTracker()
    mechanism = SAOMechanism(issues=[...], n_steps=3)
    mechanism.add(tracker, ufun=...)
    mechanism.add(other_negotiator, ufun=...)
    mechanism.run()

    # Output shows guaranteed order:
    # ['on_preferences_changed([Initialization])',
    #  'on_negotiation_start',
    #  'on_round_start(step=0)',
    #  'propose(step=0)',
    #  'respond(step=0)',
    #  ...]

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

Python-Native Genius Negotiators
--------------------------------

NegMAS provides Python-native implementations of classic Genius negotiation agents.
These implementations use transcompiled Genius BOA (Bidding-Opponent modeling-Acceptance)
components and **do NOT require the Java Genius bridge**.

The naming convention uses a ``G`` prefix to distinguish these from Java-bridge versions.

.. note::
   These negotiators are fully implemented in Python and can be used without
   any external Java dependencies. They are recommended for most use cases
   where Genius-style strategies are needed.

Classic Time-Dependent Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GBoulware``
     - Boulware strategy (e=0.2): concedes slowly, tough early
   * - ``GConceder``
     - Conceder strategy (e=2.0): concedes quickly early
   * - ``GLinear``
     - Linear strategy (e=1.0): constant concession rate
   * - ``GHardliner``
     - Hardliner strategy (e=0): never concedes

ANAC Competition Agents
~~~~~~~~~~~~~~~~~~~~~~~

Python-native implementations of notable ANAC competition agents.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GHardHeaded``
     - ANAC 2011 Winner: Boulware offering with frequency-based opponent modeling
   * - ``GAgentK``
     - ANAC 2010: Time-dependent with combined acceptance conditions
   * - ``GAgentSmith``
     - ANAC 2010: Time-dependent with Smith-style frequency model
   * - ``GNozomi``
     - ANAC 2010: Boulware with previous-offer acceptance
   * - ``GFSEGA``
     - ANAC 2010: Conceder with constant threshold acceptance
   * - ``GCUHKAgent``
     - ANAC 2012 Winner: Conservative time-dependent with frequency modeling
   * - ``GAgentLG``
     - ANAC 2012: Time-dependent with CombiMax acceptance
   * - ``GAgentX``
     - ANAC 2015: Adaptive with window-based acceptance and exponential smoothing

Utility Agents
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GRandom``
     - Random offers with always-accept policy (baseline/testing)

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
   mechanism.add(negotiator1, ufun=U.random(issues=issues))
   mechanism.add(negotiator2, ufun=U.random(issues=issues))
   mechanism.run()

Using Python-Native Genius Negotiators (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These negotiators don't require Java or the Genius bridge.

.. code-block:: python

   from negmas.genius import GHardHeaded, GCUHKAgent, GBoulware
   from negmas.sao import SAOMechanism
   from negmas.preferences import LinearAdditiveUtilityFunction as U
   from negmas.outcomes import make_issue

   # Create a simple negotiation scenario
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]

   # Create Python-native Genius negotiators - NO bridge needed!
   negotiator1 = GHardHeaded(name="hardheaded")  # ANAC 2011 winner
   negotiator2 = GCUHKAgent(name="cuhk")  # ANAC 2012 winner

   # Run negotiation
   mechanism = SAOMechanism(issues=issues, n_steps=100)
   mechanism.add(negotiator1, ufun=U.random(issues=issues))
   mechanism.add(negotiator2, ufun=U.random(issues=issues))
   mechanism.run()

Using Genius Bridge Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For access to all 196 Genius agents (requires Java).

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

Creating Custom Negotiators
---------------------------

NegMAS offers two approaches to creating custom negotiators:

1. **Inheritance**: Subclass a base negotiator and override methods
2. **Composition**: Combine multiple negotiators using ``SAOAggMetaNegotiator``

Inheritance (Traditional Approach)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass a base negotiator and implement the required methods:

.. code-block:: python

   from negmas.sao import SAOMechanism, SAONegotiator, ResponseType
   from negmas.preferences import LinearAdditiveUtilityFunction as U
   from negmas.outcomes import make_issue, make_os


   class MyNegotiator(SAONegotiator):
       """A simple negotiator using inheritance."""

       def propose(self, state, dest=None):
           # Propose a random outcome
           return self.nmi.random_outcome()

       def respond(self, state, source=None):
           offer = state.current_offer
           # Accept any offer with utility > 0.8
           if offer is not None and self.ufun(offer) > 0.8:
               return ResponseType.ACCEPT_OFFER
           return ResponseType.REJECT_OFFER


   # Use the custom negotiator
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]
   os = make_os(issues)
   session = SAOMechanism(issues=issues, n_steps=100)
   session.add(MyNegotiator(name="custom"), ufun=U.random(os, reserved_value=0.0))
   session.add(MyNegotiator(name="opponent"), ufun=U.random(os, reserved_value=0.0))
   session.run()

Composition (Ensemble Approach)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SAOAggMetaNegotiator`` to combine multiple negotiators and aggregate their decisions.
This is useful for ensemble strategies, voting mechanisms, or dynamic strategy switching.

.. code-block:: python

   from negmas.sao import SAOMechanism, ResponseType
   from negmas.sao.negotiators import (
       SAOAggMetaNegotiator,
       BoulwareTBNegotiator,
       NaiveTitForTatNegotiator,
       AspirationNegotiator,
   )
   from negmas.preferences import LinearAdditiveUtilityFunction as U
   from negmas.outcomes import make_issue, make_os


   class MajorityVoteNegotiator(SAOAggMetaNegotiator):
       """An ensemble negotiator that uses majority voting."""

       def aggregate_proposals(self, state, proposals, dest=None):
           # Use the proposal from the first negotiator that offers something
           for neg, proposal in proposals:
               if proposal is not None:
                   return proposal
           return None

       def aggregate_responses(self, state, responses, offer, source=None):
           # Majority vote: accept if more than half accept
           accept_count = sum(1 for _, r in responses if r == ResponseType.ACCEPT_OFFER)
           if accept_count > len(responses) / 2:
               return ResponseType.ACCEPT_OFFER
           return ResponseType.REJECT_OFFER


   # Create an ensemble of different strategies
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]
   os = make_os(issues)
   ufun = U.random(os, reserved_value=0.0)

   ensemble = MajorityVoteNegotiator(
       negotiators=[
           BoulwareTBNegotiator(),  # Tough strategy
           NaiveTitForTatNegotiator(),  # Reactive strategy
           BoulwareTBNegotiator(),  # Another tough vote
       ],
       name="ensemble",
   )

   # Use in a negotiation
   session = SAOMechanism(issues=issues, n_steps=100)
   session.add(ensemble, ufun=ufun)
   session.add(
       AspirationNegotiator(name="opponent"), ufun=U.random(os, reserved_value=0.0)
   )
   session.run()

The ensemble approach is useful for:

- **Voting strategies**: Combine multiple negotiators via majority/weighted voting
- **Dynamic delegation**: Switch between strategies at runtime
- **A/B testing**: Compare strategies within the same negotiation

.. note::
   **Important Consideration**: Sub-negotiators within an ensemble may behave differently
   than they would in isolation. Their recommended offers and decisions are not guaranteed
   to be executed in the actual negotiation, which may break implicit assumptions they make
   about the negotiation flow. For example, a negotiator that expects its offers to be sent
   in sequence may not work as intended if the ensemble aggregator selects offers from
   different sub-negotiators.

.. note::
   ``GBMetaNegotiator`` is also available for GB (General Bargaining) protocols
   with additional callbacks for partner events.

Composition (BOA Components)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``BOANegotiator`` to build negotiators from reusable components following
the Bidding-Opponent modeling-Acceptance (BOA) pattern:

.. code-block:: python

   from negmas.gb.negotiators.modular import BOANegotiator
   from negmas.gb.components import (
       GSmithFrequencyModel,  # Opponent modeling
       GACTime,  # Acceptance strategy
       GTimeDependentOffering,  # Offering strategy
   )

   # Create a BOA negotiator with Genius-style components
   negotiator = BOANegotiator(
       offering=GTimeDependentOffering(e=0.2),  # Boulware-style offering
       acceptance=GACTime(t=0.95),  # Accept after 95% of time
       model=GSmithFrequencyModel(),  # Opponent frequency model
       name="my_boa_agent",
   )

The BOA approach is useful for:

- **Mix-and-match**: Combine different strategies from the Genius library
- **Research**: Easily swap components to compare different strategies
- **Extensibility**: Create custom components that integrate with existing ones

See Also
--------

- :doc:`tutorials/01.running_simple_negotiation` - Basic negotiation tutorial
- :doc:`tutorials/02.integrating_with_genius` - Genius integration guide
- :doc:`tutorials/03.develop_new_negotiator` - Creating custom negotiators
- :doc:`components` - Negotiation components (acceptance, offering strategies)
- :doc:`modules/sao` - SAO mechanism API reference
- :doc:`modules/gb` - GB mechanism API reference
