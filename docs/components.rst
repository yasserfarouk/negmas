Negotiation Components
======================

NegMAS provides modular components that can be combined to build custom negotiators.
These components implement specific aspects of negotiation behavior such as
acceptance strategies, offering strategies, and opponent modeling.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Components are the building blocks of modular negotiators. Instead of implementing
a complete negotiator from scratch, you can compose negotiators from reusable
components:

- **Acceptance Policies**: Decide whether to accept, reject, or end negotiation
- **Offering Policies**: Decide what offer to make
- **Opponent Models**: Model the opponent's preferences
- **Concession Strategies**: Control how quickly to concede

Acceptance Policies
-------------------

Acceptance policies determine how a negotiator responds to incoming offers.

Basic Acceptance Policies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AcceptImmediately``
     - Accepts any offer immediately
   * - ``RejectAlways``
     - Rejects all offers
   * - ``EndImmediately``
     - Ends negotiation immediately
   * - ``AcceptAnyRational``
     - Accepts any outcome not worse than disagreement
   * - ``AcceptBetterRational``
     - Accepts outcomes better than all previously accepted
   * - ``AcceptNotWorseRational``
     - Accepts outcomes not worse than best accepted so far

Utility-Based Acceptance
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AcceptAbove``
     - Accepts outcomes above a utility threshold
   * - ``AcceptBest``
     - Accepts only the best possible outcome
   * - ``AcceptTop``
     - Accepts outcomes in the top fraction or top k
   * - ``ACConst``
     - Accepts outcomes with utility above a constant threshold
   * - ``RandomAcceptancePolicy``
     - Accepts with configurable probability

Time-Based Acceptance
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``ACTime``
     - Accepts after a relative time threshold (tau)
   * - ``AcceptAfter``
     - Alias for ACTime
   * - ``AcceptAround``
     - Accepts around a specific relative time
   * - ``AcceptBetween``
     - Accepts within a time range

Offer-Based Acceptance
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``ACNext``
     - Accepts if offer is better than what we would propose next
   * - ``ACLast``
     - Accepts based on our last offer utility
   * - ``ACLastKReceived``
     - Accepts based on last k received offers
   * - ``ACLastFractionReceived``
     - Accepts based on offers in a time fraction
   * - ``TFTAcceptancePolicy``
     - Tit-for-tat: concedes as much as partner

Composite Acceptance
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``AllAcceptanceStrategies``
     - Accepts only if all child strategies accept
   * - ``AnyAcceptancePolicy``
     - Accepts if any child strategy accepts
   * - ``ConcensusAcceptancePolicy``
     - Base class for consensus-based acceptance
   * - ``NegotiatorAcceptancePolicy``
     - Uses another negotiator's acceptance logic
   * - ``LimitedOutcomesAcceptancePolicy``
     - Accepts from a predefined outcome list

Offering Policies
-----------------

Offering policies determine what offers a negotiator makes.

Basic Offering Policies
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``OfferBest``
     - Always offers the best outcome
   * - ``OfferTop``
     - Offers from top fraction or top k outcomes
   * - ``NoneOfferingPolicy``
     - Always offers None (no agreement)
   * - ``RandomOfferingPolicy``
     - Offers random outcomes
   * - ``LimitedOutcomesOfferingPolicy``
     - Offers from a predefined list

Time-Based Offering
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``TimeBasedOfferingPolicy``
     - Offers based on aspiration curve over time
   * - ``HybridOfferingPolicy``
     - Combines time-based and behavior-based strategies

Rational Concession Offering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``MiCROOfferingPolicy``
     - Monotonic concession - one outcome at a time
   * - ``FastMiCROOfferingPolicy``
     - Faster MiCRO that may skip outcomes
   * - ``CABOfferingPolicy``
     - Conceding Accepting Better strategy
   * - ``WAROfferingPolicy``
     - Wasting Accepting Rational strategy
   * - ``TFTOfferingPolicy``
     - Tit-for-tat offering based on partner concession

Composite Offering
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``ConcensusOfferingPolicy``
     - Base for consensus-based offering
   * - ``UnanimousConcensusOfferingPolicy``
     - Offers only if all strategies agree
   * - ``RandomConcensusOfferingPolicy``
     - Randomly selects from child strategies
   * - ``MyBestConcensusOfferingPolicy``
     - Offers best outcome from child strategies
   * - ``MyWorstConcensusOfferingPolicy``
     - Offers worst outcome from child strategies
   * - ``NegotiatorOfferingPolicy``
     - Uses another negotiator's offering logic

Genius BOA Components
---------------------

NegMAS provides Python implementations of Genius BOA (Bidding, Opponent modeling,
Acceptance) framework components.

Genius Acceptance Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GeniusAcceptancePolicy``
     - Base class for Genius acceptance policies
   * - ``GACNext``
     - Accept if better than next offer
   * - ``GACConst``
     - Accept above constant threshold
   * - ``GACTime``
     - Time-based acceptance
   * - ``GACPrevious``
     - Accept based on previous offers
   * - ``GACGap``
     - Gap-based acceptance
   * - ``GACTrue``
     - Always accept
   * - ``GACFalse``
     - Always reject
   * - ``GACCombi``
     - Combined acceptance strategy
   * - ``GACCombiMax``
     - Combined with max aggregation
   * - ``GACCombiAvg``
     - Combined with average aggregation
   * - ``GACCombiBestAvg``
     - Combined with best average
   * - ``GACCombiV2``
     - Combined version 2
   * - ``GACCombiV3``
     - Combined version 3
   * - ``GACCombiV4``
     - Combined version 4
   * - ``GACCombiMaxInWindow``
     - Combined max in sliding window
   * - ``GACCombiProb``
     - Probabilistic combined acceptance
   * - ``GACConstDiscounted``
     - Discounted constant threshold
   * - ``GACCombiBestAvgDiscounted``
     - Discounted best average
   * - ``GACCombiMaxInWindowDiscounted``
     - Discounted max in window
   * - ``GACCombiProbDiscounted``
     - Discounted probabilistic

Genius Offering Policies
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GeniusOfferingPolicy``
     - Base class for Genius offering policies
   * - ``GTimeDependentOffering``
     - Time-dependent bidding strategy
   * - ``GRandomOffering``
     - Random bidding
   * - ``GBoulwareOffering``
     - Boulware (tough) bidding
   * - ``GConcederOffering``
     - Conceder (soft) bidding
   * - ``GLinearOffering``
     - Linear concession bidding
   * - ``GHardlinerOffering``
     - Hardliner (no concession) bidding
   * - ``GChoosingAllBids``
     - Chooses from all possible bids

Genius Opponent Models
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GeniusOpponentModel``
     - Base class for opponent models
   * - ``GDefaultModel``
     - Default opponent model
   * - ``GUniformModel``
     - Assumes uniform preferences
   * - ``GOppositeModel``
     - Assumes opposite preferences
   * - ``GHardHeadedFrequencyModel``
     - HardHeaded frequency-based model
   * - ``GSmithFrequencyModel``
     - Smith frequency-based model
   * - ``GAgentXFrequencyModel``
     - AgentX frequency-based model
   * - ``GNashFrequencyModel``
     - Nash-based frequency model
   * - ``GBayesianModel``
     - Bayesian opponent model
   * - ``GScalableBayesianModel``
     - Scalable Bayesian model

Usage Examples
--------------

Building a Custom Negotiator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from negmas.sao import SAOMechanism
   from negmas.sao.negotiators import SAONegotiator
   from negmas.sao.components import (
       TimeBasedOfferingPolicy,
       ACNext,
   )
   from negmas.negotiators.helpers import PolyAspiration

   # Create a negotiator with custom components
   class MyNegotiator(SAONegotiator):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Time-based offering with boulware curve
           self.offering = TimeBasedOfferingPolicy(curve=PolyAspiration(1.0, "boulware"))
           # Accept if offer is better than what we'd propose
           self.acceptance = ACNext(offering_strategy=self.offering)

Using Genius BOA Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from negmas.sao.components import (
       GBoulwareOffering,
       GACCombi,
       GHardHeadedFrequencyModel,
   )

   # Genius-style components
   offering = GBoulwareOffering()
   acceptance = GACCombi()
   opponent_model = GHardHeadedFrequencyModel()

Combining Multiple Acceptance Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from negmas.sao.components import (
       AllAcceptanceStrategies,
       ACTime,
       AcceptAbove,
   )

   # Accept only if both conditions are met:
   # 1. After 80% of negotiation time
   # 2. Utility is above 0.6
   combined = AllAcceptanceStrategies(
       strategies=[
           ACTime(tau=0.8),
           AcceptAbove(limit=0.6),
       ]
   )

See Also
--------

- :doc:`negotiators` - Available negotiators
- :doc:`tutorials/03.develop_new_negotiator` - Creating custom negotiators
- :doc:`modules/sao` - SAO mechanism API reference
- :doc:`modules/gb` - GB mechanism API reference
