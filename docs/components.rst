Negotiation Components
======================

NegMAS provides modular components that can be combined to build custom negotiators.
These components implement specific aspects of negotiation behavior such as
acceptance strategies, offering strategies, and opponent modeling.

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
Acceptance) framework components. These are **fully implemented in Python** and
do NOT require the Java Genius bridge.

.. note::
   These components can be used standalone or combined with the Python-native
   Genius negotiators (``GBoulware``, ``GHardHeaded``, etc.) documented in
   :doc:`negotiators`.

Genius Acceptance Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Basic Acceptance Policies**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GACNext``
     - Accept if offer ≥ next planned offer (AC_Next)
   * - ``GACConst``
     - Accept if utility > constant threshold
   * - ``GACTime``
     - Time-based acceptance (after time threshold)
   * - ``GACPrevious``
     - Accept if better than opponent's previous offer
   * - ``GACGap``
     - Gap-based acceptance (utility gap analysis)
   * - ``GACTrue``
     - Always accept (baseline)
   * - ``GACFalse``
     - Always reject (baseline)

**Combined Acceptance Policies**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GACCombi``
     - Combined acceptance (AC_Next OR AC_Const)
   * - ``GACCombiMax``
     - Combined with max aggregation
   * - ``GACCombiAvg``
     - Combined with average aggregation
   * - ``GACCombiBestAvg``
     - Combined with best average
   * - ``GACCombiV2``, ``GACCombiV3``, ``GACCombiV4``
     - Combined acceptance variants
   * - ``GACCombiMaxInWindow``
     - Combined max in sliding window
   * - ``GACCombiProb``
     - Probabilistic combined acceptance

**Discounted Acceptance Policies**

For domains with time discounts:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GACConstDiscounted``
     - Discounted constant threshold
   * - ``GACCombiBestAvgDiscounted``
     - Discounted best average
   * - ``GACCombiMaxInWindowDiscounted``
     - Discounted max in window
   * - ``GACCombiProbDiscounted``
     - Discounted probabilistic

**ANAC Agent-Specific Acceptance**

Acceptance strategies derived from ANAC competition agents:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GACHardHeaded``
     - HardHeaded (ANAC 2011 winner) acceptance
   * - ``GACCUHKAgent``
     - CUHKAgent (ANAC 2012 winner) acceptance
   * - ``GACAgentK``, ``GACAgentK2``
     - AgentK series acceptance
   * - ``GACAgentLG``
     - AgentLG (ANAC 2012) acceptance
   * - ``GACNozomi``
     - Nozomi (ANAC 2010) acceptance
   * - ``GACAgentSmith``
     - AgentSmith (ANAC 2010) acceptance
   * - ``GACTheFawkes``
     - TheFawkes (ANAC 2013) acceptance
   * - ``GACGahboninho``
     - Gahboninho (ANAC 2011) acceptance
   * - ``GACABMP``
     - ABMP-style acceptance
   * - ``GACUncertain``
     - Uncertainty-aware acceptance
   * - ``GACMAC``
     - MAC acceptance strategy

Genius Offering Policies
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GTimeDependentOffering``
     - Time-dependent bidding with configurable e parameter
   * - ``GBoulwareOffering``
     - Boulware (e=0.2): concedes slowly
   * - ``GConcederOffering``
     - Conceder (e=2.0): concedes quickly
   * - ``GLinearOffering``
     - Linear (e=1.0): constant concession
   * - ``GHardlinerOffering``
     - Hardliner (e=0): never concedes
   * - ``GRandomOffering``
     - Random bid selection
   * - ``GChoosingAllBids``
     - Chooses from all possible bids

Genius Opponent Models
~~~~~~~~~~~~~~~~~~~~~~

**Basic Models**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GDefaultModel``
     - Default opponent model (no modeling)
   * - ``GUniformModel``
     - Assumes uniform preferences
   * - ``GOppositeModel``
     - Assumes opposite preferences to self
   * - ``GWorstModel``
     - Assumes worst-case opponent
   * - ``GPerfectModel``
     - Perfect knowledge (for testing)

**Frequency-Based Models**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GHardHeadedFrequencyModel``
     - HardHeaded frequency model (tracks unchanged issues)
   * - ``GSmithFrequencyModel``
     - AgentSmith frequency model
   * - ``GAgentXFrequencyModel``
     - AgentX exponential smoothing model
   * - ``GNashFrequencyModel``
     - Nash-based frequency model
   * - ``GCUHKFrequencyModel``
     - CUHKAgent frequency model
   * - ``GAgentLGModel``
     - AgentLG opponent model

**Bayesian Models**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - ``GBayesianModel``
     - Bayesian opponent model
   * - ``GScalableBayesianModel``
     - Scalable Bayesian model for large spaces
   * - ``GFSEGABayesianModel``
     - FSEGA Bayesian model
   * - ``GIAMhagglerBayesianModel``
     - IAMhaggler Bayesian model

**Agent-Specific Models**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``GTheFawkesModel``
     - TheFawkes opponent model
   * - ``GInoxAgentModel``
     - InoxAgent opponent model

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

Genius BOA components can be used to build custom negotiators without Java.

.. code-block:: python

   from negmas.gb.negotiators.modular.boa import BOANegotiator
   from negmas.gb.components.genius import (
       GBoulwareOffering,
       GACCombi,
       GHardHeadedFrequencyModel,
   )


   # Create a custom HardHeaded-style negotiator
   class MyHardHeadedAgent(BOANegotiator):
       def __init__(self, **kwargs):
           offering = GBoulwareOffering()
           acceptance = GACCombi(offering_policy=offering)
           model = GHardHeadedFrequencyModel()
           super().__init__(
               offering=offering, acceptance=acceptance, model=model, **kwargs
           )

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
