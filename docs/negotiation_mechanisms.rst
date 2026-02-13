Negotiation Mechanisms
======================

NegMAS implements several negotiation mechanisms (protocols) that govern how
negotiators interact and reach agreements. This page provides an overview of
available mechanisms and when to use each one.

Available Mechanisms
--------------------

SAO Mechanism (Stacked Alternating Offers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SAO mechanism is the most commonly used protocol for bilateral negotiations.
It implements the classic alternating offers protocol where negotiators take
turns proposing and responding to offers.

**Key Features:**

- Bilateral (two-party) negotiations
- Alternating offers protocol
- Supports time limits (steps or real-time)
- Rich state tracking and history

**When to Use:**

- Standard bilateral negotiations
- ANAC-style competitions
- When you need a simple, well-understood protocol

**Example:**

.. code-block:: python

    from negmas.sao import SAOMechanism
    from negmas.sao.negotiators import AspirationNegotiator
    from negmas.preferences import LinearAdditiveUtilityFunction as U
    from negmas.outcomes import make_issue

    # Create issues
    issues = [make_issue(10, "price"), make_issue(5, "quantity")]

    # Create mechanism
    mechanism = SAOMechanism(issues=issues, n_steps=100)

    # Add negotiators
    mechanism.add(AspirationNegotiator(name="buyer"), ufun=U.random(issues=issues))
    mechanism.add(AspirationNegotiator(name="seller"), ufun=U.random(issues=issues))

    # Run negotiation
    mechanism.run()

    # Check result
    print(f"Agreement: {mechanism.agreement}")

**API Reference:** :doc:`modules/sao`


GB Mechanism (General Bargaining)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GB mechanism family provides more flexible protocols supporting various
bargaining scenarios including multi-party negotiations.

**Key Features:**

- Supports bilateral and multi-party negotiations
- Flexible response types
- Extended outcome support
- Thread-based negotiation tracking

**When to Use:**

- Multi-party negotiations
- When you need more flexible response options
- Complex negotiation scenarios

**API Reference:** :doc:`modules/gb`


ST Mechanism (Single Text)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Single Text mechanism implements a mediated negotiation protocol where
a single text (proposal) is iteratively refined based on negotiator feedback.

**Key Features:**

- Mediated negotiation
- Single proposal refined over time
- Suitable for complex multi-issue negotiations

**When to Use:**

- Mediated negotiations
- When direct offers are difficult to formulate
- Multi-party consensus building

**API Reference:** :doc:`modules/st`


MT Mechanism (Multiple Text)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Multiple Text mechanism extends the single text approach by maintaining
multiple proposals simultaneously.

**Key Features:**

- Multiple proposals tracked simultaneously
- Supports exploring different solution paths
- Good for complex negotiations

**When to Use:**

- When multiple solution paths should be explored
- Complex multi-party negotiations

**API Reference:** :doc:`modules/mt`


GA Mechanism (Genetic Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GA mechanism uses genetic algorithm principles for negotiation, evolving
proposals over generations.

**Key Features:**

- Evolutionary approach to finding agreements
- Population-based proposal generation
- Fitness-based selection

**When to Use:**

- Large outcome spaces
- When traditional negotiation is computationally expensive
- Exploratory negotiations

**API Reference:** :doc:`modules/ga`


Mechanism Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Mechanism
     - Parties
     - Mediated
     - Best For
   * - SAO
     - 2
     - No
     - Standard bilateral negotiations, competitions
   * - GB
     - 2+
     - No
     - Flexible multi-party negotiations
   * - ST
     - 2+
     - Yes
     - Consensus building with mediator
   * - MT
     - 2+
     - Yes
     - Exploring multiple solution paths
   * - GA
     - 2+
     - Yes
     - Large outcome spaces, evolutionary search


Common Mechanism Features
-------------------------

All NegMAS mechanisms share common features:

**Time Limits:**

- ``n_steps``: Maximum number of negotiation rounds
- ``time_limit``: Real-time limit in seconds
- ``step_time_limit``: Time limit per step

**State Tracking:**

- Full history of offers and responses
- Current state accessible via ``mechanism.state``
- Relative time tracking (0.0 to 1.0)

**Callbacks:**

- ``on_negotiation_start``: Called when negotiation begins
- ``on_negotiation_end``: Called when negotiation ends
- ``on_round_start``/``on_round_end``: Per-round callbacks


API Reference
-------------

.. toctree::
    :maxdepth: 2

    modules/sao
    modules/gb
    modules/st
    modules/mt
    modules/ga


See Also
--------

- :doc:`negotiators` - Available negotiators for each mechanism
- :doc:`components` - Modular components for building negotiators
- :doc:`tutorials/01.running_simple_negotiation` - Getting started tutorial
- :doc:`tutorials/04.develop_new_mechanism` - Creating custom mechanisms
