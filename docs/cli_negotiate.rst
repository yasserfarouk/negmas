negotiate CLI
=============

The ``negotiate`` CLI provides a simple way for running negotiations, plotting them, and saving their statistics.

Installation
------------

The ``negotiate`` command is automatically available after installing NegMAS:

.. code-block:: console

    $ pip install negmas

Basic Usage
-----------

Run a simple negotiation with default settings:

.. code-block:: console

    $ negotiate

Run with a specific scenario:

.. code-block:: console

    $ negotiate --scenario path/to/scenario.yml

You can find out about all the available options by running:

.. code-block:: console

    $ negotiate --help

Command Reference
-----------------

.. typer:: negmas.scripts.negotiate:app
   :prog: negotiate
   :make-sections:
   :show-nested:

Available Negotiators
---------------------

The ``negotiate`` command supports multiple types of negotiators through prefix-based selection:

.. list-table:: Negotiator Prefixes
   :header-rows: 1
   :widths: 15 85

   * - Prefix
     - Description
   * - (none)
     - NegMAS native negotiators (e.g., ``AspirationNegotiator``)
   * - ``genius:``
     - Java Genius agents via JVM bridge (e.g., ``genius:HardHeaded``)
   * - ``anl:``
     - ANL competition agents from the ``anl-agents`` package
   * - ``boa:``
     - BOA negotiators with custom components (e.g., ``boa:offering=GTimeDependentOffering,acceptance=GACNext``)
   * - ``map:``
     - MAP negotiators with custom components
   * - ``llm:``
     - LLM-based negotiators from ``negmas-llm`` package
   * - ``negolog:``
     - Prolog-based negotiators from ``negmas-negolog`` package
   * - ``ga:``
     - Genius Agents from ``negmas-genius-agents`` package

NegMAS Native Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~~

These are pure Python negotiators that work out of the box without any external dependencies.

**Time-Based Negotiators**

These negotiators follow a time-dependent concession strategy, offering decreasing utility over time.

================================== ================================================================================
 Negotiator                         Description
================================== ================================================================================
``AspirationNegotiator``            The default time-based negotiator with configurable aspiration curve
``BoulwareTBNegotiator``            Concedes slowly (sub-linearly), making most concessions near the deadline
``LinearTBNegotiator``              Concedes at a constant rate throughout the negotiation
``ConcederTBNegotiator``            Concedes quickly (super-linearly), making most concessions early
================================== ================================================================================

**Tit-for-Tat Negotiators**

These negotiators base their concessions on the opponent's behavior.

================================== ================================================================================
 Negotiator                         Description
================================== ================================================================================
``NaiveTitForTatNegotiator``        Mirrors opponent concessions without explicit opponent modeling
``SimpleTitForTatNegotiator``       Alias for NaiveTitForTatNegotiator
================================== ================================================================================

**Other Native Negotiators**

=================================== ================================================================================
 Negotiator                          Description
=================================== ================================================================================
``ToughNegotiator``                  Only proposes and accepts the best outcome (hardliner strategy)
``TopFractionNegotiator``            Proposes and accepts only top-fraction outcomes
``NiceNegotiator``                   Offers randomly and accepts everything (maximally cooperative)
``RandomNegotiator``                 Makes random offers with configurable acceptance probability
``RandomAlwaysAcceptingNegotiator``  Random offers but accepts near-optimal outcomes
=================================== ================================================================================

**Offer-Oriented Negotiators**

These negotiators consider partner offers when selecting their own offers.

============================================== ================================================================================
 Negotiator                                     Description
============================================== ================================================================================
``FirstOfferOrientedTBNegotiator``              Considers the partner's first offer
``LastOfferOrientedTBNegotiator``               Considers the partner's most recent offer
``BestOfferOrientedTBNegotiator``               Considers the partner's best offer so far
``AdditiveParetoFollowingTBNegotiator``         Follows Pareto frontier using additive weights
``MultiplicativeParetoFollowingTBNegotiator``   Follows Pareto frontier using multiplicative weights
============================================== ================================================================================

Python-Native Genius BOA Negotiators (G-Prefix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are Python implementations of classic Genius agents, transcompiled from the original Java code.
They use the BOA (Bidding, Opponent modeling, Acceptance) architecture and do **NOT** require
the Java Genius bridge. All names are prefixed with ``G`` to distinguish them from the Java versions.

**Classic Time-Dependent Agents**

============== ================================================================================
 Negotiator     Description
============== ================================================================================
``GBoulware``   Conservative strategy (e=0.2), concedes slowly near deadline
``GConceder``   Accommodating strategy (e=2.0), concedes quickly early on
``GLinear``     Constant concession rate (e=1.0)
``GHardliner``  Never concedes (e=0), always offers best outcome
============== ================================================================================

**ANAC Competition Winners and Notable Agents**

=============== ========= ================================================================================
 Negotiator      Year      Description
=============== ========= ================================================================================
``GHardHeaded``  2011      Winner. Boulware-style with frequency-based opponent modeling
``GAgentK``      2010      Top performer. Time-dependent with combined acceptance conditions
``GAgentSmith``  2010      Notable. Time-dependent with constant threshold acceptance
``GNozomi``      2010      Competitive. Boulware-style with previous-offer acceptance
``GFSEGA``       2010      Faculty of CS agent. Conceder with constant threshold
``GCUHKAgent``   2012      Winner. Conservative time-dependent with frequency opponent model
``GAgentLG``     2012      Notable. Time-dependent with max-based combined acceptance
``GAgentX``      2015      Notable. Adaptive time-dependent with window-based acceptance
=============== ========= ================================================================================

**Utility Agents**

============== ================================================================================
 Negotiator     Description
============== ================================================================================
``GRandom``     Random offers and accepts everything (baseline agent)
============== ================================================================================

Genius Bridge Negotiators (Java)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These negotiators run actual Java Genius agents through a JVM bridge. They provide access to
the full library of ~200 Genius agents but require additional setup.

**Setup Requirements**

Before using Genius bridge negotiators, you must:

1. Install Java (JRE 8 or higher)
2. Run the Genius setup command:

.. code-block:: console

    $ negmas genius-setup

3. Start the Genius bridge (in a separate terminal or as a background process):

.. code-block:: console

    $ negmas genius

The bridge must be running whenever you use Genius negotiators.

**Using Genius Negotiators**

Genius negotiators are specified with the ``genius:`` prefix:

.. code-block:: console

    $ negotiate -n genius:HardHeaded -n genius:AgentK --steps 100

Some notable Genius negotiators include:

========================= ========= ================================================================================
 Negotiator                Year      Description
========================= ========= ================================================================================
``genius:HardHeaded``      2011      ANAC 2011 winner, frequency-based opponent modeling
``genius:CUHKAgent``       2012      ANAC 2012 winner, handles discount factors well
``genius:AgentK``          2010      Top ANAC 2010 performer
``genius:AgentSmith``      2010      Frequency-based opponent modeling
``genius:Nozomi``          2010      Boulware-style negotiation
``genius:Atlas3``          2015      ANAC 2015 top performer
``genius:ParsAgent``       2016      ANAC 2016 top performer
``genius:Caduceus``        2016      Multi-party negotiation specialist
========================= ========= ================================================================================

For a complete list of available Genius agents, see the ``negmas.genius.gnegotiators`` module.

**Note**: The G-prefixed negotiators (e.g., ``GHardHeaded``) are Python-native and do not require
the Genius bridge. They are recommended for most use cases due to better performance and simpler setup.

**Note**: The ``negotiate`` CLI will automatically attempt to start the Genius bridge if it is not
running when you use the ``genius:`` prefix. If the bridge cannot be started, helpful instructions
will be displayed.

External Package Negotiators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NegMAS supports negotiators from external packages through prefixes. These packages must be
installed separately.

**LLM Negotiators (negmas-llm)**

Use large language models for negotiation:

.. code-block:: console

    $ pip install negmas-llm
    $ negotiate -n llm:GPTNegotiator -n AspirationNegotiator --steps 50

**Negolog Negotiators (negmas-negolog)**

Use Prolog-based negotiation strategies:

.. code-block:: console

    $ pip install negmas-negolog
    $ negotiate -n negolog:PrologNegotiator -n AspirationNegotiator --steps 50

**Genius Agents (negmas-genius-agents)**

Use pre-packaged Genius-style agents:

.. code-block:: console

    $ pip install negmas-genius-agents
    $ negotiate -n ga:SomeAgent -n AspirationNegotiator --steps 50

Examples
--------

Run a negotiation with 3 issues and 100 steps:

.. code-block:: console

    $ negotiate -i 3 -s 100

Run with specific negotiators:

.. code-block:: console

    $ negotiate -n AspirationNegotiator -n NaiveTitForTatNegotiator

Run with Python-native Genius agents (no bridge required):

.. code-block:: console

    $ negotiate -n GBoulware -n GConceder --steps 50

Run with ANAC competition winners:

.. code-block:: console

    $ negotiate -n GHardHeaded -n GCUHKAgent --steps 100

Run with Java Genius agents (requires bridge):

.. code-block:: console

    $ negmas genius &  # Start bridge in background
    $ negotiate -n genius:HardHeaded -n genius:AgentK --steps 100

Run without plotting:

.. code-block:: console

    $ negotiate --no-plot

Save results to a specific path:

.. code-block:: console

    $ negotiate --save-path ./results

Run with a time limit of 60 seconds:

.. code-block:: console

    $ negotiate -t 60
