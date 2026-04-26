negmas.genius
=============

The genius module manages connections to the Genius platform and provides both Java-bridge and
Python-native negotiators based on the Genius BOA (Bidding, Opponent modeling, Acceptance) architecture.

Overview
--------

This module provides three ways to use Genius agents in NegMAS:

1. **GeniusNegotiator** - Connect to Java-based Genius agents through the JVM bridge
2. **Python-Native BOA Negotiators** (G-prefix) - Pure Python implementations of classic Genius agents
3. **BOA Components** - Individual components (offering, acceptance, opponent modeling) for building custom negotiators

Python-Native BOA Negotiators (Recommended)
-------------------------------------------

These negotiators are Python implementations of classic Genius agents that do NOT require the Java bridge.
They use components from :mod:`negmas.gb.components.genius` and are recommended for most use cases due to
better performance and simpler setup. All names are prefixed with ``G`` to distinguish them from Java versions.

**Classic Time-Dependent Agents**

========================================= ================================================================================
Negotiator                                Description
========================================= ================================================================================
:class:`~negmas.genius.GBoulware`         Conservative strategy (e=0.2), concedes slowly near deadline
:class:`~negmas.genius.GConceder`         Accommodating strategy (e=2.0), concedes quickly early on
:class:`~negmas.genius.GLinear`           Constant concession rate (e=1.0)
:class:`~negmas.genius.GHardliner`        Never concedes (e=0), always offers best outcome
========================================= ================================================================================

**ANAC Competition Winners and Notable Agents**

========================================= ========= ================================================================================
Negotiator                                Year      Description
========================================= ========= ================================================================================
:class:`~negmas.genius.GHardHeaded`       2011      Winner. Boulware-style with frequency-based opponent modeling
:class:`~negmas.genius.GAgentK`           2010      Top performer. Time-dependent with combined acceptance conditions
:class:`~negmas.genius.GAgentSmith`       2010      Notable. Time-dependent with constant threshold acceptance
:class:`~negmas.genius.GNozomi`           2010      Competitive. Boulware-style with previous-offer acceptance
:class:`~negmas.genius.GFSEGA`            2010      Faculty of CS agent. Conceder with constant threshold
:class:`~negmas.genius.GCUHKAgent`        2012      Winner. Conservative time-dependent with frequency opponent model
:class:`~negmas.genius.GAgentLG`          2012      Notable. Time-dependent with max-based combined acceptance
:class:`~negmas.genius.GAgentX`           2015      Notable. Adaptive time-dependent with window-based acceptance
:class:`~negmas.genius.GRandom`           ---       Random offers and accepts everything (baseline agent)
========================================= ========= ================================================================================

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from negmas.genius import GHardHeaded, GConceder
   from negmas.sao import SAOMechanism
   from negmas.preferences import LinearAdditiveUtilityFunction
   from negmas.outcomes import make_issue

   # Create a simple negotiation scenario
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]

   # Create utility functions for each negotiator
   ufun1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
   ufun2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

   # Create negotiators
   n1 = GHardHeaded(name="buyer", ufun=ufun1)
   n2 = GConceder(name="seller", ufun=ufun2)

   # Run negotiation
   mechanism = SAOMechanism(issues=issues, n_steps=100)
   mechanism.add(n1)
   mechanism.add(n2)
   mechanism.run()

Java Bridge (GeniusNegotiator)
------------------------------

For access to the full ~200 Genius agents, you can use the Java bridge. This requires:

1. Java JRE 8 or higher installed
2. Running ``negmas genius-setup`` to download the bridge
3. Starting the bridge with ``negmas genius`` before using Java agents

.. autoclass:: negmas.genius.GeniusNegotiator
   :members:
   :show-inheritance:

Bridge Management
~~~~~~~~~~~~~~~~~

.. autoclass:: negmas.genius.GeniusBridge
   :members:
   :show-inheritance:

.. autofunction:: negmas.genius.init_genius_bridge

.. autofunction:: negmas.genius.genius_bridge_is_running

.. autofunction:: negmas.genius.genius_bridge_is_installed

Constants
~~~~~~~~~

.. autodata:: negmas.genius.DEFAULT_JAVA_PORT

.. autodata:: negmas.genius.DEFAULT_PYTHON_PORT

.. autodata:: negmas.genius.DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT

.. autodata:: negmas.genius.ANY_JAVA_PORT

.. autodata:: negmas.genius.RANDOM_JAVA_PORT

.. autofunction:: negmas.genius.get_free_tcp_port

Python-Native BOA Negotiator Classes
------------------------------------

.. autoclass:: negmas.genius.GBoulware
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GConceder
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GLinear
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GHardliner
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GHardHeaded
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GAgentK
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GAgentSmith
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GNozomi
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GFSEGA
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GCUHKAgent
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GAgentLG
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GAgentX
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.genius.GRandom
   :members:
   :show-inheritance:
   :noindex:

See Also
--------

- :mod:`negmas.gb.components.genius` - BOA components used by these negotiators
- :mod:`negmas.sao` - SAO mechanism for running negotiations
- :doc:`/cli_negotiate` - CLI documentation for ``negotiate`` command

BOA Components
--------------

The G-prefixed negotiators are built from modular BOA (Bidding, Opponent modeling, Acceptance)
components available in :mod:`negmas.gb.components.genius`. These can be used to build custom
negotiators or mixed with other components.

Opponent Models
~~~~~~~~~~~~~~~

Opponent models predict the opponent's utility function based on their bidding behavior.
NegMAS includes both frequency-based and Bayesian opponent models transcompiled from Genius.

**Bayesian Models**

=============================================================== ================================================================================
Model                                                           Description
=============================================================== ================================================================================
:class:`~negmas.gb.components.genius.GBayesianModel`            Full Bayesian inference with hypothesis maintenance
:class:`~negmas.gb.components.genius.GScalableBayesianModel`    Scalable Bayesian with online learning (better for large domains)
=============================================================== ================================================================================

**Frequency-Based Models**

=============================================================== ================================================================================
Model                                                           Description
=============================================================== ================================================================================
:class:`~negmas.gb.components.genius.GHardHeadedFrequencyModel` Tracks unchanged issues between bids (from ANAC 2011 winner)
:class:`~negmas.gb.components.genius.GSmithFrequencyModel`      Simple frequency model (from AgentSmith, ANAC 2010)
:class:`~negmas.gb.components.genius.GAgentXFrequencyModel`     Advanced model with exponential smoothing (ANAC 2015)
:class:`~negmas.gb.components.genius.GNashFrequencyModel`       Frequency model biased toward Nash-optimal outcomes
=============================================================== ================================================================================

**Utility Models**

=============================================================== ================================================================================
Model                                                           Description
=============================================================== ================================================================================
:class:`~negmas.gb.components.genius.GDefaultModel`             No-op model, returns constant 0.5 utility
:class:`~negmas.gb.components.genius.GUniformModel`             Random but consistent utility for each outcome
:class:`~negmas.gb.components.genius.GOppositeModel`            Assumes opponent has opposite preferences (1 - our_utility)
=============================================================== ================================================================================

Bayesian Opponent Model Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using a Bayesian opponent model with a custom negotiator:

.. code-block:: python

   from negmas.gb.negotiators.modular import BOANegotiator
   from negmas.gb.components import (
       GTimeDependentOffering,
       GACTime,
   )
   from negmas.sao import SAOMechanism
   from negmas.preferences import LinearAdditiveUtilityFunction
   from negmas.outcomes import make_issue

   # Create scenario
   issues = [make_issue(10, "price"), make_issue(5, "quantity")]
   ufun = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)

   # Create negotiator with BOA components
   negotiator = BOANegotiator(
       name="boa_agent",
       ufun=ufun,
       offering=GTimeDependentOffering(e=0.5),
       acceptance=GACTime(t=0.95),
   )

The ``GBayesianModel`` maintains multiple hypotheses about opponent preferences and
updates their probabilities using Bayes' rule after each opponent bid. The ``GScalableBayesianModel``
uses online learning for better scalability with large outcome spaces.

Opponent Model Classes
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: negmas.gb.components.genius.GBayesianModel
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.gb.components.genius.GScalableBayesianModel
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.gb.components.genius.GHardHeadedFrequencyModel
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.gb.components.genius.GSmithFrequencyModel
   :members:
   :show-inheritance:
   :noindex:

.. autoclass:: negmas.gb.components.genius.GAgentXFrequencyModel
   :members:
   :show-inheritance:
   :noindex:
