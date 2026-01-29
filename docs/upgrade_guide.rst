0.8->0.9 Upgrade Guide
======================

NegMAS 0.9 is *not* backward compatible with NegMAS 0.8 and to use it you
will need to make some modifications to your code. This guide aims at helping
you achieve this with minimal hassle.

Summary
-------

**Must Do**

============================  ===================================  ===============================================
 from                          to                                   Notes
============================  ===================================  ===============================================
Issue(...)                    make_issue(...)                      The same parameters are accepted
outcome_as_dict(x, ...)       outcome2dict(x, ..., issues=issue)   Must pass the issues
outcome_as_tuple(x, ...)      x                                    Just remove the call (all outcomes are tuples)
negmas.java                   negmas.serialization
to_java                       to_dict                              Java interfaces (through jnemgas) are not supported anymore.
from_genius/to_genius                                              Remove keep_issue_names, keep_issue_values (not supported anymore)
Negotiator.on_ufun_changed    Negotiator.on_preferences_changed
Negotiator._utility_function  Negotiator.ufun
SAOAMI                        SAONMI
PassThroughNegotiator         ControlledNegotiator
PassThroughSAONegotiator      ControlledNegotiator
from negmas.*.passthrough     from negmas.*.controlled             where `*` here stands for any submodule
============================  ===================================  ===============================================

The following rules  must be followed:

- Never change the internal structure of a utility function after it is
  constructed (i.e. do not change the weights on a `LinearUtilityFunction` )
  because the negotiator will have no way to know
  about this change. To help enforcing that, almost all members of all ufuns
  are now private (i.e. starting with an `_` ) and only getter properties for
  them are provided. Some members like `outcome_spaace` ,
  `issues`  are still just data members but you should never set them after the
  ufun is constructed (in the pythonic should also spirit of "we are all
  consenting adults here").
- In general, you should never need to access a private member of any class
  (i.e. a member starting with `_` ). If you do need that, it is a bug. Please
  raise an issue in github.
- Do not pass `outcome_type` to any ufun constructor. It is now removed as all
  outcomes are guaranteed to have the type `tuple` now.
- Some I/O methods (and few others) had `force_single_issue`, `keep_issue_names` and `keep_issue_values`
  parameters to do on-the-fly type conversions and change the type of the outcomes.
  All of this is removed now. Conversion of issue spaces can be done explicitly
  using methods in the `OutcomeSpace` protocol like `to_single` and `to_numeric`.


**Should Do**

=================================  ===================================  ===================================================================
 from                              to                                   Notes
=================================  ===================================  ===================================================================
import negmas.utilities            import negmas.preferences            If not done, a deprication warning will be issued.
load_genius_domain_from_folder     Scenario.from_genius_folder          Some of the parameters are no longer supported. Check your use-case
LinearUtilityAggregationFunction   LinearAdditiveUtilityFunction        The old class name is still provided.
Negotiator.ami                     Negotiator.nmi                       Member of all `Negotiator` objects
Controller.get_ami                 Controller.get_nmi
AgentMechanismInterface            NegotiatorMechanismInterface         The class was renamed to better reflect its role. The old name still works but is deprecated
LinearUtilityFunction(bias!=0)     AffineUtilityFunction                We are reserving the name `LinearUtilityFunction` to ufuns with zero offset and `AffineUtilityFunction` for those with potentially nonzero offset
=================================  ===================================  ===================================================================

The following rules *should* be followed:

- It is better to always tell the ufun its outcome-space by passing `outcome_space` , `issues`
  , or `outcomes` to it. Strictly speaking you do not need to do that for many
  scenarios (specially for affine and linear utility functions) but some
  operations may fail if the ufun does not know its outcome-space. For example,
  if you construct a utility function without passing an outcome space to it, you
  will get an exception if you try to call `extreme_outcomes()` or `minmax()` on
  it later as the ufun has no way to know how to evaluate either. Because some
  builtin negotiators (e.g. `TimeBasedNegotiator` ) do use these methods internally,
  you will get these exceptions at the time of calling said methods which may be
  tricky to debug.

Outcome Type
------------

In NegMAS 0.8, you were able to use dicts, lists, tuples, or `OutcomeType` objects as
outcomes. For example if we have two issues (price and quantity), you could represent an
outcome  as any of the following:

.. code-block ::

   w = (0, 1)
   w = dict(price=0, quantity=1)
   w = O(price=0, quantity=1)

whare `O` is a dataclass inherited from `OutcomeType`.

All of this is gone in NegMAS 0.9 to simplify the code base (by removing hundreds of `isinstance` calls).
Now all outcomes are **tuples**. In the above examples only the first one is supported.

This actually simplify your code as you do not need to check the type of the outcome you are receiving in
any callbacks (e.g. the `respond()` method of the `SAONegotiator` class).

As a side effect, some functions have been removed or renamed.


.. caution::

   Note that `Contract` in the `situated` module still stores the `agreement` as a dictionary as it is usually
   easier to understand from logs.


Multiple Issue Types
--------------------

In NegMAS 0.8 we had a single `Issue` class representing all types of issues.
This led to several complications as each method of this class had to check for
the exact *type* of itself (i.e. is it an ordinal issue, a continuous issue,
...). To make the codebase more maintainable, `Issue` is now an abstract base
class that is inherited by specific issue types (e.g. `OrdinalIssue`,
`ContinuousIssue` , etc). We also provide a factory function `make_issue` that
takes the same parameters as the constructor of NegMAS 0.8's `Issue` class and
create the appropriate type.

To upgrade to the new version with minimum effort, replace all calls to `Issue`
with `make_issue`. For example:

.. code-block::

  issue = Issue(...)

becomes:

.. code-block::

  issue = make_issue(...)


Outcome Space Class
-------------------

In NegMAS 0.8, outcome spaces were represented with lists of `Issue` s. You can
still do that in NegMAS 0.9  but it is recommended to use the newly added
`OutcomeSpace` hierarchy of classes for that. This allows you to use convenient
functions defined on these classes to manipulate outcome-spaces which can be
specially helpful for mechanism designers.

You do not need to change your code in any way to be compatible with this
feature but it is recommended that you start using outcome-spaces instead of
lists of issues. We provide a convenient `make_os` factory function for
constructing outcome spaces from lists of issues, or lists of outcomes.

All mechanisms now receive their outcome-space either as an `oucome_space`, a
list of `Issue` objects, or a list of `Outcome` objects.


Preferences Module Restructuring
--------------------------------

In NegMAS 0.8, we had a single `UtilityFunction` class that represented all
sorts of interfaces. For example if you implemented the `eval()` method it
acted like a normal utility function that can be  called to return the utility
of an outcome. If you implemented instead the `is_better()` method the same
class acted like a representation of ordinal preferences (with no utility value
per outcome being defined). Needless to say, desipte its ease of use, several
edge cases were difficult to handle and again we had to resort to runtime type
checking too much. Moreover, it is difficult to follow the code of our
implementation. All of this was implemented in a single-file `utilities` module
with thousands of lines.

In NegMAS 0.9, the `utilities` module was renamed `preferences` and we replaced
the monolithic `UtiltiyFunction` class with multiple classes implementing
different types of preferences. You can check the new hierarchy in the
`preferences.protocols` and `preferences.base_ufun` modules. Now `UtilityFunction`
is reserved for crisp utility functions that define a real value for each outcome.

**If you are using `UtilityFunction` in that  sense (which is likely), you do not need
to change anything in your code  except importing from `preferences` instead of `utiltiies`**.

Input and Output
----------------

We added a new class `Scenario` to represent a negotiation scenario (i.e. agneda and ufuns).
It is the recommended way to load/save negotiation scenarios now. It can be used to load/save
Genius XML scenarios as well as json versions.

Moreover, we removed some of the parameters in `load_genius_domain_from_*` functions
(`keep_issue_names`, `keep_value_names`, ...) that are not needed anymroe now that outcomes
are always tuples.

Once you create a `Scenario` using something like `from_genius_folder` , you can now
do several operations on it like converting it to a single-issue negotiation using `to_single_issue()`
or to an all-numeric negotiation using `to_numeric()` . Whenever you do something like this
the ufuns will be changed appropriately.


Java Support
------------

Developing agents and negotiators in Java is no longer supported. This means that `jnegmas` is no longer needed or used.


Other Changes
-------------

NegMAS 0.9 has other changing that can be potentially breaking but are
justified by the more consistency they bring and/or their performance edge.
Most of these changes have no effect on well-behaving code using the library:

- We renamed PassThroughNegotiator types to `ControlledNegotiator` types to better
  document their roles. These negotiators allow for user-controlled separation of
  responsibilities between the `Controller` and the `Negotiator` . The old name
  suggested that the negotiator **cannot** do anything (just a pass-through entity).
- In most cases, we use the more general term `preferences` instead of `ufun`
  whenever possible. For example, `on_ufun_changed` was renamed to
  `on_preferences_changed` to make it clear that general preferences can be
  used not only ufuns.
- Some methods now receive both `preferences` and `ufun` arguments (instead of
  only `ufun` ) with the `ufun` argument overriding the `preferences` argument
  when given.
  This was done (instead of just renaming the `ufun` argument to `preferences`
  ) to reduce the effect on downstream code.
- The negotiator is not notified that its preferences have changed (through a
  call to its `on_preferences_changed()` method) only when it is about to start
  a negotiation even if the assignment of preferences was done in construction
  (by passing `preferences` to the constructor) or by `set_preferences()`
  before joining. This has two advantages:

  1. The later call makes it more likely that all data needed for the
     negotiator for using this callback is available. For example, if the
     negotiator is created by an agent to be used with multiple negotiations,
     it may be the case that the setting of preferences happens in the agent's
     `init()` method before the `awi` is set.
     By delaying the call to `on_preferences_changed()` we make sure that the
     `awi` is available in case it is needed.
  2. In some cases, the negotiator may be constructed by never joins a
     negotiation. It is a waste of resources to compute whatever
     `on_preferences_changed()` is computing in such cases as the preferences
     will never be really used.

0.9->0.10 Upgrade Guide
=======================

NegMAS 0.10 is *not* backward compatible with NegMAS 0.9 and to use it you
will need to make some modifications to your code. This guide aims at helping
you achieve this with minimal hassle.

=================================== ===================================  =======================================================================================================================
 From                                To                                   Notes
=================================== ===================================  =======================================================================================================================
def respond(self,state, offer,...)      def respond(self, state, ...)            SAO negotiator's respond() does not receive the offer anymore. You can get the offer as `state.current_offer`
PreferenceChangeType.Scaled          PreferenceChangeType.Scale
PreferenceChangeType.Shifted          PreferenceChangeType.Shift
=================================== ===================================  =======================================================================================================================

Other Changes
-------------

- You can now step any world focusing on negotiations instead of simulation step boundaries. See documentation of `World.step` for details which helps with exposing NegMAS worlds as RL environments.
- You can now pass negotiation actions to the `SAOMechanism` (and to some extend the `GBMechanism` ) which is useful when using RL on negmas.

0.10->0.11 Upgrade Guide
========================

NegMAS 0.11 is *not* backward compatible with NegMAS 0.10 and to use it you
will need to make some modifications to your code. This guide aims at helping
you achieve this with minimal hassle.

=================================== ======================================  =======================================================================================================================
 From                                To                                      Notes
=================================== ======================================  =======================================================================================================================
def propose(self, state)             def propose(self, state, dest=None)     The `propose()` function of SAONegotiator now takes an optional `dest` parameter.
def respond(self, state)             def respond(self, state, src=None)      The `respond()` function of SAONegotiator now takes an optional `src` parameter.
def __call__(self, state)            def __call__(self, state, dest=None)    The `__call__()` function of SAONegotiator now takes an optional `dest` parameter.
=================================== ======================================  =======================================================================================================================

0.14->0.15 Upgrade Guide
========================

NegMAS 0.15 introduces per-negotiator limits and deprecates ``Mechanism.nmi`` in favor of ``shared_nmi``.
Most code will continue to work without changes, but you should migrate to the new API.

Breaking Changes
----------------

Mechanism NMI Deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Mechanism.nmi`` property is deprecated in favor of ``shared_nmi`` to better distinguish
between shared mechanism interfaces and per-negotiator interfaces.

=================================== ===================================  =======================================================================================================================
 From                                To                                   Notes
=================================== ===================================  =======================================================================================================================
``mechanism.nmi``                   ``mechanism.shared_nmi``             Use ``shared_nmi`` to access the shared negotiator-mechanism interface
=================================== ===================================  =======================================================================================================================

**Example Migration:**

.. code-block:: python

   # Old way (deprecated, still works with warning)
   relative_time = mechanism.nmi.relative_time
   n_steps = mechanism.nmi.n_steps

   # New way (recommended)
   relative_time = mechanism.shared_nmi.relative_time
   n_steps = mechanism.shared_nmi.n_steps

**Why this change?**

With the introduction of per-negotiator limits in 0.15, each negotiator can have its own
``NegotiatorMechanismInterface`` (NMI) with different time and step limits. The ``shared_nmi``
property clarifies that you're accessing the shared interface that applies to all negotiators,
not a specific negotiator's interface.

New Features
------------

Per-Negotiator Time and Step Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mechanisms now support individual time and step limits for each negotiator. This enables
asymmetric negotiation scenarios where different agents have different resources.

**Basic Usage:**

.. code-block:: python

   from negmas import SAOMechanism, TimeBasedConcedingNegotiator

   # Create a mechanism
   mechanism = SAOMechanism(outcomes=10, n_steps=100)

   # Add negotiators with different limits
   mechanism.add(
       TimeBasedConcedingNegotiator(name="fast"),
       time_limit=5.0,  # 5 seconds max
       n_steps=50,  # 50 steps max
   )

   mechanism.add(
       TimeBasedConcedingNegotiator(name="slow"),
       time_limit=10.0,  # 10 seconds max
       n_steps=100,  # 100 steps max
   )

   # Run the negotiation
   mechanism.run()

**Three-Tier Limit System:**

1. **Shared limits**: Apply to all negotiators (set via ``Mechanism.__init__`` or on ``shared_nmi``)
2. **Private limits**: Apply to individual negotiators (set via ``Mechanism.add(..., time_limit=X, n_steps=Y)``)
3. **Effective limits**: Computed as ``min(shared, private)`` - the most restrictive wins

**Accessing Per-Negotiator State:**

.. code-block:: python

   # Get state for a specific negotiator
   state = mechanism.state_for(negotiator)

   # The state includes per-negotiator relative_time
   # based on that negotiator's effective limits
   print(f"Relative time for {negotiator.name}: {state.relative_time}")

**Backward Compatibility:**

Existing code continues to work without changes. If you don't specify per-negotiator limits,
negotiators use the shared limits (same behavior as before).

Offline Negotiation Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New ``CompletedRun`` class enables saving and visualizing completed negotiations offline.

**Saving a Completed Negotiation:**

.. code-block:: python

   from negmas import SAOMechanism

   # Run a negotiation
   mechanism = SAOMechanism(...)
   mechanism.add(...)
   mechanism.run()

   # Convert to CompletedRun and save
   completed_run = mechanism.to_completed_run(source="full_trace")
   completed_run.save("my_negotiation.yaml")

**Loading and Plotting:**

.. code-block:: python

   from negmas import CompletedRun

   # Load the completed run
   run = CompletedRun.load("my_negotiation.yaml", load_scenario=True)

   # Plot the negotiation (same as SAOMechanism.plot())
   run.plot(show_all_offers=True, show_annotations=True, save_fig="negotiation.png")

**Requirements for Plotting:**

- Must use ``source="full_trace"`` when creating the ``CompletedRun``
- Must load with ``load_scenario=True`` to include utility functions
- Produces identical visualizations to ``SAOMechanism.plot()``

Scenario Rotation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New ``Scenario.rotate_ufuns()`` method enables efficient scenario rotation without
recalculating statistics.

.. code-block:: python

   from negmas import Scenario

   # Load a scenario
   scenario = Scenario.from_genius_folder("path/to/domain")

   # Rotate utility functions by 1 position
   # (u0, u1, u2) → (u2, u0, u1)
   rotated = scenario.rotate_ufuns(n=1)

   # Statistics are automatically rotated (no recalculation needed)
   # This is much faster than recalculating stats for each rotation

**Performance Impact:**

When using ``cartesian_tournament`` with ``rotate_ufuns=True`` and the new default
``recalculate_stats=False``, you can see 1.3-2x speedup depending on scenario complexity.

Matplotlib Plotting Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Scenario.plot()`` now supports both matplotlib and plotly backends.

.. code-block:: python

   from negmas import Scenario

   scenario = Scenario.from_genius_folder("path/to/domain")

   # Use plotly (default, interactive)
   fig = scenario.plot(backend="plotly")

   # Use matplotlib (faster for static images)
   fig = scenario.plot(backend="matplotlib", save_fig="scenario.png")

**When to use each backend:**

- **plotly**: Interactive plots, web applications, Jupyter notebooks
- **matplotlib**: Static images, publications, faster rendering for large datasets

Both backends produce visually consistent output (same colors, markers, legends).

0.13->0.14 Upgrade Guide
========================

NegMAS 0.14 introduces a major refactoring of the registry system to use tags instead of
boolean fields. While backward compatibility is maintained through deprecation warnings,
you should update your code to use the new tag-based API.

Registry System Changes
-----------------------

The registry system (``MechanismInfo``, ``NegotiatorInfo``, ``ScenarioInfo``) now uses
**tags** instead of individual boolean fields. This provides more flexibility and
extensibility.

**MechanismInfo Changes:**

=============================  ===========================================
 Old Parameter                  New Tag
=============================  ===========================================
``requires_deadline=True``     ``tags=["requires-deadline"]``
=============================  ===========================================

**NegotiatorInfo Changes:**

=============================  ===========================================
 Old Parameter                  New Tag
=============================  ===========================================
``bilateral_only=True``        ``tags=["bilateral-only"]``
``requires_opponent_ufun``     ``tags=["requires-opponent-ufun"]``
``learns=True``                ``tags=["learning"]``
``anac_year=2019``             ``tags=["anac-2019"]``
``supports_uncertainty``       ``tags=["supports-uncertainty"]``
``supports_discounting``       ``tags=["supports-discounting"]``
=============================  ===========================================

**ScenarioInfo Changes:**

=============================  ===========================================
 Old Parameter                  New Tag
=============================  ===========================================
``normalized=True``            ``tags=["normalized"]``
``anac=True``                  ``tags=["anac"]``
``file=True``                  ``tags=["file-based"]``
``format="xml"``               ``tags=["format-xml"]``
``has_stats=True``             ``tags=["has-stats"]``
``has_plot=True``              ``tags=["has-plot"]``
=============================  ===========================================

**Example Migration:**

.. code-block:: python

   # Old way (deprecated, still works with warning)
   @negotiator_info(bilateral_only=True, learns=True, anac_year=2019)
   class MyNegotiator(SAONegotiator):
       pass


   # New way (recommended)
   @negotiator_info(tags=["bilateral-only", "learning", "anac-2019"])
   class MyNegotiator(SAONegotiator):
       pass

**Querying the Registry:**

.. code-block:: python

   # Find all learning negotiators
   learning_negotiators = registry.negotiators.filter(tags=["learning"])

   # Find all ANAC 2019 negotiators
   anac2019 = registry.negotiators.filter(tags=["anac-2019"])

   # Find negotiators with multiple tags
   bilateral_learning = registry.negotiators.filter(tags=["bilateral-only", "learning"])

Tournament Default Changes
--------------------------

The ``cartesian_tournament`` function now uses more memory-efficient defaults:

=============================  ========================  ========================
 Parameter                      Old Default               New Default
=============================  ========================  ========================
``storage_optimization``       ``"speed"``               ``"space"``
``memory_optimization``        ``"speed"``               ``"balanced"``
Stats file                     ``stats.json``            ``_stats.yaml``
=============================  ========================  ========================

If you need the old behavior for performance reasons:

.. code-block:: python

   results = cartesian_tournament(
       ...,
       storage_optimization="speed",
       memory_optimization="speed",
   )

Elicitation Module Extraction
-----------------------------

The elicitation module has been extracted to a separate package ``negmas-elicit``.

If you were using elicitation features:

.. code-block:: bash

   pip install negmas-elicit

.. code-block:: python

   # Old import (no longer works)
   # from negmas.elicitation import SomeClass

   # New import
   from negmas_elicit import SomeClass
