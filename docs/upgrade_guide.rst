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
`preferences.protocols` and `preferences.ufun` modules. Now `UtilityFunction`
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
