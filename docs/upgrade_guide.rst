0.8->0.9 Upgrade Guide
======================

NegMAS 0.9 is not backward  compatible with NegMAS 0.8 and to use it you
will need to make some modifications to your code. This guide aims at helping
you achieve this with minimal hassle.

Summary
-------

**Must Do**

============================  ===================================  ===============================================
 from                          to                                   Notes
============================  ===================================  ===============================================
Issue(...)                    make_issue(...)                      The same paramters are accepted
outcome_as_dict(x, ...)       outcome2dict(x, ..., issues=issue)   Must pass the issues
outcome_as_tuple(x, ...)      x                                    Just remove the call (all outcomes are tuples)
negmas.java                   negmas.serialization
to_java                       to_dict                              Java interfaces (through jnemgas) are not supported anymore.
============================  ===================================  ===============================================

**Should Do**

=================================  ===================================  ===================================================================
 from                              to                                   Notes
=================================  ===================================  ===================================================================
import negmas.utilities            import negmas.preferences            If not done, a deprication warning will be issued.
load_genius_domain_from_folder     Scenario.from_genius_folder          Some of the parameters are no longer supported. Check your use-case
LinearAggregationUtilityFunction   LinearAdditiveUtilityFunction        The old class name is still provided.
ami                                nmi                                  Member of all `Negotiator` objects
AgentMechanismInterface            NegotiatorMechanismInterface         The class was renamed to better reflect its role. The old name still works but is depricated
=================================  ===================================  ===================================================================

Outcome Type
------------

In NegMAS 0.8, you were able to use dicts, lists, tuples, or `OutcomeType` objects as
outcomes. For examle if we have two issues (price and quantity), you coud represent an
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

All mechnisms now receive their outcome-space either as an `oucome_space`, a
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

Developping agents and negotiators in Java is no longer supported. This means that `jnegmas` is no longer needed or used.
