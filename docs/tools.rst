************
NegMAS Tools
************

Beyond negotiators and mechanisms, NegMAS ships a large toolbox of **reusable
building blocks** for defining problems, describing preferences, and analyzing
outcomes. These tools are useful on their own — for generating benchmarks,
pre-processing scenarios, or studying the geometry of a negotiation — even if you
never run a full session.

This page is a practical, example-driven tour of four toolboxes:

* :ref:`tools-outcomes` — issues, outcome spaces, enumeration/sampling, discretization.
* :ref:`tools-ufuns` — utility-function operations (ranges, extremes, scaling, ranking).
* :ref:`tools-scenario` — loading, transforming, saving and running whole scenarios.
* :ref:`tools-analysis` — Pareto frontier, Nash / Kalai / Kalai-Smorodinsky points, and conflict measures.

.. contents::
    :local:
    :depth: 2
    :class: this-will-duplicate-information-and-it-is-still-useful-here

Every code block on this page is self-contained and tested against the current
API.


.. _tools-outcomes:

Outcome-space tools
===================

Issues
------

An **issue** is one dimension of a negotiation (a price, a quantity, a delivery
date, ...). :func:`~negmas.outcomes.make_issue` is the one-stop constructor; the
kind of issue it builds depends on the ``values`` you pass:

.. code-block:: python

    from negmas import make_issue

    categorical = make_issue(["dell", "hp", "lenovo"], "brand")  # a list -> categorical
    countable = make_issue(11, "price")  # an int  -> 0..10 (contiguous)
    ranged = make_issue((0, 100), "quantity")  # int tuple -> integer range
    continuous = make_issue((0.0, 1.0), "quality")  # float tuple -> continuous

    assert categorical.is_discrete() and countable.is_discrete()
    assert continuous.is_continuous()
    assert countable.cardinality == 11

The concrete classes (all importable from :mod:`negmas.outcomes`) include
:class:`~negmas.outcomes.CategoricalIssue`,
:class:`~negmas.outcomes.ContiguousIssue` (integer range),
:class:`~negmas.outcomes.ContinuousIssue`,
:class:`~negmas.outcomes.OrdinalIssue`, and the infinite variants
:class:`~negmas.outcomes.ContinuousInfiniteIssue` /
:class:`~negmas.outcomes.CountableInfiniteIssue`. You normally never instantiate
these directly — ``make_issue`` picks the right one.

Outcome spaces
--------------

An **outcome space** is the (usually Cartesian) product of issues — the full set
of possible agreements. :func:`~negmas.outcomes.make_os` builds one:

.. code-block:: python

    from negmas import make_issue
    from negmas.outcomes import make_os

    os = make_os([make_issue(11, "price"), make_issue(["new", "used"], "condition")])
    assert os.cardinality == 22  # 11 x 2
    assert os.is_finite() and os.is_discrete()

    an_outcome = os.random_outcome()  # e.g. (7, "used")
    assert os.is_valid(an_outcome)

Key outcome-space types are
:class:`~negmas.outcomes.CartesianOutcomeSpace` (the general product) and
:class:`~negmas.outcomes.DiscreteCartesianOutcomeSpace` (all issues discrete, so
it can be fully enumerated). Useful methods available on any outcome space:

======================================  ====================================================
Method                                  Purpose
======================================  ====================================================
``cardinality``                         Number of outcomes (``inf`` if infinite/continuous).
``is_finite()`` / ``is_discrete()``     Whether it can be enumerated.
``enumerate_or_sample(...)``            Enumerate if finite, else sample.
``random_outcome()`` / ``sample(n)``    Draw random outcome(s).
``is_valid(outcome)``                   Membership test.
``to_discrete(levels)``                 Discretize continuous issues into ``levels`` steps.
``to_largest_discrete(levels, max_cardinality)``  Discretize as finely as a cardinality budget allows.
``to_single_issue(...)``                Flatten to one enumerated issue.
``cardinality_if_discretized(levels)``  Size the discretized space *without* building it.
======================================  ====================================================

Enumeration and sampling
------------------------

For a finite space, ``enumerate_or_sample`` gives you every outcome; for an
infinite one it samples. The free functions
:func:`~negmas.outcomes.enumerate_issues` and
:func:`~negmas.outcomes.sample_outcomes` work directly on a list of issues:

.. code-block:: python

    from negmas import make_issue
    from negmas.outcomes import make_os, enumerate_issues, sample_outcomes

    issues = [make_issue(3, "x"), make_issue(3, "y")]
    all_outcomes = enumerate_issues(issues)  # 9 outcomes
    assert len(all_outcomes) == 9

    os = make_os(issues)
    assert len(list(os.enumerate_or_sample())) == 9

    # Sample from a (possibly huge or continuous) space:
    some = sample_outcomes([make_issue((0.0, 1.0), "p")], n_outcomes=20)

Discretization
--------------

Continuous or very large spaces must be discretized before they can be
enumerated or analyzed. The simplest approach is uniform:

.. code-block:: python

    from negmas import make_issue
    from negmas.outcomes import make_os

    os = make_os([make_issue((0.0, 10.0), "price"), make_issue(["a", "b", "c"], "grade")])
    assert not os.is_finite()

    grid = os.to_discrete(levels=5)  # 5 price levels x 3 grades
    assert grid.cardinality == 15

    # Or: "as fine as possible under a budget"
    capped = os.to_largest_discrete(levels=1000, max_cardinality=30)

For **non-uniform** discretization NegMAS provides a *Discretizer framework*. A
discretizer is a callable that maps an outcome space to a discrete one; the
built-ins are registered by name in
:data:`~negmas.outcomes.DISCRETIZERS` and resolved with
:func:`~negmas.outcomes.get_discretizer`:

.. code-block:: python

    from negmas import make_issue
    from negmas.outcomes import make_os, DefaultDiscretizer, get_discretizer

    os = make_os([make_issue((0.0, 1.0), "a"), make_issue((0.0, 1.0), "b")])

    # DefaultDiscretizer is a uniform grid capped by an outcome budget:
    discretizer = DefaultDiscretizer(max_outcomes=25)
    discrete_os = discretizer(os)  # discretizers are called on the space
    assert discrete_os.cardinality == 25

    GridBased = get_discretizer("grid_based")  # look a discretizer up by name

Besides ``grid_based``, the registry contains *balanced* discretizers
(``balanced_ufun_variance``, ``balanced_ufuns_variance``,
``balanced_outcome_counts_in_ufun_bins``,
``balanced_outcome_counts_in_ufuns_bins``) that place bin edges using one or more
utility functions so that the resulting levels are balanced in utility variance
or in the number of outcomes per bin — handy when a uniform grid would over- or
under-sample interesting regions.


.. _tools-ufuns:

Utility-function tools
======================

Every utility function (subclass of
:class:`~negmas.preferences.BaseUtilityFunction`) exposes a set of tools for
inspecting and transforming it.

Ranges and extremes
--------------------

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearUtilityFunction

    issues = [make_issue(11, "price"), make_issue(6, "quantity")]
    u = LinearUtilityFunction(weights=[1.0, -1.0], issues=issues, reserved_value=0.0)

    mn, mx = u.minmax()  # utility range over the whole space
    worst, best = u.extreme_outcomes()  # the argmin / argmax outcomes
    assert abs(u(best) - mx) < 1e-9
    # eval_normalized maps utilities into [0, 1] on the fly (0 == reserved value)
    assert 0.0 <= u.eval_normalized((10, 0)) <= 1.0

Scaling, shifting and normalizing
---------------------------------

``scale_by`` / ``shift_by`` apply an affine transform (optionally to the reserved
value too); ``normalize`` / ``normalize_for`` map utilities into a target range.
Normalization has its own detailed page — see :doc:`normalization`.

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearUtilityFunction

    u = LinearUtilityFunction(
        weights=[1.0, -1.0], issues=[make_issue(11, "p"), make_issue(6, "q")]
    )
    doubled = u.scale_by(2.0)
    shifted = u.shift_by(5.0)
    n = u.normalize(to=(0.0, 1.0))  # range becomes exactly [0, 1]
    lo, hi = n.minmax()
    assert abs(lo) < 1e-6 and abs(hi - 1.0) < 1e-6

The same operations are available as **free functions** in
:mod:`negmas.preferences.ops`, which additionally offers ranking and
reserved-value repair:

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearUtilityFunction
    from negmas.preferences.ops import scale_max, make_rank_ufun, correct_reserved_value

    u = LinearUtilityFunction(
        weights=[1.0, -1.0], issues=[make_issue(11, "p"), make_issue(6, "q")]
    ).normalize()

    u5 = scale_max(u, to=5.0)  # rescale so the maximum equals 5
    assert abs(u5.max() - 5.0) < 1e-6

    ranks = make_rank_ufun(u)  # ordinal ufun: worst=0 ... best=1
    # Repair a non-finite reserved value to ufun.min() - penalty:
    fixed, was_corrected = correct_reserved_value(float("nan"), u)


.. _tools-scenario:

Scenario (inout) tools
======================

A :class:`~negmas.inout.Scenario` bundles an outcome space, a tuple of utility
functions, and an (optional) mechanism configuration. It is the unit that gets
loaded from disk, transformed, analyzed, saved, and run.

Building and loading
--------------------

.. code-block:: python

    from negmas import make_issue
    from negmas.outcomes import make_os
    from negmas.preferences import LinearUtilityFunction, AffineUtilityFunction
    from negmas.inout import Scenario

    os = make_os([make_issue(11, "price"), make_issue(6, "quantity")])
    buyer = AffineUtilityFunction(
        weights=[-1.0, 1.0], bias=10.0, outcome_space=os, reserved_value=0.0
    )
    seller = LinearUtilityFunction(
        weights=[1.0, -1.0], outcome_space=os, reserved_value=0.0
    )
    scenario = Scenario(outcome_space=os, ufuns=(buyer, seller))

Scenarios can also be read from the common on-disk formats::

    Scenario.from_genius_folder(path)     # Genius .xml domain + ufun files
    Scenario.from_geniusweb_folder(path)  # GeniusWeb .json files
    Scenario.from_yaml_folder(path)       # NegMAS native YAML
    Scenario.load(folder)                 # auto-detect format

and written back with ``scenario.dumpas(folder, type="yml")`` /
``to_genius_folder(...)`` / ``to_yaml(...)``.

Transforming a scenario
-----------------------

These return a transformed scenario (recomputing stats by default):

.. code-block:: python

    normalized = scenario.normalize(to=(0.0, 1.0))  # common scale across ufuns
    independent = scenario.normalize(common_range=False)  # each ufun to its own [0, 1]
    smaller = scenario.discretize(levels=5)  # discretize the outcome space
    flat = scenario.to_single_issue()  # collapse to one enumerated issue
    no_disc = scenario.remove_discounting()  # strip discount factors

    assert normalized.is_normalized()  # checks max == 1 by default

.. note::

    ``Scenario.normalize`` normalizes all ufuns onto **one shared scale** by
    default (``common_range=True``), which keeps utilities comparable across
    negotiators. Pass ``common_range=False`` to normalize each ufun independently
    to its own ``[0, 1]``. (The older ``independent=`` keyword still works but is
    deprecated: ``independent=True`` == ``common_range=False``.)

Stats and checks
----------------

``calc_stats`` computes a :class:`~negmas.preferences.ops.ScenarioStats` bundle
with the Pareto frontier and the Nash/Kalai/etc. points precomputed:

.. code-block:: python

    stats = normalized.calc_stats()
    # stats.pareto_utils, stats.pareto_outcomes, stats.nash_utils, ...
    assert hasattr(stats, "pareto_utils") and hasattr(stats, "nash_utils")

    assert normalized.is_linear  # all ufuns are (affine) linear
    print(normalized.n_issues, normalized.n_negotiators)

Running a scenario
------------------

``make_session`` wires the scenario's ufuns to negotiators and returns a ready
mechanism you can ``run()``:

.. code-block:: python

    from negmas.sao import AspirationNegotiator

    session = normalized.make_session(
        [AspirationNegotiator(), AspirationNegotiator()], n_steps=50
    )
    result = session.run()


.. _tools-analysis:

Analysis tools
==============

The functions in :mod:`negmas.preferences.ops` describe the *geometry* of a
negotiation. They all take a sequence of ufuns and share a two-step idiom:
compute the **Pareto frontier** first, then locate special points on it.

Throughout, a "point" is a ``(utility_tuple, index)`` pair, where ``index`` refers
into the list of frontier outcomes.

The Pareto frontier
-------------------

:func:`~negmas.preferences.ops.pareto_frontier` returns
``(frontier_utils, frontier_indices)`` — the utility tuples of the
non-dominated outcomes and their indices into the outcome list:

.. code-block:: python

    from negmas import make_issue
    from negmas.preferences import LinearUtilityFunction, AffineUtilityFunction
    from negmas.preferences.ops import pareto_frontier

    price = make_issue(11, "price")  # 0..10
    buyer = AffineUtilityFunction(
        weights=[-1.0], bias=10.0, issues=[price], reserved_value=0.0
    ).normalize()
    seller = LinearUtilityFunction(
        weights=[1.0], issues=[price], reserved_value=0.0
    ).normalize()

    front_utils, front_index = pareto_frontier((buyer, seller), issues=[price])
    assert front_utils[0] == (1.0, 0.0) and front_utils[-1] == (0.0, 1.0)

Nash, Kalai and Kalai-Smorodinsky points
-----------------------------------------

Given the frontier, these locate the classic bargaining solutions:

.. code-block:: python

    from negmas.preferences.ops import (
        nash_points,
        kalai_points,
        ks_points,
        max_welfare_points,
    )

    nash = nash_points((buyer, seller), front_utils, issues=[price])
    kalai = kalai_points((buyer, seller), front_utils, issues=[price])
    ks = ks_points((buyer, seller), front_utils, issues=[price])
    welfare = max_welfare_points((buyer, seller), front_utils, issues=[price])

    # each is a tuple of ((u1, u2, ...), frontier_index) pairs
    nash_utils, nash_idx = nash[0]
    assert nash_utils == (0.5, 0.5)  # symmetric problem -> the midpoint

* :func:`~negmas.preferences.ops.nash_points` — maximizes the product of gains
  over the reserved values (the Nash bargaining solution).
* :func:`~negmas.preferences.ops.kalai_points` — the egalitarian point (equal
  gains); pass ``subtract_reserved_value=False`` to measure from 0 instead.
* :func:`~negmas.preferences.ops.ks_points` — the Kalai-Smorodinsky point.
* :func:`~negmas.preferences.ops.max_welfare_points` /
  :func:`~negmas.preferences.ops.max_relative_welfare_points` — maximum
  (relative) utilitarian welfare.

Conflict measures
-----------------

Scalar summaries of how opposed the preferences are, without needing the
frontier:

.. code-block:: python

    from negmas.preferences.ops import opposition_level, conflict_level, winwin_level

    opp = opposition_level((buyer, seller), outcomes=[(i,) for i in range(11)])
    conf = conflict_level(buyer, seller, outcomes=[(i,) for i in range(11)])
    ww = winwin_level(buyer, seller, outcomes=[(i,) for i in range(11)])

* :func:`~negmas.preferences.ops.opposition_level` — distance of the frontier
  from the ideal point (0 = fully cooperative, 1 = fully opposed).
* :func:`~negmas.preferences.ops.conflict_level` — correlation-based conflict
  between two ufuns (1 = perfectly opposed).
* :func:`~negmas.preferences.ops.winwin_level` — how much room there is for
  mutual gain.


See also
========

* :doc:`normalization` — the full specification of how ufuns are normalized.
* :doc:`base_modules` — the negotiation-facing side (mechanisms, negotiators).
* :doc:`api` — the complete API reference.
