Open-ended outcome spaces
=========================

Some negotiations have a part whose size is not fixed: any number of lots,
each with a price and a quantity, or a free-text note. NegMAS models these with
issues whose value can be any of infinitely many values. The outcome keeps a
fixed number of issues, so mechanisms work unchanged. Such spaces are never
enumerated: they are sampled and searched.

List issues
-----------

A `ListIssue` holds a variable-length list of elements. An element is any issue,
or a `RecordIssue`: a tuple of named fields.

.. code-block:: python

    from negmas.outcomes import ListIssue, RecordIssue, make_issue, make_os

    lot = RecordIssue([make_issue((1, 20), "price"), make_issue((1, 10), "quantity")])
    lots = ListIssue("lots", lot, min_count=1)  # any number of lots
    os = make_os([make_issue(["fast", "slow"], "shipping"), lots])

    assert lots.cardinality == float("inf")
    assert lots.is_valid([(5, 2), (7, 1)])
    assert os.is_valid(("fast", ((5, 2), (7, 1))))

- ``max_count`` caps the number of elements. Without it the issue is infinite.
- ``ordered=False`` (the default) makes the list a multiset: values are
  canonicalized by sorting, so the order of elements does not matter. Use
  ``ordered=True`` for sequences.
- ``unique_by`` names fields whose combination must not repeat, e.g.
  ``unique_by=("price",)``.
- ``is_valid`` accepts lists (e.g. JSON) in any order. ``canonicalize`` returns
  the canonical value.
- ``rand()`` samples a value and ``mutate(value)`` returns a neighbour (an
  element added, removed or changed). ``to_discrete(n)`` gives ``n`` sampled
  values, and ``to_discrete(levels=..., max_count_cap=...)`` gives a finite
  list issue.

Utility functions over lists
----------------------------

A list is scored by *lifting* an element scorer: score each element, aggregate
the scores, and optionally add a function of the count.

.. code-block:: python

    from negmas.preferences import (
        LinearAdditiveUtilityFunction,
        ListAggregationFun,
        RecordAdditiveFun,
    )
    from negmas.preferences.value_fun import AffineFun, LinearFun, TableFun

    lot_value = RecordAdditiveFun((AffineFun(-1.0, bias=20.0), AffineFun(1.0)))
    ufun = LinearAdditiveUtilityFunction(
        [
            TableFun({"fast": 1.0, "slow": 0.0}),
            ListAggregationFun(lot_value, aggregate="mean"),
        ],
        outcome_space=os,
    )
    assert ufun(("fast", ((5, 2),))) == 1.0 + 17.0
    ufun = ufun.normalize()

    # a small cost per lot needs a bounded count to have a minimum
    capped = ListIssue("lots", lot, min_count=1, max_count=5)
    f = ListAggregationFun(lot_value, aggregate="sum", count_fun=LinearFun(-0.1))
    assert f.minmax(capped) == (0.9, 144.5)

The element scorer can be:

- a value function (for a plain element issue);
- a `RecordAdditiveFun` (one function per field);
- any multi-issue function, e.g. ``LambdaMultiFun``, over a record's fields;
- any utility function defined over ``make_os(record.fields)``;
- a callable.

``aggregate`` is ``"sum"``, ``"mean"``, ``"min"``, ``"max"`` or a callable.
``count_fun`` is a soft alternative to a hard ``max_count``.

`ListAggregationMultiFun` lifts a multi-issue function that also reads other
issues, as a `GLAUtilityFunction` factor. `lift_element_ufun` converts a ufun
written over plain issues plus *one* copy of the element's fields into a ufun
over the plain issues plus the list issue:

.. code-block:: python

    from negmas.preferences import lift_element_ufun

    flat = make_os([make_issue(["fast", "slow"], "shipping"), *lot.fields])
    one_lot = LinearAdditiveUtilityFunction(
        [TableFun({"fast": 1.0, "slow": 0.0}), AffineFun(-1.0, bias=20.0), AffineFun(1.0)],
        outcome_space=flat,
    )
    many_lots = lift_element_ufun(one_lot, ListIssue("lots", lot, max_count=5))
    assert many_lots(("fast", ((5, 2),))) == one_lot(("fast", 5, 2))

**Extremes and normalization.** The best list is built greedily: the
best-scoring feasible elements, one per ``unique_by`` key, for the best count.
``minmax`` is exact for mean/min/max. A sum over an unbounded count usually has
no maximum, so it raises `UnboundedListUtilityError`. Set ``max_count``, use
``aggregate="mean"``, or add a ``count_fun`` that makes extra elements
unprofitable.

Negotiating
-----------

Inverters and Pareto samplers search list spaces instead of enumerating them.
`AdaptiveInverseUtilityFunction`, the default, picks
`SearchInverseUtilityFunction`. The built-in time-based negotiators therefore
work as usual:

.. code-block:: python

    from negmas.sao import SAOMechanism
    from negmas.sao.negotiators import BoulwareTBNegotiator

    seller = LinearAdditiveUtilityFunction(
        [
            TableFun({"fast": 0.0, "slow": 1.0}),
            ListAggregationFun(
                RecordAdditiveFun((AffineFun(1.0), AffineFun(0.5))), aggregate="mean"
            ),
        ],
        outcome_space=os,
    ).normalize()
    m = SAOMechanism(outcome_space=os, n_steps=100)
    m.add(BoulwareTBNegotiator(name="buyer"), ufun=ufun)
    m.add(BoulwareTBNegotiator(name="seller"), ufun=seller)
    m.run()
    assert m.agreement is not None

Strategies that rank every outcome (e.g. ``WARNegotiator``) do not support
infinite spaces, and Genius cannot export list issues.

Infinite integers, reals and strings
------------------------------------

.. code-block:: python

    from negmas.outcomes import IntIssue, RealIssue, StrIssue

    quantity = IntIssue("quantity", min=1)  # any positive integer
    price = RealIssue("price", min=0.0, typical=(10.0, 50.0))
    note = StrIssue("note", max_length=200)

    assert quantity.is_valid(10**9) and not quantity.is_valid(0)
    assert 1 <= quantity.rand()

`IntIssue` and `RealIssue` are factories for `CountableInfiniteIssue` and
`ContinuousInfiniteIssue`. With both bounds given, they return a finite range.
Sampling draws mostly from the ``typical`` range, which defaults to 100 values
next to the finite bound (or around 0). `StrIssue` holds any string within the
length bounds; score it with a callable (e.g. ``LambdaFun``). None of these
issues is enumerated, and normalizing a value function that is unbounded over
one of them raises a clear error.
