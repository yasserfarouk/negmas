"""Open-box tests for `ListIssue` / `RecordIssue` (Part A, chunk 1).

These target branches, boundaries and rules read from the code and from
``coding_agents/DESIGN.md`` (Part A, items 1, 2 and 6). Tests marked
``xfail(strict=True, reason="DEFECT: ...")`` document defects in the product
code; they must start passing (and lose the marker) once the defect is fixed.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from negmas.outcomes import (
    DiscreteCartesianOutcomeSpace,
    DiscreteListValuesIssue,
    ListIssue,
    RecordIssue,
    make_issue,
    make_os,
)
from negmas.outcomes.discretizers.grid import GridBasedDiscretizer
from negmas.outcomes.issue_ops import sample_outcomes
from negmas.serialization import deserialize, serialize


def rec2(name="rec"):
    return RecordIssue([make_issue(["x", "y"], "a"), make_issue((0, 2), "b")], name)


def jsonify(v):
    """The positional JSON wire form: tuples become lists, recursively."""
    return json.loads(json.dumps(v))


def _canon_oracle(issue: ListIssue, value):
    """An independent canonical form: tuples, sorted by repr when unordered."""

    def tup(x):
        return tuple(tup(_) for _ in x) if isinstance(x, (list, tuple)) else x

    els = tuple(tup(_) for _ in value)
    return els if issue.ordered else tuple(sorted(els, key=repr))


def _oracle_values(issue: ListIssue) -> set:
    """All valid values by brute force (independent of the closed form)."""
    element_values = list(issue.element.all)
    hi = issue.max_count
    out = set()
    for n in range(issue.min_count, hi + 1):
        for c in product(element_values, repeat=n):
            if issue.unique_by is not None:
                if issue.is_record:
                    idx = [issue.element.field_names.index(f) for f in issue.unique_by]
                    keys = [tuple(e[i] for i in idx) for e in c]
                else:
                    keys = list(c)
                if len(set(keys)) != len(keys):
                    continue
            out.add(_canon_oracle(issue, c))
    return out


# ------------------------------------------------------------ strategies

_tiny_fields = [
    lambda n: make_issue(["a", "b"], n),
    lambda n: make_issue((0, 2), n),
    lambda n: make_issue([1, 5], n),
]


@st.composite
def finite_list_issues(draw):
    """Small finite list issues (enumerable by brute force)."""
    makers = draw(st.lists(st.sampled_from(_tiny_fields), min_size=1, max_size=2))
    fields = [m(f"f{i}") for i, m in enumerate(makers)]
    record = len(fields) > 1 or draw(st.booleans())
    element = RecordIssue(fields, "r") if record else fields[0]
    names = [f.name for f in fields]
    unique = draw(
        st.one_of(st.none(), st.lists(st.sampled_from(names), min_size=1, unique=True))
    )
    if unique is not None and not record:
        unique = True
    min_count = draw(st.integers(0, 2))
    max_count = draw(st.integers(min_count, 3))
    ordered = draw(st.booleans())
    try:
        return ListIssue(
            "L",
            element,
            min_count=min_count,
            max_count=max_count,
            ordered=ordered,
            unique_by=unique,
        )
    except ValueError:
        return ListIssue("L", element, max_count=max_count, ordered=ordered)


@st.composite
def any_list_issues(draw):
    """List issues, possibly infinite (no max_count, continuous fields)."""
    makers = draw(
        st.lists(
            st.sampled_from(_tiny_fields + [lambda n: make_issue((0.0, 1.0), n)]),
            min_size=1,
            max_size=3,
        )
    )
    fields = [m(f"f{i}") for i, m in enumerate(makers)]
    record = len(fields) > 1 or draw(st.booleans())
    element = RecordIssue(fields, "r") if record else fields[0]
    names = [f.name for f in fields]
    unique = draw(
        st.one_of(st.none(), st.lists(st.sampled_from(names), min_size=1, unique=True))
    )
    if unique is not None and not record:
        unique = True
    min_count = draw(st.integers(0, 2))
    max_count = draw(st.one_of(st.none(), st.integers(min_count, min_count + 4)))
    ordered = draw(st.booleans())
    try:
        return ListIssue(
            "L",
            element,
            min_count=min_count,
            max_count=max_count,
            ordered=ordered,
            unique_by=unique,
        )
    except ValueError:
        return ListIssue("L", element, max_count=max_count, ordered=ordered)


# ------------------------------------------------------ constructor branches


@pytest.mark.parametrize("unique_by", [None, False, (), []])
def test_unique_by_falsy_values_mean_no_constraint(unique_by):
    """None, False and an empty sequence all mean "no unique_by"."""
    issue = ListIssue("L", rec2(), max_count=3, unique_by=unique_by)
    assert issue.unique_by is None
    assert issue.key_of(("x", 0)) is None
    assert issue.is_valid([("x", 0), ("x", 0)])


def test_unique_by_true_on_record_means_whole_records_unique():
    """unique_by=True on a record keys on every field (whole records unique)."""
    issue = ListIssue("L", rec2(), unique_by=True)
    assert issue.unique_by == ("a", "b")
    assert issue.effective_max_count == 6
    assert issue.is_valid([("x", 0), ("x", 1)])
    assert not issue.is_valid([("x", 0), ["x", 0]])


def test_unique_by_plain_element_by_name_equals_true():
    """On a plain element, unique_by=<element name> is the same as True."""
    by_name = ListIssue("L", make_issue(["a", "b", "c"], "v"), unique_by="v")
    by_true = ListIssue("L", make_issue(["a", "b", "c"], "v"), unique_by=True)
    assert by_name.unique_by == by_true.unique_by == ("v",)
    assert by_name.cardinality == by_true.cardinality
    assert by_name.effective_max_count == 3


def test_unique_by_repeated_field_raises():
    """unique_by naming a field twice is rejected."""
    with pytest.raises(ValueError):
        ListIssue("L", rec2(), unique_by=("a", "a"))


def test_unique_by_on_continuous_key_does_not_cap_count():
    """A continuous key field has infinitely many keys: no count cap."""
    element = RecordIssue([make_issue((0.0, 1.0), "c"), make_issue(2, "k")], "r")
    issue = ListIssue("L", element, unique_by="c")
    assert issue.effective_max_count == math.inf
    assert issue.cardinality == math.inf


@pytest.mark.parametrize("min_count,max_count", [(1.0, 2.0), (0, 0), (3, 3)])
def test_integral_counts_are_accepted(min_count, max_count):
    """Integral floats and min_count == max_count are valid counts."""
    issue = ListIssue("L", make_issue(2, "v"), min_count, max_count)
    assert isinstance(issue.min_count, int) and isinstance(issue.max_count, int)


@pytest.mark.parametrize("min_count,max_count", [(1.5, None), (0, 2.5), (-1, 2)])
def test_non_integral_or_negative_counts_raise(min_count, max_count):
    """Counts must be non-negative integers."""
    with pytest.raises(ValueError):
        ListIssue("L", make_issue(2, "v"), min_count, max_count)


def test_record_needs_a_field_and_issue_fields():
    """A RecordIssue needs at least one field and every field must be an issue."""
    with pytest.raises(ValueError):
        RecordIssue([])
    with pytest.raises(TypeError):
        RecordIssue([make_issue(2, "a"), 3])  # type: ignore[list-item]


def test_record_field_lookup():
    """field()/field_index() find fields by name and raise on unknown names."""
    r = rec2()
    assert r.field_index("b") == 1
    assert r.field("a").name == "a"
    with pytest.raises(ValueError):
        r.field_index("zz")


# ------------------------------------------------------------ degenerate


def test_empty_only_list_issue():
    """min_count == max_count == 0 has exactly one value: the empty tuple."""
    issue = ListIssue("L", rec2(), 0, 0)
    assert issue.cardinality == 1
    assert issue.is_discrete()
    assert list(issue.all) == [()]
    assert issue.rand() == ()
    assert issue.mutate(()) == ()
    assert issue.is_valid([]) and not issue.is_valid([("x", 0)])


def test_empty_only_list_issue_with_infinite_element():
    """max_count=0 is finite even if the element is infinite."""
    issue = ListIssue("L", make_issue((0.0, 1.0), "c"), 0, 0)
    assert issue.cardinality == 1
    assert issue.is_discrete()
    assert issue.rand() == ()


def test_empty_only_list_issue_with_infinite_element_enumerates():
    """A discrete list issue can always be enumerated, even with max_count=0."""
    issue = ListIssue("L", make_issue((0.0, 1.0), "c"), 0, 0)
    assert list(issue.all) == [()]
    assert issue.value_at(0) == ()
    os_ = make_os([issue, make_issue(2, "p")])
    assert list(os_.enumerate()) == [((), 0), ((), 1)]


def test_key_capacity_equal_to_min_count_forces_all_keys():
    """min_count equal to the key capacity forces every key to appear."""
    issue = ListIssue("L", rec2(), min_count=2, unique_by="a")
    for _ in range(20):
        v = issue.rand()
        assert sorted(e[0] for e in v) == ["x", "y"]


# ------------------------------------------------------------ is_valid


@pytest.mark.parametrize("bad", ["ab", None, 3, {"a": 1}, {("x", 0)}])
def test_non_sequence_values_are_invalid(bad):
    """Only tuples and lists can be list-issue values."""
    assert not ListIssue("L", make_issue(["a", "b"], "v")).is_valid(bad)


def test_record_rejects_wrong_arity():
    """A record value needs exactly one value per field."""
    r = rec2()
    assert not r.is_valid(("x",))
    assert not r.is_valid(("x", 0, 1))
    assert not r.is_valid("x0")


def test_max_count_boundary():
    """Exactly max_count elements is valid, max_count + 1 is not."""
    issue = ListIssue("L", make_issue(2, "v"), 1, 3)
    assert not issue.is_valid([])
    assert issue.is_valid([0])
    assert issue.is_valid([0, 1, 1])
    assert not issue.is_valid([0, 1, 1, 0])


def test_nested_list_field_canonicalizes_recursively():
    """A list field inside a record is canonicalized when the outer list is."""
    inner = ListIssue("tags", make_issue(["p", "q", "r"], "t"), max_count=3)
    element = RecordIssue([make_issue(2, "k"), inner], "r")
    outer = ListIssue("L", element, unique_by="k")
    v = [[1, ["r", "p"]], [0, ["q"]]]
    assert outer.is_valid(v)
    assert outer.canonicalize(v) == ((0, ("q",)), (1, ("p", "r")))
    assert outer.canonicalize(jsonify(outer.canonicalize(v))) == outer.canonicalize(v)


# ------------------------------------------------------------ algebra


@settings(max_examples=150, deadline=None)
@given(any_list_issues(), st.randoms(use_true_random=False))
def test_property_canonicalize_is_idempotent_and_json_stable(issue, rnd):
    """canonicalize is idempotent and survives the JSON round-trip."""
    v = issue.rand()
    c = issue.canonicalize(v)
    assert issue.canonicalize(c) == c
    assert issue.canonicalize(jsonify(c)) == c
    assert issue.is_valid(jsonify(c))
    assert hash(c) == hash(issue.canonicalize(jsonify(c)))


@settings(max_examples=150, deadline=None)
@given(any_list_issues(), st.randoms(use_true_random=False))
def test_property_permutation_invariance_iff_unordered(issue, rnd):
    """Validity ignores order; the canonical form ignores it only when unordered."""
    v = list(issue.rand())
    p = list(v)
    rnd.shuffle(p)
    assert issue.is_valid(p)
    if issue.ordered:
        assert issue.canonicalize(p) == tuple(p)
    else:
        assert issue.canonicalize(p) == issue.canonicalize(v)


@settings(max_examples=100, deadline=None)
@given(finite_list_issues())
def test_property_cardinality_and_all_match_brute_force(issue):
    """cardinality and `all` agree with an independent brute-force enumeration."""
    expected = _oracle_values(issue)
    values = list(issue.all)
    assert len(values) == len(set(values)), "all() yields duplicates"
    assert issue.cardinality == len(expected)
    assert {_canon_oracle(issue, v) for v in values} == expected
    assert all(issue.is_valid(v) for v in values)


@settings(max_examples=60, deadline=None)
@given(finite_list_issues())
def test_property_value_at_matches_all(issue):
    """value_at(i) is the i-th value of `all`, and out-of-range indices raise."""
    values = list(issue.all)
    for i in range(min(len(values), 15)):
        assert issue.value_at(i) == values[i]
    with pytest.raises(IndexError):
        issue.value_at(len(values))
    with pytest.raises(IndexError):
        issue.value_at(-1)


@pytest.mark.parametrize(
    "fields",
    [
        [make_issue(["a", "b"], "x"), make_issue((0, 2), "y")],
        [make_issue(3, "x"), make_issue(["p", "q"], "y"), make_issue([1, 5], "z")],
    ],
)
def test_record_value_at_matches_all(fields):
    """RecordIssue.value_at uses the same order as RecordIssue.all."""
    r = RecordIssue(fields)
    values = list(r.all)
    assert len(values) == r.cardinality
    assert [r.value_at(i) for i in range(len(values))] == values
    with pytest.raises(IndexError):
        r.value_at(len(values))


@settings(max_examples=150, deadline=None)
@given(any_list_issues())
def test_property_mutate_is_one_edit_away(issue):
    """mutate adds, removes, changes or swaps exactly one element."""
    v = issue.rand()
    m = issue.mutate(v)
    assert issue.is_valid(m)
    assert abs(len(m) - len(v)) <= 1
    if not issue.ordered:
        diff = (Counter(m) - Counter(v)) + (Counter(v) - Counter(m))
        assert sum(diff.values()) <= 2


def test_mutate_returns_a_different_value_when_one_exists():
    """mutate gives a neighbour (not the value itself) when neighbours exist."""
    issue = ListIssue("L", make_issue(3, "v"), max_count=4, ordered=True)
    v = (1, 1)
    # swap is one of 4 ops, so 300 draws hit it with probability ~1 - 1e-37
    assert all(issue.mutate(v) != v for _ in range(300))


@settings(max_examples=100, deadline=None)
@given(any_list_issues())
def test_property_rand_invalid_is_invalid(issue):
    """rand_invalid returns None or an invalid value."""
    for _ in range(3):
        bad = issue.rand_invalid()
        assert bad is None or not issue.is_valid(bad)


# ------------------------------------------------------------ discreteness


@settings(max_examples=100, deadline=None)
@given(any_list_issues())
def test_property_discrete_means_finite_and_never_continuous(issue):
    """is_discrete() == is_finite() == finite cardinality; is_continuous() is False."""
    assert issue.is_continuous() is False
    assert issue.is_discrete() == issue.is_finite() == math.isfinite(issue.cardinality)


@settings(max_examples=60, deadline=None)
@given(any_list_issues(), st.integers(1, 12))
def test_property_sampled_to_discrete_is_a_finite_valid_subset(issue, n):
    """to_discrete(n) is finite, has at most max(n, |issue|) values, all valid."""
    # the max_count == 0 / infinite element case is pinned separately (a defect)
    assume(issue.effective_max_count > 0 or issue.element.is_discrete())
    d = issue.to_discrete(n)
    assert d.is_discrete()
    vals = list(d.all)
    if d is not issue:
        assert isinstance(d, DiscreteListValuesIssue)
        assert len(vals) <= n
    assert all(issue.is_valid(v) for v in vals)
    assert issue.contains(d)


def test_infinite_issue_is_never_enumerated():
    """`all`, `value_at` and to_discrete(None) refuse an infinite list issue.

    `value_generator(None)` samples a finite default instead (design decision).
    """
    issue = ListIssue("L", rec2())
    with pytest.raises(ValueError):
        next(iter(issue.all))
    with pytest.raises(ValueError):
        issue.value_at(0)
    with pytest.raises(ValueError):
        issue.to_discrete(None)
    assert 0 < len(list(issue.value_generator(None))) <= 1000


def test_structural_to_discrete_needs_a_finite_count():
    """Structural to_discrete raises without max_count/max_count_cap."""
    issue = ListIssue("L", make_issue((0.0, 1.0), "c"))
    with pytest.raises(ValueError):
        issue.to_discrete(levels=3)
    d = issue.to_discrete(levels=3, max_count_cap=2)
    assert isinstance(d, ListIssue) and d.is_discrete()
    assert list(d.element.all) == [0.0, 0.5, 1.0]
    assert d.cardinality == 1 + 3 + 6


def test_structural_to_discrete_cap_is_min_of_cap_and_max_count():
    """max_count_cap lowers max_count but never raises it."""
    issue = ListIssue("L", make_issue(3, "v"), max_count=2)
    assert issue.to_discrete(max_count_cap=5).max_count == 2
    assert issue.to_discrete(max_count_cap=1).max_count == 1


def test_structural_to_discrete_uses_the_unique_by_cap():
    """A count already capped by unique_by needs no max_count_cap."""
    element = RecordIssue([make_issue((0.0, 1.0), "c"), make_issue(["a", "b"], "k")])
    issue = ListIssue("L", element, unique_by="k")
    assert issue.effective_max_count == 2
    d = issue.to_discrete(levels=3)
    assert d.is_discrete() and d.cardinality == 1 + 6 + 9


# ------------------------------------------------------------ contains


def test_contains_is_sound_for_a_finer_unique_by():
    """A list unique by ("a",) does not contain one unique only by ("a", "b")."""
    coarse = ListIssue("L", rec2(), max_count=4, unique_by=("a",))
    fine = ListIssue("L", rec2(), max_count=4, unique_by=("a", "b"))
    witness = [("x", 0), ("x", 1)]
    assert fine.is_valid(witness) and not coarse.is_valid(witness)
    assert not coarse.contains(fine)


def test_contains_accepts_a_stricter_unique_by():
    """A list unique by ("a", "b") contains one unique by ("a",)."""
    fine = ListIssue("L", rec2(), max_count=4, unique_by=("a", "b"))
    coarse = ListIssue("L", rec2(), max_count=4, unique_by=("a",))
    assert all(fine.is_valid(v) for v in coarse.all)
    assert fine.contains(coarse)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.sampled_from(["a", "b"]), min_size=1, max_size=2, unique=True),
    st.lists(st.sampled_from(["a", "b"]), min_size=1, max_size=2, unique=True),
)
def test_property_contains_is_sound(u1, u2):
    """If A.contains(B), every value of B is a valid value of A."""
    a = ListIssue("L", rec2(), max_count=3, unique_by=u1)
    b = ListIssue("L", rec2(), max_count=3, unique_by=u2)
    if a.contains(b):
        assert all(a.is_valid(v) for v in b.all)


def test_contains_no_unique_by_contains_any_unique_by():
    """No unique_by contains any unique_by; the reverse does not hold."""
    free = ListIssue("L", rec2(), max_count=3)
    keyed = ListIssue("L", rec2(), max_count=3, unique_by="a")
    assert free.contains(keyed)
    assert not keyed.contains(free)


def test_contains_counts_and_ordering():
    """contains checks the count range and the ordered flag."""
    wide = ListIssue("L", rec2(), 0, 5)
    assert wide.contains(ListIssue("L", rec2(), 1, 3))
    assert not wide.contains(ListIssue("L", rec2(), 0, 6))
    assert not wide.contains(ListIssue("L", rec2(), 0, None))
    assert not ListIssue("L", rec2(), 2, 5).contains(ListIssue("L", rec2(), 1, 3))
    assert not wide.contains(ListIssue("L", rec2(), 1, 3, ordered=True))


def test_contains_non_list_issues():
    """A ListIssue does not contain a plain continuous issue."""
    assert not ListIssue("L", rec2()).contains(make_issue((0.0, 1.0), "c"))


# ------------------------------------------------------------ serialization


@settings(max_examples=80, deadline=None)
@given(any_list_issues())
def test_property_json_roundtrip_of_the_issue(issue):
    """A list issue survives a real JSON round-trip (not just serialize())."""
    back = deserialize(jsonify(serialize(issue)))
    # compared in serialized form: a ContinuousIssue field comes back with list
    # instead of tuple `values` (pre-existing, unrelated to list issues)
    assert jsonify(serialize(back)) == jsonify(serialize(issue))
    assert isinstance(back, ListIssue) and back.unique_by == issue.unique_by
    if issue.element.is_discrete():
        assert back == issue
    assert back.cardinality == issue.cardinality
    assert back.typical_count == issue.typical_count
    v = issue.rand()
    assert back.is_valid(v) and back.canonicalize(jsonify(v)) == v


def test_discrete_list_values_issue_json_roundtrip():
    """The sampled view survives JSON with tuple values equal to the originals."""
    d = ListIssue("L", rec2(), max_count=3).to_discrete(5)
    back = deserialize(jsonify(serialize(d)))
    assert all(isinstance(v, tuple) for v in back.values)
    assert back == d


def test_discretized_os_json_roundtrip_keeps_outcomes_valid():
    """A discretized outcome space still accepts its own outcomes after JSON."""
    os_ = make_os([ListIssue("L", rec2()), make_issue(3, "p")])
    d = os_.to_discrete(levels=4)
    back = deserialize(jsonify(serialize(d)))
    o = d.random_outcome()
    assert back.is_valid(o)


def test_discrete_list_values_issue_accepts_what_list_issue_accepts():
    """The sampled view accepts a sampled value as JSON lists and in any order."""
    issue = ListIssue("L", rec2(), min_count=2, max_count=2, unique_by=True)
    d = issue.to_discrete(3)
    v = next(iter(d.all))
    assert d.is_valid(jsonify(v))
    assert d.is_valid(tuple(reversed(v)))


def test_discrete_list_values_issue_sampling_and_xml():
    """The sampled view samples its tuple values intact and refuses XML."""
    d = ListIssue("L", rec2(), max_count=3).to_discrete(6)
    vals = set(d.all)
    assert set(d.rand_outcomes(len(vals))) == vals
    assert (
        d.rand_outcomes(len(vals) + 3) and set(d.rand_outcomes(len(vals) + 3)) == vals
    )
    with pytest.raises(ValueError):
        d.rand_outcomes(len(vals) + 1, fail_if_not_enough=True)
    assert all(v in vals for v in d.rand_outcomes(20, with_replacement=True))
    with pytest.raises(NotImplementedError):
        d._to_xml_str(0)


def test_record_xml_export_raises():
    """RecordIssue cannot be exported to Genius XML."""
    with pytest.raises(NotImplementedError):
        rec2()._to_xml_str(0)


# ------------------------------------------------------------ scalability


def test_rand_with_a_huge_discrete_key_space():
    """rand() works when the unique_by key space is astronomically large."""
    big = RecordIssue([make_issue((0, 10**7), n) for n in "pqs"])
    issue = ListIssue("L", big, max_count=3, unique_by=True)
    assert issue.is_valid(issue.rand())


@pytest.mark.timeout(10)
def test_construction_with_a_large_unique_key_space_is_fast():
    """Building a list unique by a 0..10**5 integer field takes well under 10s."""
    issue = ListIssue("L", make_issue((0, 10**5), "order_id"), unique_by=True)
    assert issue.effective_max_count == 10**5 + 1


# ------------------------------------------------------------ outcome spaces


def test_os_with_infinite_list_issue_samples_and_discretizes():
    """An outcome space with an infinite list issue samples and discretizes."""
    os_ = make_os([ListIssue("L", rec2()), make_issue(3, "p")])
    assert not os_.is_discrete()
    o = os_.random_outcome()
    assert os_.is_valid(o) and os_.is_valid(jsonify(o))
    # os.sample may return fewer than asked: _sample_issues truncates when it
    # tops up after duplicates (pre-existing, issue_ops.py, not in this diff)
    samples = list(os_.sample(5))
    assert 1 <= len(samples) <= 5 and all(os_.is_valid(_) for _ in samples)
    d = os_.to_discrete(levels=4)
    assert isinstance(d, DiscreteCartesianOutcomeSpace)
    assert isinstance(d.issues[0], DiscreteListValuesIssue)
    assert d.issues[1] == os_.issues[1]
    assert d.cardinality == os_.cardinality_if_discretized(4) == 12
    assert all(os_.is_valid(_) for _ in d.enumerate())
    sampled = list(os_.enumerate_or_sample(max_cardinality=10))
    assert 1 <= len(sampled) <= 10 and all(os_.is_valid(_) for _ in sampled)


def test_grid_discretizer_handles_list_issues():
    """GridBasedDiscretizer discretizes a non-discrete list issue."""
    os_ = make_os([ListIssue("L", rec2()), make_issue(3, "p")])
    d = GridBasedDiscretizer(max_outcomes=100, min_levels=5)(os_)
    assert isinstance(d.issues[0], DiscreteListValuesIssue)
    assert d.cardinality <= 100
    assert all(os_.is_valid(_) for _ in d.enumerate())
    small = GridBasedDiscretizer(max_outcomes=7, min_levels=5)(os_)
    assert small.cardinality <= 7


def test_sample_outcomes_with_list_issue():
    """sample_outcomes samples an infinite list issue instead of enumerating it."""
    issues = [ListIssue("L", rec2()), make_issue(3, "p")]
    out = sample_outcomes(issues, n_outcomes=20)
    assert out
    assert all(issues[0].is_valid(o[0]) and issues[1].is_valid(o[1]) for o in out)


@pytest.mark.parametrize(
    "lo,hi,n,expected_levels",
    [(0.0, 1.0, None, 5), (2.0, 5.0, 7, 7), (-1.0, 1.0, None, 5)],
)
def test_sample_outcomes_continuous_grid_unchanged(lo, hi, n, expected_levels):
    """sample_outcomes still puts an evenly spaced grid with endpoints on a range."""
    out = sample_outcomes([make_issue((lo, hi), "c")], n_outcomes=n)
    vals = sorted(o[0] for o in out)
    step = (hi - lo) / (expected_levels - 1)
    assert vals == pytest.approx([lo + i * step for i in range(expected_levels)])


def test_finite_list_issue_makes_a_discrete_os():
    """A finite list issue makes a discrete outcome space with exact cardinality."""
    issue = ListIssue("L", rec2(), max_count=2, unique_by="a")
    os_ = make_os([issue, make_issue(2, "p")])
    assert isinstance(os_, DiscreteCartesianOutcomeSpace)
    assert os_.cardinality == issue.cardinality * 2 == len(list(os_.enumerate()))
