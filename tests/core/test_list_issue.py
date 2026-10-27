"""Tests for `ListIssue` and `RecordIssue` (repeated issue groups)."""

from __future__ import annotations

import copy
import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from negmas.outcomes import (
    CartesianOutcomeSpace,
    DiscreteCartesianOutcomeSpace,
    DiscreteListValuesIssue,
    ListIssue,
    RecordIssue,
    make_issue,
    make_os,
)
from negmas.serialization import deserialize, serialize


def delivery_record():
    return RecordIssue(
        [
            make_issue(["PO-1", "PO-7"], "order_id"),
            make_issue((1, 100), "quantity"),
            make_issue((0, 30), "date"),
        ],
        name="delivery",
    )


def small_record():
    return RecordIssue([make_issue(["a", "b"], "x"), make_issue((0, 2), "y")])


# ---------------------------------------------------------------- RecordIssue


def test_record_issue_validates_per_field():
    r = small_record()
    assert r.is_valid(("a", 0))
    assert r.is_valid(["b", 2])
    assert not r.is_valid(("c", 0))
    assert not r.is_valid(("a", 3))
    assert not r.is_valid(("a",))
    assert not r.is_valid("a0")


def test_record_issue_cardinality_and_enumeration():
    r = small_record()
    vals = list(r.all)
    assert r.cardinality == 6 == len(vals) == len(set(vals))
    assert [r.value_at(i) for i in range(6)] == vals


def test_record_issue_rejects_duplicate_field_names():
    with pytest.raises(ValueError):
        RecordIssue([make_issue(2, "x"), make_issue(3, "x")])


def test_record_issue_with_continuous_field_is_continuous_and_infinite():
    r = RecordIssue([make_issue((0.0, 1.0), "p"), make_issue(3, "q")])
    assert r.is_continuous() and not r.is_discrete()
    assert r.cardinality == float("inf")
    d = r.to_discrete(5)
    assert d.is_discrete() and d.cardinality == 15


def test_record_issue_roundtrips():
    r = delivery_record()
    assert RecordIssue.from_dict(r.to_dict()) == r
    assert deserialize(serialize(r)) == r
    assert copy.deepcopy(r) == r and copy.copy(r) == r
    assert pickle.loads(pickle.dumps(r)) == r


# ---------------------------------------------------------------- ListIssue basics


def test_list_issue_unbounded_is_infinite_and_not_discrete():
    issue = ListIssue("lots", delivery_record())
    assert issue.cardinality == float("inf")
    assert not issue.is_discrete() and not issue.is_finite()
    assert not issue.is_continuous()
    with pytest.raises(ValueError):
        list(issue.all)


def test_list_issue_validity_counts_and_elements():
    issue = ListIssue("d", delivery_record(), min_count=1, max_count=3)
    assert not issue.is_valid(())
    assert issue.is_valid((("PO-1", 3, 4),))
    assert not issue.is_valid((("PO-1", 3, 4),) * 4)
    assert not issue.is_valid((("PO-9", 3, 4),))
    assert not issue.is_valid("PO-1")
    assert not issue.is_valid(None)


def test_list_issue_accepts_json_lists_in_any_order():
    issue = ListIssue("d", delivery_record())
    assert issue.is_valid([["PO-7", 50, 12], ["PO-1", 30, 12]])
    assert issue.canonicalize([["PO-7", 50, 12], ["PO-1", 30, 12]]) == (
        ("PO-1", 30, 12),
        ("PO-7", 50, 12),
    )


def test_unordered_canonicalization_is_order_free():
    issue = ListIssue("d", delivery_record())
    a = issue.canonicalize([("PO-7", 50, 12), ("PO-1", 30, 12), ("PO-1", 30, 12)])
    b = issue.canonicalize([("PO-1", 30, 12), ("PO-7", 50, 12), ("PO-1", 30, 12)])
    assert a == b and hash(a) == hash(b)


def test_ordered_canonicalization_keeps_order():
    issue = ListIssue("d", delivery_record(), ordered=True)
    a = issue.canonicalize([("PO-7", 50, 12), ("PO-1", 30, 12)])
    b = issue.canonicalize([("PO-1", 30, 12), ("PO-7", 50, 12)])
    assert a != b
    assert a == (("PO-7", 50, 12), ("PO-1", 30, 12))


def test_canonical_sort_falls_back_for_unorderable_values():
    issue = ListIssue("mixed", make_issue([1, "a", None], "v"))
    a = issue.canonicalize([None, "a", 1])
    b = issue.canonicalize([1, None, "a"])
    assert a == b


def test_unique_by_rejects_repeated_keys():
    issue = ListIssue("d", delivery_record(), unique_by=("order_id", "date"))
    assert issue.is_valid([("PO-1", 30, 12), ("PO-1", 20, 13), ("PO-7", 1, 12)])
    assert not issue.is_valid([("PO-1", 30, 12), ("PO-1", 20, 12)])


def test_unique_by_caps_the_count_and_makes_space_finite():
    issue = ListIssue("d", small_record(), unique_by="x")
    assert issue.effective_max_count == 2
    assert issue.is_finite()
    vals = list(issue.all)
    assert len(vals) == issue.cardinality == len(set(vals))
    # 1 empty + 2 keys x 3 y values + 1 (both keys) x 3 x 3
    assert issue.cardinality == 1 + 6 + 9


def test_unique_by_on_plain_element_means_unique_values():
    issue = ListIssue("v", make_issue(4, "v"), unique_by=True)
    assert issue.is_valid((0, 1, 3))
    assert not issue.is_valid((0, 0))
    same = ListIssue("v", make_issue(4, "v"), unique_by="v")
    assert same.unique_by == issue.unique_by


def test_unique_by_bad_field_raises():
    with pytest.raises(ValueError):
        ListIssue("d", small_record(), unique_by=("nope",))


def test_min_count_above_key_capacity_raises():
    with pytest.raises(ValueError):
        ListIssue("d", small_record(), min_count=3, unique_by="x")


@pytest.mark.parametrize("bad", [dict(min_count=-1), dict(min_count=2, max_count=1)])
def test_bad_counts_raise(bad):
    with pytest.raises(ValueError):
        ListIssue("d", small_record(), **bad)


def test_sequence_of_issues_becomes_a_record():
    issue = ListIssue("d", [make_issue(2, "a"), make_issue(2, "b")], max_count=1)
    assert isinstance(issue.element, RecordIssue)
    assert issue.field_names == ("a", "b")


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("unique", [None, "x", True])
@pytest.mark.parametrize("min_count,max_count", [(0, 0), (0, 2), (1, 3), (2, 2)])
def test_finite_cardinality_matches_enumeration(ordered, unique, min_count, max_count):
    try:
        issue = ListIssue(
            "d",
            small_record(),
            min_count=min_count,
            max_count=max_count,
            ordered=ordered,
            unique_by=unique,
        )
    except ValueError:
        # min_count larger than the number of unique keys
        assert unique == "x" and min_count > 2
        return
    vals = list(issue.all)
    assert len(vals) == issue.cardinality
    assert len(set(vals)) == len(vals)
    assert all(issue.is_valid(_) for _ in vals)
    assert all(issue.canonicalize(_) == _ for _ in vals)


# ---------------------------------------------------------------- rand / mutate


def test_rand_respects_bounds_and_unique_by():
    issue = ListIssue(
        "d",
        delivery_record(),
        min_count=2,
        max_count=40,
        unique_by=("order_id", "date"),
    )
    for _ in range(200):
        v = issue.rand()
        assert issue.is_valid(v)
        assert v == issue.canonicalize(v)


def test_rand_unbounded_has_useful_lengths():
    issue = ListIssue("d", delivery_record())
    lengths = [len(issue.rand()) for _ in range(500)]
    assert max(lengths) < 200
    assert 2 < sum(lengths) / len(lengths) < 10


def test_rand_typical_count_moves_the_mean():
    issue = ListIssue("d", delivery_record(), typical_count=30)
    mean = sum(len(issue.rand()) for _ in range(500)) / 500
    assert 20 < mean < 40


def test_rand_with_unique_continuous_key_uses_rejection():
    issue = ListIssue(
        "p", [make_issue((0.0, 1.0), "price"), make_issue(3, "q")], unique_by="price"
    )
    for _ in range(50):
        assert issue.is_valid(issue.rand())


def test_rand_invalid_is_invalid():
    issue = ListIssue("d", small_record(), min_count=1, max_count=2, unique_by="x")
    for _ in range(50):
        bad = issue.rand_invalid()
        assert bad is not None
        assert not issue.is_valid(bad)


def test_mutate_returns_valid_neighbour():
    issue = ListIssue(
        "d", delivery_record(), min_count=1, max_count=5, unique_by=("order_id", "date")
    )
    v = issue.rand()
    changed = 0
    for _ in range(300):
        w = issue.mutate(v)
        assert issue.is_valid(w) and w == issue.canonicalize(w)
        assert abs(len(w) - len(v)) <= 1
        changed += w != v
        v = w
    assert changed > 250


def test_mutate_on_singleton_space_returns_value():
    issue = ListIssue("d", make_issue(1, "v"), min_count=1, max_count=1)
    assert issue.mutate((0,)) == (0,)


# ---------------------------------------------------------------- views / OS


def test_value_generator_samples_infinite_issue():
    issue = ListIssue("d", delivery_record())
    vals = list(issue.value_generator(25))
    assert len(vals) == 25 == len(set(vals))
    with pytest.raises(ValueError):
        list(issue.value_generator(None))


def test_to_discrete_sampled_and_structural():
    issue = ListIssue("d", [make_issue((0.0, 1.0), "p"), make_issue(3, "q")])
    sampled = issue.to_discrete(7)
    assert isinstance(sampled, DiscreteListValuesIssue)
    assert sampled.cardinality == 7 and sampled.name == "d"
    assert all(issue.is_valid(_) for _ in sampled.all)
    assert all(len(_) == 2 for _ in sampled.rand_outcomes(5, with_replacement=True)[0])
    structural = issue.to_discrete(levels=3, max_count_cap=2)
    assert isinstance(structural, ListIssue) and structural.is_finite()
    assert structural.max_count == 2
    assert structural.element.cardinality == 9
    with pytest.raises(ValueError):
        issue.to_discrete(levels=3)  # still unbounded


def test_finite_small_issue_to_discrete_is_itself():
    issue = ListIssue("v", make_issue(2, "v"), max_count=1)
    assert issue.to_discrete(10) is issue


def test_make_os_with_infinite_list_issue_is_not_discrete():
    os = make_os([make_issue(3, "price"), ListIssue("d", delivery_record())])
    assert isinstance(os, CartesianOutcomeSpace)
    assert not isinstance(os, DiscreteCartesianOutcomeSpace)
    assert os.cardinality == float("inf")
    samples = list(os.sample(20, with_replacement=False, fail_if_not_enough=False))
    assert samples and all(os.is_valid(_) for _ in samples)
    assert os.is_valid(os.random_outcome())
    d = os.to_discrete(levels=5)
    assert d.cardinality == 15


def test_make_os_with_finite_list_issue_is_discrete():
    os = make_os(
        [make_issue(2, "price"), ListIssue("v", make_issue(2, "v"), max_count=2)]
    )
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    outcomes = list(os.enumerate())
    assert len(outcomes) == os.cardinality == 2 * (1 + 2 + 3)


def test_list_issue_roundtrips():
    issue = ListIssue(
        "d",
        delivery_record(),
        min_count=1,
        max_count=9,
        ordered=True,
        unique_by=("order_id", "date"),
        typical_count=3,
    )
    for other in (
        ListIssue.from_dict(issue.to_dict()),
        deserialize(serialize(issue)),
        copy.deepcopy(issue),
        copy.copy(issue),
        pickle.loads(pickle.dumps(issue)),
    ):
        assert other == issue
        assert other.typical_count == 3
        assert other.cardinality == issue.cardinality


def test_os_with_list_issue_roundtrips():
    os = make_os([make_issue(3, "price"), ListIssue("d", delivery_record())], name="x")
    assert deserialize(serialize(os)) == os


def test_genius_xml_export_raises_clearly():
    os = make_os([make_issue(3, "price"), ListIssue("d", delivery_record())])
    with pytest.raises(NotImplementedError, match="Genius"):
        os.to_xml_str()


def test_contains():
    big = ListIssue("d", delivery_record(), max_count=10)
    small = ListIssue(
        "d", delivery_record(), min_count=1, max_count=5, unique_by="date"
    )
    assert big.contains(small)
    assert not small.contains(big)
    assert small in big


# ---------------------------------------------------------------- properties

_field = st.sampled_from(
    [
        lambda n: make_issue(["a", "b", "c"], n),
        lambda n: make_issue((0, 3), n),
        lambda n: make_issue((0.0, 1.0), n),
        lambda n: make_issue([1, 5, 9], n),
    ]
)


@st.composite
def list_issues(draw):
    makers = draw(st.lists(_field, min_size=1, max_size=3))
    fields = [m(f"f{i}") for i, m in enumerate(makers)]
    record = draw(st.booleans()) or len(fields) > 1
    element = RecordIssue(fields) if record else fields[0]
    names = [f.name for f in fields]
    unique = draw(
        st.one_of(st.none(), st.lists(st.sampled_from(names), min_size=1, unique=True))
    )
    min_count = draw(st.integers(0, 2))
    max_count = draw(st.one_of(st.none(), st.integers(min_count, min_count + 6)))
    if unique is not None and not record:
        unique = True
    try:
        return ListIssue(
            "L",
            element,
            min_count=min_count,
            max_count=max_count,
            ordered=draw(st.booleans()),
            unique_by=unique,
        )
    except ValueError:
        return ListIssue("L", element, max_count=max_count)


@settings(max_examples=150, deadline=None)
@given(list_issues())
def test_property_rand_is_valid_and_canonical(issue):
    for _ in range(5):
        v = issue.rand()
        assert issue.is_valid(v)
        assert issue.canonicalize(v) == v
        assert issue.canonicalize(list(reversed(v))) == v or issue.ordered


@settings(max_examples=150, deadline=None)
@given(list_issues())
def test_property_mutate_keeps_validity(issue):
    v = issue.rand()
    for _ in range(10):
        v = issue.mutate(v)
        assert issue.is_valid(v)
        assert issue.canonicalize(v) == v


@settings(max_examples=100, deadline=None)
@given(list_issues())
def test_property_serialization_roundtrips(issue):
    assert ListIssue.from_dict(issue.to_dict()) == issue
    assert deserialize(serialize(issue)) == issue
    v = issue.rand()
    # the positional JSON wire form: a list of lists
    as_json = [list(e) if isinstance(e, tuple) else e for e in v]
    assert issue.is_valid(as_json)
    assert issue.canonicalize(as_json) == v


@settings(max_examples=100, deadline=None)
@given(list_issues())
def test_property_random_lists_are_valid_iff_constraints_hold(issue):
    element = issue.element
    n = issue.min_count + 1
    elements = [element.rand() for _ in range(n)]
    keys = [issue.key_of(issue.canonicalize([e])[0]) for e in elements]
    expected = (issue.max_count is None or n <= issue.max_count) and (
        issue.unique_by is None or len(set(keys)) == len(keys)
    )
    assert issue.is_valid(elements) == expected
