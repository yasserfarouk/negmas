"""Comprehensive tests for outcome spaces including SingletonOutcomeSpace and set operations."""

from __future__ import annotations

import pytest

from negmas.outcomes import (
    DiscreteCartesianOutcomeSpace,
    EnumeratingOutcomeSpace,
    SingletonIssue,
    SingletonOutcomeSpace,
    make_issue,
    make_os,
    os_difference,
    os_intersection,
    os_union,
)


class TestSingletonIssue:
    """Tests for SingletonIssue class."""

    def test_singleton_issue_creation(self):
        """Test basic SingletonIssue creation."""
        issue = SingletonIssue(5, name="test")
        assert issue.name == "test"
        assert issue.value == 5
        assert issue.cardinality == 1

    def test_singleton_issue_with_string(self):
        """Test SingletonIssue with string value."""
        issue = SingletonIssue("hello", name="greeting")
        assert issue.value == "hello"
        assert issue.cardinality == 1

    def test_singleton_issue_all_generator(self):
        """Test that all generator yields the single value."""
        issue = SingletonIssue(42, name="answer")
        values = list(issue.all)
        assert values == [42]

    def test_singleton_issue_rand(self):
        """Test that rand always returns the single value."""
        issue = SingletonIssue("only", name="single")
        for _ in range(10):
            assert issue.rand() == "only"

    def test_singleton_issue_is_valid(self):
        """Test is_valid for SingletonIssue."""
        issue = SingletonIssue(10, name="num")
        assert issue.is_valid(10)
        assert not issue.is_valid(11)
        assert not issue.is_valid("10")

    def test_singleton_issue_contains(self):
        """Test contains for SingletonIssue."""
        issue1 = SingletonIssue(5, name="a")
        issue2 = SingletonIssue(5, name="b")
        issue3 = SingletonIssue(6, name="c")

        assert issue1.contains(issue2)
        assert not issue1.contains(issue3)

    def test_singleton_issue_is_discrete(self):
        """Test that SingletonIssue is discrete."""
        issue = SingletonIssue(1, name="x")
        assert issue.is_discrete()
        assert not issue.is_continuous()

    def test_singleton_issue_numeric(self):
        """Test numeric properties of SingletonIssue."""
        int_issue = SingletonIssue(5, name="int")
        assert int_issue.is_numeric()
        assert int_issue.is_integer()
        assert int_issue.min_value == 5
        assert int_issue.max_value == 5

        float_issue = SingletonIssue(3.14, name="float")
        assert float_issue.is_numeric()
        assert float_issue.is_float()

        str_issue = SingletonIssue("text", name="str")
        assert not str_issue.is_numeric()

    def test_singleton_issue_rand_invalid(self):
        """Test rand_invalid returns a non-valid value."""
        int_issue = SingletonIssue(5, name="int")
        invalid = int_issue.rand_invalid()
        assert not int_issue.is_valid(invalid)

        str_issue = SingletonIssue("hello", name="str")
        invalid = str_issue.rand_invalid()
        assert not str_issue.is_valid(invalid)


class TestSingletonOutcomeSpace:
    """Tests for SingletonOutcomeSpace class."""

    def test_singleton_os_from_outcome(self):
        """Test creating SingletonOutcomeSpace from an outcome."""
        outcome = (1, "a", 3.5)
        os = SingletonOutcomeSpace(outcome)

        assert os.outcome == outcome
        assert os.cardinality == 1
        assert len(os.issues) == 3

    def test_singleton_os_with_issue_names(self):
        """Test creating SingletonOutcomeSpace with custom issue names."""
        outcome = (1, 2, 3)
        os = SingletonOutcomeSpace(outcome, issue_names=["x", "y", "z"], name="test_os")

        assert os.outcome == outcome
        assert list(os.issue_names) == ["x", "y", "z"]
        assert os.name == "test_os"

    def test_singleton_os_auto_issue_names(self):
        """Test auto-generated issue names."""
        outcome = ("a", "b")
        os = SingletonOutcomeSpace(outcome)

        assert [i.name for i in os.issues] == ["issue00", "issue01"]

    def test_singleton_os_enumerate(self):
        """Test enumerate returns single outcome."""
        outcome = (1, 2)
        os = SingletonOutcomeSpace(outcome)

        outcomes = list(os.enumerate())
        assert outcomes == [outcome]

    def test_singleton_os_sample(self):
        """Test sample returns the outcome."""
        outcome = (5, 10)
        os = SingletonOutcomeSpace(outcome)

        samples = list(os.sample(3, with_replacement=True))
        assert len(samples) == 3
        assert all(s == outcome for s in samples)

    def test_singleton_os_sample_without_replacement(self):
        """Test sample without replacement raises for n > 1."""
        outcome = (1, 2)
        os = SingletonOutcomeSpace(outcome)

        with pytest.raises(ValueError):
            list(os.sample(2, with_replacement=False, fail_if_not_enough=True))

    def test_singleton_os_is_valid(self):
        """Test is_valid for SingletonOutcomeSpace."""
        outcome = (1, 2)
        os = SingletonOutcomeSpace(outcome)

        assert os.is_valid(outcome)
        assert not os.is_valid((1, 3))
        assert not os.is_valid((2, 2))

    def test_singleton_os_is_discrete(self):
        """Test that SingletonOutcomeSpace is discrete."""
        os = SingletonOutcomeSpace((1, 2))
        assert os.is_discrete()
        assert os.is_finite()

    def test_singleton_os_mismatched_names_raises(self):
        """Test that mismatched issue names raises ValueError."""
        with pytest.raises(ValueError):
            SingletonOutcomeSpace((1, 2, 3), issue_names=["a", "b"])


class TestContainsOs:
    """Tests for contains_os method across all outcome space types."""

    def test_cartesian_contains_smaller_cartesian(self):
        """Test CartesianOutcomeSpace contains smaller CartesianOutcomeSpace."""
        large_os = make_os(issues=[make_issue(10, "a"), make_issue(10, "b")])
        small_os = make_os(issues=[make_issue((2, 5), "a"), make_issue((3, 7), "b")])

        assert large_os.contains_os(small_os)

    def test_cartesian_does_not_contain_larger(self):
        """Test CartesianOutcomeSpace does not contain larger space."""
        small_os = make_os(issues=[make_issue((0, 5), "a")])
        large_os = make_os(issues=[make_issue((0, 10), "a")])

        assert not small_os.contains_os(large_os)

    def test_cartesian_contains_singleton(self):
        """Test CartesianOutcomeSpace contains SingletonOutcomeSpace."""
        os = make_os(issues=[make_issue(10, "a"), make_issue(10, "b")])
        singleton = SingletonOutcomeSpace((3, 5), issue_names=["a", "b"])

        assert os.contains_os(singleton)

    def test_cartesian_does_not_contain_invalid_singleton(self):
        """Test CartesianOutcomeSpace does not contain invalid singleton."""
        os = make_os(issues=[make_issue(5, "a"), make_issue(5, "b")])
        singleton = SingletonOutcomeSpace((10, 10), issue_names=["a", "b"])

        assert not os.contains_os(singleton)

    def test_singleton_contains_singleton(self):
        """Test SingletonOutcomeSpace contains same singleton."""
        s1 = SingletonOutcomeSpace((1, 2))
        s2 = SingletonOutcomeSpace((1, 2))

        assert s1.contains_os(s2)

    def test_singleton_does_not_contain_different_singleton(self):
        """Test SingletonOutcomeSpace does not contain different singleton."""
        s1 = SingletonOutcomeSpace((1, 2))
        s2 = SingletonOutcomeSpace((1, 3))

        assert not s1.contains_os(s2)

    def test_enumerating_contains_subset(self):
        """Test EnumeratingOutcomeSpace contains subset."""
        large = EnumeratingOutcomeSpace(
            baseset={(1, 2), (3, 4), (5, 6), (7, 8)}, name="large"
        )
        small = EnumeratingOutcomeSpace(baseset={(1, 2), (3, 4)}, name="small")

        assert large.contains_os(small)
        assert not small.contains_os(large)

    def test_enumerating_contains_singleton(self):
        """Test EnumeratingOutcomeSpace contains SingletonOutcomeSpace."""
        eos = EnumeratingOutcomeSpace(baseset={(1, 2), (3, 4), (5, 6)}, name="enum")
        singleton = SingletonOutcomeSpace((3, 4))

        assert eos.contains_os(singleton)

    def test_cartesian_contains_enumerating(self):
        """Test CartesianOutcomeSpace contains EnumeratingOutcomeSpace."""
        cos = make_os(issues=[make_issue(10, "a"), make_issue(10, "b")])
        eos = EnumeratingOutcomeSpace(baseset={(1, 2), (3, 4), (5, 6)}, name="enum")

        assert cos.contains_os(eos)


class TestSetOperationsUnion:
    """Tests for union operation (| operator)."""

    def test_cartesian_union_cartesian(self):
        """Test union of two CartesianOutcomeSpaces."""
        os1 = make_os(issues=[make_issue(3, "a")])
        os2 = make_os(issues=[make_issue((2, 5), "a")])

        result = os1 | os2
        outcomes = set(result.enumerate())

        # os1 has (0,1,2), os2 has (2,3,4,5)
        expected = {(0,), (1,), (2,), (3,), (4,), (5,)}
        assert outcomes == expected

    def test_singleton_union_singleton(self):
        """Test union of two SingletonOutcomeSpaces."""
        s1 = SingletonOutcomeSpace((1,))
        s2 = SingletonOutcomeSpace((2,))

        result = s1 | s2
        outcomes = set(result.enumerate())

        assert outcomes == {(1,), (2,)}

    def test_singleton_union_same(self):
        """Test union of same singleton."""
        s1 = SingletonOutcomeSpace((1,))
        s2 = SingletonOutcomeSpace((1,))

        result = s1 | s2
        outcomes = set(result.enumerate())

        assert outcomes == {(1,)}

    def test_cartesian_union_singleton(self):
        """Test union of CartesianOutcomeSpace and SingletonOutcomeSpace."""
        os = make_os(issues=[make_issue(3, "a")])
        singleton = SingletonOutcomeSpace((5,), issue_names=["a"])

        result = os | singleton
        outcomes = set(result.enumerate())

        assert outcomes == {(0,), (1,), (2,), (5,)}

    def test_enumerating_union_enumerating(self):
        """Test union of EnumeratingOutcomeSpaces."""
        e1 = EnumeratingOutcomeSpace(baseset={(1,), (2,)})
        e2 = EnumeratingOutcomeSpace(baseset={(2,), (3,)})

        result = e1 | e2
        outcomes = set(result.enumerate())

        assert outcomes == {(1,), (2,), (3,)}

    def test_os_union_function(self):
        """Test os_union function directly."""
        os1 = make_os(issues=[make_issue(2, "a")])
        os2 = make_os(issues=[make_issue((1, 3), "a")])

        result = os_union(os1, os2, name="union_result")
        assert result.name == "union_result"
        outcomes = set(result.enumerate())
        assert outcomes == {(0,), (1,), (2,), (3,)}


class TestSetOperationsIntersection:
    """Tests for intersection operation (& operator)."""

    def test_cartesian_intersection_cartesian(self):
        """Test intersection of two CartesianOutcomeSpaces."""
        os1 = make_os(issues=[make_issue(5, "a")])  # 0-4
        os2 = make_os(issues=[make_issue((2, 6), "a")])  # 2-6

        result = os1 & os2
        outcomes = set(result.enumerate())

        assert outcomes == {(2,), (3,), (4,)}

    def test_singleton_intersection_containing_space(self):
        """Test intersection of singleton with containing space."""
        os = make_os(issues=[make_issue(10, "a")])
        singleton = SingletonOutcomeSpace((5,), issue_names=["a"])

        result = os & singleton
        outcomes = set(result.enumerate())

        assert outcomes == {(5,)}

    def test_singleton_intersection_non_containing_space(self):
        """Test intersection of singleton with non-containing space."""
        os = make_os(issues=[make_issue(5, "a")])  # 0-4
        singleton = SingletonOutcomeSpace((10,), issue_names=["a"])

        result = os & singleton
        outcomes = set(result.enumerate())

        assert outcomes == set()  # Empty intersection

    def test_disjoint_cartesian_intersection(self):
        """Test intersection of disjoint CartesianOutcomeSpaces."""
        os1 = make_os(issues=[make_issue((0, 5), "a")])  # 0-5
        os2 = make_os(issues=[make_issue((10, 15), "a")])  # 10-15

        result = os1 & os2
        outcomes = set(result.enumerate())

        assert outcomes == set()

    def test_enumerating_intersection_enumerating(self):
        """Test intersection of EnumeratingOutcomeSpaces."""
        e1 = EnumeratingOutcomeSpace(baseset={(1,), (2,), (3,)})
        e2 = EnumeratingOutcomeSpace(baseset={(2,), (3,), (4,)})

        result = e1 & e2
        outcomes = set(result.enumerate())

        assert outcomes == {(2,), (3,)}

    def test_os_intersection_function(self):
        """Test os_intersection function directly."""
        os1 = make_os(issues=[make_issue(5, "a")])
        os2 = make_os(issues=[make_issue((2, 7), "a")])

        result = os_intersection(os1, os2, name="intersection_result")
        assert result.name == "intersection_result"


class TestSetOperationsDifference:
    """Tests for difference operation (- operator)."""

    def test_cartesian_difference_cartesian(self):
        """Test difference of CartesianOutcomeSpaces."""
        os1 = make_os(issues=[make_issue(5, "a")])  # 0-4
        os2 = make_os(issues=[make_issue((2, 6), "a")])  # 2-6

        result = os1 - os2
        outcomes = set(result.enumerate())

        assert outcomes == {(0,), (1,)}

    def test_singleton_difference_containing_space(self):
        """Test difference of singleton from containing space."""
        os = make_os(issues=[make_issue(5, "a")])  # 0-4
        singleton = SingletonOutcomeSpace((2,), issue_names=["a"])

        result = os - singleton
        outcomes = set(result.enumerate())

        assert outcomes == {(0,), (1,), (3,), (4,)}

    def test_singleton_difference_from_itself(self):
        """Test difference of singleton from itself."""
        s = SingletonOutcomeSpace((1,))

        result = s - s
        outcomes = set(result.enumerate())

        assert outcomes == set()

    def test_difference_from_disjoint_space(self):
        """Test difference from disjoint space."""
        os1 = make_os(issues=[make_issue(3, "a")])  # 0-2
        os2 = make_os(issues=[make_issue((5, 8), "a")])  # 5-8

        result = os1 - os2
        outcomes = set(result.enumerate())

        # Nothing removed since spaces are disjoint
        assert outcomes == {(0,), (1,), (2,)}

    def test_enumerating_difference_enumerating(self):
        """Test difference of EnumeratingOutcomeSpaces."""
        e1 = EnumeratingOutcomeSpace(baseset={(1,), (2,), (3,), (4,)})
        e2 = EnumeratingOutcomeSpace(baseset={(2,), (3,)})

        result = e1 - e2
        outcomes = set(result.enumerate())

        assert outcomes == {(1,), (4,)}

    def test_os_difference_function(self):
        """Test os_difference function directly."""
        os1 = make_os(issues=[make_issue(5, "a")])
        os2 = make_os(issues=[make_issue((3, 7), "a")])

        result = os_difference(os1, os2, name="difference_result")
        assert result.name == "difference_result"


class TestMixedOperations:
    """Tests for mixed set operations."""

    def test_chain_operations(self):
        """Test chaining multiple set operations."""
        os1 = make_os(issues=[make_issue(5, "a")])  # 0-4
        os2 = make_os(issues=[make_issue((3, 7), "a")])  # 3-7
        s = SingletonOutcomeSpace((2,), issue_names=["a"])

        # (os1 | os2) - s
        result = (os1 | os2) - s
        outcomes = set(result.enumerate())

        expected = {(0,), (1,), (3,), (4,), (5,), (6,), (7,)}
        assert outcomes == expected

    def test_intersection_then_union(self):
        """Test intersection followed by union."""
        os1 = make_os(issues=[make_issue(5, "a")])  # 0-4
        os2 = make_os(issues=[make_issue((2, 6), "a")])  # 2-6
        os3 = make_os(issues=[make_issue((8, 10), "a")])  # 8-10

        # (os1 & os2) | os3
        result = (os1 & os2) | os3
        outcomes = set(result.enumerate())

        expected = {(2,), (3,), (4,), (8,), (9,), (10,)}
        assert outcomes == expected

    def test_complex_expression(self):
        """Test complex set expression."""
        a = EnumeratingOutcomeSpace(baseset={(1,), (2,), (3,), (4,), (5,)})
        b = EnumeratingOutcomeSpace(baseset={(3,), (4,), (5,), (6,), (7,)})
        c = EnumeratingOutcomeSpace(baseset={(5,), (6,), (7,), (8,), (9,)})

        # (a | b) & c
        result = (a | b) & c
        outcomes = set(result.enumerate())

        expected = {(5,), (6,), (7,)}
        assert outcomes == expected


class TestMultiIssueSpaces:
    """Tests for multi-issue outcome spaces."""

    def test_two_issue_singleton(self):
        """Test two-issue SingletonOutcomeSpace."""
        outcome = (1, "a")
        os = SingletonOutcomeSpace(outcome, issue_names=["num", "char"])

        assert os.outcome == outcome
        assert os.cardinality == 1
        assert os.is_valid(outcome)
        assert not os.is_valid((1, "b"))

    def test_two_issue_union(self):
        """Test union with two-issue spaces."""
        os1 = make_os(issues=[make_issue(2, "a"), make_issue(["x", "y"], "b")])
        s = SingletonOutcomeSpace((5, "z"), issue_names=["a", "b"])

        result = os1 | s
        outcomes = set(result.enumerate())

        expected = {(0, "x"), (0, "y"), (1, "x"), (1, "y"), (5, "z")}
        assert outcomes == expected

    def test_two_issue_intersection(self):
        """Test intersection with two-issue spaces."""
        os1 = make_os(issues=[make_issue(3, "a"), make_issue(["x", "y", "z"], "b")])
        os2 = make_os(issues=[make_issue((1, 4), "a"), make_issue(["y", "z"], "b")])

        result = os1 & os2
        outcomes = set(result.enumerate())

        expected = {(1, "y"), (1, "z"), (2, "y"), (2, "z")}
        assert outcomes == expected

    def test_two_issue_difference(self):
        """Test difference with two-issue spaces."""
        os = make_os(issues=[make_issue(2, "a"), make_issue(2, "b")])
        s = SingletonOutcomeSpace((0, 0), issue_names=["a", "b"])

        result = os - s
        outcomes = set(result.enumerate())

        expected = {(0, 1), (1, 0), (1, 1)}
        assert outcomes == expected


class TestContainmentOperator:
    """Tests for the in operator (containment checks)."""

    def test_outcome_in_cartesian(self):
        """Test outcome in CartesianOutcomeSpace."""
        os = make_os(issues=[make_issue(5, "a")])
        assert (2,) in os
        assert (10,) not in os

    def test_outcome_in_singleton(self):
        """Test outcome in SingletonOutcomeSpace."""
        s = SingletonOutcomeSpace((1, 2))
        assert (1, 2) in s
        assert (1, 3) not in s

    def test_singleton_in_cartesian(self):
        """Test SingletonOutcomeSpace in CartesianOutcomeSpace."""
        os = make_os(issues=[make_issue(10, "a")])
        s = SingletonOutcomeSpace((5,), issue_names=["a"])

        assert s in os

    def test_cartesian_in_cartesian(self):
        """Test CartesianOutcomeSpace in CartesianOutcomeSpace."""
        large = make_os(issues=[make_issue(10, "a")])
        small = make_os(issues=[make_issue((2, 5), "a")])

        assert small in large
        assert large not in small


class TestCartesianProduct:
    """Tests for cartesian product (__mul__ operator)."""

    def test_cartesian_product_basic(self):
        """Test basic cartesian product."""
        os1 = make_os(issues=[make_issue(2, "a")])
        os2 = make_os(issues=[make_issue(2, "b")])

        result = os1 * os2

        assert len(result.issues) == 2
        assert result.cardinality == 4

    def test_cartesian_product_via_method(self):
        """Test cartesian product via method."""
        os1 = make_os(issues=[make_issue(2, "a")])
        os2 = make_os(issues=[make_issue(3, "b")])

        result = os1.cartesian_product(os2)

        assert len(result.issues) == 2
        assert result.cardinality == 6

    def test_cartesian_product_name(self):
        """Test cartesian product naming."""
        os1 = DiscreteCartesianOutcomeSpace(issues=(make_issue(2, "a"),), name="first")
        os2 = DiscreteCartesianOutcomeSpace(issues=(make_issue(2, "b"),), name="second")

        result = os1 * os2

        assert result.name == "first*second"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_enumerating_space(self):
        """Test operations with empty EnumeratingOutcomeSpace."""
        empty = EnumeratingOutcomeSpace(baseset=set())
        os = make_os(issues=[make_issue(3, "a")])

        union = os | empty
        assert set(union.enumerate()) == set(os.enumerate())

        intersection = os & empty
        assert set(intersection.enumerate()) == set()

        diff = os - empty
        assert set(diff.enumerate()) == set(os.enumerate())

    def test_singleton_with_tuple_value(self):
        """Test SingletonIssue with tuple value."""
        issue = SingletonIssue((1, 2, 3), name="tuple_issue")
        assert issue.value == (1, 2, 3)
        assert issue.is_valid((1, 2, 3))
        assert not issue.is_valid((1, 2, 4))

    def test_singleton_os_repr_and_str(self):
        """Test string representations of SingletonOutcomeSpace."""
        os = SingletonOutcomeSpace((1, 2), name="test")
        repr_str = repr(os)
        str_str = str(os)

        assert "1" in repr_str and "2" in repr_str
        assert "1" in str_str and "2" in str_str
