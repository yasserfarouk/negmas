"""The Discrete*OutcomeSpace -> Enumerable*OutcomeSpace rename and its aliases."""

from __future__ import annotations

import importlib

import pytest

from negmas import make_issue, make_os
from negmas.outcomes import EnumerableCartesianOutcomeSpace, EnumerableOutcomeSpace
from negmas.serialization import deserialize, serialize
from negmas.warnings import NegmasDeprecationWarning


@pytest.mark.parametrize(
    "module, old, new",
    [
        ("negmas", "DiscreteCartesianOutcomeSpace", EnumerableCartesianOutcomeSpace),
        ("negmas", "DiscreteOutcomeSpace", EnumerableOutcomeSpace),
        (
            "negmas.outcomes",
            "DiscreteCartesianOutcomeSpace",
            EnumerableCartesianOutcomeSpace,
        ),
        ("negmas.outcomes", "DiscreteOutcomeSpace", EnumerableOutcomeSpace),
        (
            "negmas.outcomes.outcome_space",
            "DiscreteCartesianOutcomeSpace",
            EnumerableCartesianOutcomeSpace,
        ),
        ("negmas.outcomes.protocols", "DiscreteOutcomeSpace", EnumerableOutcomeSpace),
    ],
)
def test_old_names_are_deprecated_aliases(module, old, new):
    with pytest.warns(NegmasDeprecationWarning, match=old):
        cls = getattr(importlib.import_module(module), old)
    assert cls is new


def test_star_imports_do_not_warn(recwarn):
    exec("from negmas import *", {})
    exec("from negmas.outcomes import *", {})
    assert not [w for w in recwarn if issubclass(w.category, NegmasDeprecationWarning)]


def test_saving_writes_the_new_name_and_loading_the_old_name_warns():
    os_ = make_os([make_issue(3, "a"), make_issue(["x", "y"], "b")])
    assert type(os_) is EnumerableCartesianOutcomeSpace
    d = serialize(os_)
    assert "EnumerableCartesianOutcomeSpace" in str(d)
    assert "DiscreteCartesianOutcomeSpace" not in str(d)
    old = eval(
        str(d).replace(
            "EnumerableCartesianOutcomeSpace", "DiscreteCartesianOutcomeSpace"
        )
    )
    with pytest.warns(NegmasDeprecationWarning, match="DiscreteCartesianOutcomeSpace"):
        loaded = deserialize(old)
    assert type(loaded) is EnumerableCartesianOutcomeSpace and loaded == os_
