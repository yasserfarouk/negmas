"""Round-trips of list issues holding records of mixed issue kinds."""

from __future__ import annotations

import inspect
import json

import pytest

import negmas
import negmas.outcomes as O
import negmas.preferences as P
from negmas.inout import Scenario
from negmas.outcomes import (
    ContiguousIssue,
    IntIssue,
    ListIssue,
    RealIssue,
    RecordIssue,
    StrIssue,
    make_issue,
    make_os,
)
from negmas.preferences import (
    LinearAdditiveUtilityFunction,
    ListAggregationFun,
    RecordAdditiveFun,
)
from negmas.preferences.value_fun import AffineFun, LambdaFun, TableFun
from negmas.serialization import deserialize, serialize


def lots():
    lot = RecordIssue(
        [
            ContiguousIssue((1, 20), "price"),
            IntIssue("quantity", min=1, typical=(1, 9)),
            RealIssue("weight", min=0.0, typical=(0.0, 5.0)),
            StrIssue("note", max_length=4, alphabet="ab"),
            make_issue(["red", "blue"], "color"),
        ]
    )
    return ListIssue("lots", lot, min_count=1, max_count=3, unique_by=("price",))


def with_full_names(d):
    """Prefix bare type names with ``negmas.`` (as some loaders do)."""
    if isinstance(d, dict):
        return {
            k: f"negmas.{v}" if k == "type" and "." not in v else with_full_names(v)
            for k, v in d.items()
        }
    if isinstance(d, list):
        return [with_full_names(_) for _ in d]
    return d


def roundtrips(x):
    yield deserialize(serialize(x))
    yield deserialize(json.loads(json.dumps(serialize(x))))
    yield type(x).from_dict(x.to_dict())

    def short():  # deserialize consumes its input, so make a fresh one each time
        return serialize(x, shorten_type_field=True, python_class_identifier="type")

    yield deserialize(short(), python_class_identifier="type", base_module="negmas")
    yield deserialize(
        with_full_names(short()), python_class_identifier="type", base_module="negmas"
    )


@pytest.mark.parametrize("which", ["record", "list"])
def test_nested_record_roundtrips(which):
    issue = lots() if which == "list" else lots().element
    record = issue.element if which == "list" else issue
    for other in roundtrips(issue):
        assert type(other) is type(issue) and other == issue
        other_record = other.element if which == "list" else other
        assert [type(_) for _ in other_record.fields] == [
            type(_) for _ in record.fields
        ]


def test_scenario_with_list_of_records_roundtrips(tmp_path):
    os = make_os([make_issue(["fast", "slow"], "ship"), lots()], name="trade")
    lot_value = RecordAdditiveFun(
        (
            AffineFun(-1.0, bias=20.0),
            LambdaFun(lambda q: 1.0 / q, min_value=0.0, max_value=1.0),
            LambdaFun(lambda w: -w, min_value=-5.0, max_value=0.0),
            LambdaFun(lambda s: float("a" in s), min_value=0.0, max_value=1.0),
            TableFun({"red": 1.0, "blue": 0.0}),
        )
    )
    u = LinearAdditiveUtilityFunction(
        [
            TableFun({"fast": 1.0, "slow": 0.0}),
            ListAggregationFun(lot_value, aggregate="mean"),
        ],
        outcome_space=os,
        name="buyer",
    )
    Scenario(outcome_space=os, ufuns=(u,)).dumpas(
        tmp_path, "yml", save_stats=False, save_info=False
    )
    loaded = Scenario.load(tmp_path)
    assert loaded is not None
    assert loaded.outcome_space.issues == os.issues
    o = ("fast", ((5, 2, 1.0, "ab", "red"), (7, 1, 0.0, "b", "blue")))
    assert loaded.outcome_space.is_valid(o)
    assert loaded.ufuns[0](o) == pytest.approx(u(o))


@pytest.mark.parametrize("mod", [O, P], ids=lambda m: m.__name__)
def test_serializable_classes_resolve_from_the_top_level(mod):
    """Short type names in scenario files resolve as ``negmas.<Name>``."""
    missing = [
        n
        for n in mod.__all__
        if inspect.isclass(getattr(mod, n, None))
        and hasattr(getattr(mod, n), "from_dict")
        and not hasattr(negmas, n)
    ]
    assert not missing
