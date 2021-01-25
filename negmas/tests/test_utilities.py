import random
import pkg_resources
import pytest
from pytest import mark

from negmas.outcomes import Issue, outcome_as_tuple
from negmas.utilities import (
    HyperRectangleUtilityFunction,
    LinearUtilityAggregationFunction,
    LinearUtilityFunction,
    MappingUtilityFunction,
    pareto_frontier,
    UtilityFunction,
    utility_range,
)


@mark.parametrize(["n_issues"], [(2,), (3,)])
def test_ufun_range_linear(n_issues):
    issues = [Issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues)]
    rs = [(i + 1.0) * random.random() for i in range(n_issues)]
    ufun = LinearUtilityFunction(weights=rs, reserved_value=0.0)
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = utility_range(ufun, issues=issues)
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


@mark.parametrize(["n_issues"], [(2,), (3,)])
def test_ufun_range_general(n_issues):
    issues = [Issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues)]
    rs = [(i + 1.0) * random.random() for i in range(n_issues)]
    ufun = MappingUtilityFunction(
        mapping=lambda x: sum(r * v for r, v in zip(rs, outcome_as_tuple(x))),
        outcome_type=tuple,
    )
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = utility_range(ufun, issues=issues)
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


def test_pareto_frontier_does_not_depend_on_order():
    u1 = [
        0.5337723805661662,
        0.8532272031479199,
        0.4781281413197942,
        0.7242899747791032,
        0.3461879818432919,
        0.2608677043479706,
        0.9419131964655383,
        0.29368079952747694,
        0.6093201983562316,
        0.7066918086398718,
    ]
    u2 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    welfare = [_1 + _2 for _1, _2 in zip(u1, u2)]
    assert welfare.index(max(welfare)) == 3

    f1 = MappingUtilityFunction(lambda o: u1[o[0]])
    f2 = MappingUtilityFunction(lambda o: u2[o[0]])
    assert all(f1((i,)) == u1[i] for i in range(10))
    assert all(f2((i,)) == u2[i] for i in range(10))
    p1, l1 = pareto_frontier([f1, f2], outcomes=[(_,) for _ in range(10)])
    p2, l2 = pareto_frontier([f2, f1], outcomes=[(_,) for _ in range(10)])

    assert p1 == [(0.9419131964655383, 0.0), (0.7242899747791032, 1.0)]
    assert p2 == [(1.0, 0.7242899747791032), (0.0, 0.9419131964655383)]
    assert l1 == [6, 3]
    assert l2 == list(reversed(l1))
    assert len(p1) == len(p2)

    # reverse order of p2
    p2 = [(_[1], _[0]) for _ in p2]
    for a in p1:
        assert a in p2


def test_linear_utility():
    buyer_utility = LinearUtilityAggregationFunction(
        {
            "cost": lambda x: -x,
            "number of items": lambda x: 0.5 * x,
            "delivery": {"delivered": 10.0, "not delivered": -2.0},
        }
    )
    assert (
        buyer_utility({"cost": 1.0, "number of items": 3, "delivery": "not delivered"})
        == -1.0 + 1.5 - 2.0
    )


def test_hypervolume_utility():
    f = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
    )
    f_ignore_input = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_issues_not_in_input=True,
    )
    f_ignore_failing = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
    )
    f_ignore_both = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
        ignore_issues_not_in_input=True,
    )

    g = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
    )
    g_ignore_input = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_issues_not_in_input=True,
    )
    g_ignore_failing = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
    )
    g_ignore_both = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
        ignore_issues_not_in_input=True,
    )

    funs = [
        g,
        g_ignore_input,
        g_ignore_failing,
        g_ignore_both,
        f,
        f_ignore_input,
        f_ignore_failing,
        f_ignore_both,
    ]
    outcomes = [
        [1.5, 1.5, 2.5],  # belongs to all volumes
        [1.5, 1.5, 1.0],  # belongs to first
        {0: 1.5, 2: 2.5},
        {0: 11.5, 1: 11.5, 2: 12.5},
        [1.5],
        {2: 2.5},
    ]
    expected = [
        [8.5, 8.5, 8.5, 8.5, 13.5, 13.5, 13.5, 13.5],
        [2.0, 2.0, 2.0, 2.0, 7.0, 7.0, 7.0, 7.0],
        [None, 6.5, None, 6.5, None, 11.5, None, 11.5],
        [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
        [None, 0.0, None, 0.0, None, 5.0, None, 5.0],
        [None, 0.0, None, 0.0, None, 5.0, None, 5.0],
    ]

    for outcome, expectation in zip(outcomes, expected):
        utilities = [f(outcome) for f in funs]
        for i, (u, e) in enumerate(zip(utilities, expectation)):
            # print(i, utilities, outcome)
            assert u == e


def test_normalization():

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=True,
        normalize_utility=False,
    )
    assert abs(u(("Dell+60 Gb+19'' LCD",)) - 21.987727736172488) < 0.000001
    assert abs(u(("HP+80 Gb+20'' LCD",)) - 22.68559475583014) < 0.000001

    gt_max = {
        ("Dell", "60 Gb", "19'' LCD"): 0.7328862913051053,
        ("Dell", "60 Gb", "20'' LCD"): 0.6150545888614856,
        ("Dell", "60 Gb", "23'' LCD"): 0.6739704400832954,
        ("Dell", "80 Gb", "19'' LCD"): 0.6068653140240788,
        ("Dell", "80 Gb", "20'' LCD"): 0.48903361158045916,
        ("Dell", "80 Gb", "23'' LCD"): 0.547949462802269,
        ("Dell", "120 Gb", "19'' LCD"): 0.4682422390149499,
        ("Dell", "120 Gb", "20'' LCD"): 0.3504105365713301,
        ("Dell", "120 Gb", "23'' LCD"): 0.40932638779313996,
        ("Macintosh", "60 Gb", "19'' LCD"): 0.851603495169503,
        ("Macintosh", "60 Gb", "20'' LCD"): 0.7337717927258832,
        ("Macintosh", "60 Gb", "23'' LCD"): 0.7926876439476931,
        ("Macintosh", "80 Gb", "19'' LCD"): 0.7255825178884765,
        ("Macintosh", "80 Gb", "20'' LCD"): 0.6077508154448569,
        ("Macintosh", "80 Gb", "23'' LCD"): 0.6666666666666666,
        ("Macintosh", "120 Gb", "19'' LCD"): 0.5869594428793475,
        ("Macintosh", "120 Gb", "20'' LCD"): 0.4691277404357278,
        ("Macintosh", "120 Gb", "23'' LCD"): 0.5280435916575378,
        ("HP", "60 Gb", "19'' LCD"): 1.0,
        ("HP", "60 Gb", "20'' LCD"): 0.8821682975563803,
        ("HP", "60 Gb", "23'' LCD"): 0.9410841487781901,
        ("HP", "80 Gb", "19'' LCD"): 0.8739790227189735,
        ("HP", "80 Gb", "20'' LCD"): 0.7561473202753539,
        ("HP", "80 Gb", "23'' LCD"): 0.8150631714971638,
        ("HP", "120 Gb", "19'' LCD"): 0.7353559477098447,
        ("HP", "120 Gb", "20'' LCD"): 0.6175242452662248,
        ("HP", "120 Gb", "23'' LCD"): 0.6764400964880347,
    }
    gt_range = {
        ("Dell", "60 Gb", "19'' LCD"): 0.5887961185746265,
        ("Dell", "60 Gb", "20'' LCD"): 0.40740200879076527,
        ("Dell", "60 Gb", "23'' LCD"): 0.4980990636826959,
        ("Dell", "80 Gb", "19'' LCD"): 0.39479516200759546,
        ("Dell", "80 Gb", "20'' LCD"): 0.21340105222373418,
        ("Dell", "80 Gb", "23'' LCD"): 0.3040981071156648,
        ("Dell", "120 Gb", "19'' LCD"): 0.1813941097838614,
        ("Dell", "120 Gb", "20'' LCD"): 0.0,
        ("Dell", "120 Gb", "23'' LCD"): 0.09069705489193064,
        ("Macintosh", "60 Gb", "19'' LCD"): 0.7715533992081258,
        ("Macintosh", "60 Gb", "20'' LCD"): 0.5901592894242645,
        ("Macintosh", "60 Gb", "23'' LCD"): 0.6808563443161952,
        ("Macintosh", "80 Gb", "19'' LCD"): 0.5775524426410947,
        ("Macintosh", "80 Gb", "20'' LCD"): 0.3961583328572335,
        ("Macintosh", "80 Gb", "23'' LCD"): 0.48685538774916415,
        ("Macintosh", "120 Gb", "19'' LCD"): 0.3641513904173608,
        ("Macintosh", "120 Gb", "20'' LCD"): 0.18275728063349939,
        ("Macintosh", "120 Gb", "23'' LCD"): 0.27345433552543014,
        ("HP", "60 Gb", "19'' LCD"): 1.0,
        ("HP", "60 Gb", "20'' LCD"): 0.8186058902161387,
        ("HP", "60 Gb", "23'' LCD"): 0.9093029451080693,
        ("HP", "80 Gb", "19'' LCD"): 0.8059990434329689,
        ("HP", "80 Gb", "20'' LCD"): 0.6246049336491076,
        ("HP", "80 Gb", "23'' LCD"): 0.7153019885410383,
        ("HP", "120 Gb", "19'' LCD"): 0.592597991209235,
        ("HP", "120 Gb", "20'' LCD"): 0.41120388142537345,
        ("HP", "120 Gb", "23'' LCD"): 0.501900936317304,
    }
    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=False,
        normalize_utility=True,
        normalize_max_only=False,
        keep_issue_names=True,
        keep_value_names=True,
    )

    for k, v in gt_range.items():
        assert abs(v - u(k)) < 1e-3, (k, v, u(k))

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=False,
        normalize_utility=True,
        normalize_max_only=True,
        keep_issue_names=True,
        keep_value_names=True,
    )

    for k, v in gt_max.items():
        assert abs(v - u(k)) < 1e-3, (k, v, u(k))

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=True,
        keep_issue_names=False,
        keep_value_names=False,
        normalize_utility=False,
    )
    assert abs(u((0,)) - 21.987727736172488) < 0.000001

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=False,
        normalize_utility=False,
    )
    assert (
        abs(
            u({"Laptop": "Dell", "Harddisk": "60 Gb", "External Monitor": "19'' LCD"})
            - 21.987727736172488
        )
        < 0.000001
    )
    assert (
        abs(
            u({"Laptop": "HP", "Harddisk": "80 Gb", "External Monitor": "20'' LCD"})
            - 22.68559475583014
        )
        < 0.000001
    )

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=True,
        normalize_utility=True,
    )
    assert abs(u(("Dell+60 Gb+19'' LCD",)) - 0.599329436957658) < 0.1
    assert abs(u(("HP+80 Gb+20'' LCD",)) - 0.6342209804130308) < 0.01

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=True,
        keep_issue_names=False,
        keep_value_names=False,
        normalize_utility=True,
    )
    assert abs(u((0,)) - 0.599329436957658) < 0.1

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=False,
        normalize_utility=True,
    )
    assert (
        abs(
            u({"Laptop": "Dell", "Harddisk": "60 Gb", "External Monitor": "19'' LCD"})
            - 0.599329436957658
        )
        < 0.1
    )
    assert (
        abs(
            u({"Laptop": "HP", "Harddisk": "80 Gb", "External Monitor": "20'' LCD"})
            - 0.6342209804130308
        )
        < 0.01
    )
    assert (
        abs(
            u({"Laptop": "HP", "Harddisk": "60 Gb", "External Monitor": "19'' LCD"})
            - 1.0
        )
        < 0.0001
    )


@mark.parametrize(["normalize"], [(True,), (False,)])
def test_inverse_genius_domain(normalize):
    issues, _ = Issue.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ),
            "r",
        ).read(),
    )
    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
            "r",
        ).read(),
        force_single_issue=False,
        normalize_utility=normalize,
    )
    u.init_inverse(issues=issues)
    for i in range(100):
        v = u(u.inverse(i / 100.0, eps=(0.001, 0.1), assume_normalized=normalize))
        assert v-1e-3 <= v <= v+0.1


if __name__ == "__main__":
    pytest.main(args=[__file__])
