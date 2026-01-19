from __future__ import annotations
import os
from importlib.resources import files
from os import walk
from pathlib import Path

import pytest

from negmas import load_genius_domain_from_folder
from negmas.inout import Scenario, STATS_FILE_NAME
from negmas.outcomes import enumerate_issues
from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.ops import ScenarioStats
from negmas.sao import AspirationNegotiator

MAX_CARDINALITY = 10_000

try:
    # try to raise the maximum limit for open files
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (50000, -1))
except Exception:
    pass

GENIUSWEB_FOLDERS = Path(str(files("negmas").joinpath("tests/data/geniusweb")))


@pytest.fixture
def scenarios_folder():
    return str(files("negmas").joinpath("tests/data/scenarios"))


# todo: get these to work
SCENARIOS_TO_IGNORE = [
    "S-1NIKFRT-3",
    "IntegerDomain",
    "S-1NIKFRT-2",
    "S-1NAGUNL-128",
    "S-1NAGUNL-255",
    "50issueswithRV",
    "50issues",
    "30issuesDiscounted",
    "30issueswithRV",
    "30issues",
    "50issuesDiscounted",
    "30issuesDiscountedwithRV",
    "50issuesDiscountedwithRV",
    "AgentHp2",
    "web_service",
    "four_issues",
    "AMPOvsCity",
    "laptopdomain",
    "S-1NIKFRT-1",
    "S-1NAGUNL-114",
    "laptopdomainNoBayes",
    "inheritancedomain",
    "10issues",
    "10issuesDiscounted",
    "10issueswithRV",
    "10issuesDiscountedwithRV",
    "group9-vacation",
    "group9-killer_robot",
    "FitnessA",
    "FitnessB",
    "FitnessC",
    "Grandma",
    "YXAgent",
    # "kite",
    # "group5-car_domain",
    # "group2-new_sporthal",
    # "group12-symposium",
    # "group11-car_purchase",
    # "group9-vacation",
    # "group8-holiday",
]


def get_all_scenarios():
    base = Path(__file__).parent.parent / "data" / "scenarios"
    data = []
    for root, dirs, filenames in walk(base):
        if len(filenames) == 0 or len(dirs) != 0:
            continue
        if root.split("/")[-1] in SCENARIOS_TO_IGNORE:
            continue
        data.append(root)
    return data


def test_reading_writing_linear_preferences(tmp_path):
    from negmas.preferences import LinearAdditiveUtilityFunction, UtilityFunction

    base_folder = str(files("negmas").joinpath("tests/data/Laptop"))
    domain = load_genius_domain_from_folder(base_folder)
    ufuns, issues = domain.ufuns, domain.issues
    for ufun in ufuns:
        assert isinstance(ufun, LinearAdditiveUtilityFunction)
        dst = tmp_path / "tmp.xml"
        UtilityFunction.to_genius(ufun, issues=issues, file_name=dst)
        print(str(dst))
        ufun2, _ = UtilityFunction.from_genius(dst, issues=issues)
        try:
            os.unlink(dst)
        except Exception:
            pass
        assert isinstance(ufun2, LinearAdditiveUtilityFunction)
        for outcome in enumerate_issues(issues):
            assert abs(ufun2(outcome) - ufun(outcome)) < 1e-3


def test_importing_file_without_exceptions(scenarios_folder):
    folder_name = scenarios_folder + "/other/S-1NIKFRT-1"
    load_genius_domain_from_folder(folder_name)


def test_simple_run_with_aspiration_agents():
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    assert os.path.exists(file_name)
    domain = Scenario.from_genius_folder(Path(file_name))
    assert domain
    domain.to_single_issue()
    mechanism = domain.make_session(AspirationNegotiator, n_steps=100, time_limit=30)
    assert mechanism is not None
    mechanism.run()


@pytest.mark.parametrize(
    ("r0", "r1", "n_above"),
    (
        (0.0, 0.0, 11),
        (-0.1, -0.1, 11),
        (0.9, 0.9, 0),
        (0.0, 0.9, 2),
        (0.9, 0.0, 2),
        (0.0, 0.95, 1),
        (0.95, 0.0, 1),
    ),
)
def test_enumerate_discrete_rational(r0, r1, n_above):
    domain = Scenario.from_genius_folder(  # type: ignore
        Path(__file__).parent.parent
        / "data"
        / "scenarios"
        / "anac"
        / "y2013"
        / "Fifty2013",
        safe_parsing=False,
        ignore_discount=True,
    )
    assert domain
    domain.scale_max()
    domain: Scenario
    ufuns = domain.ufuns
    ufuns[0].reserved_value = r0
    ufuns[1].reserved_value = r1
    outcomes = list(domain.outcome_space.enumerate_or_sample())
    assert len(outcomes) == 11
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns,
                    aggregator=lambda x: True,  # type: ignore
                )
            )
        )
        == 0
    )
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns,
                    aggregator=lambda x: False,  # type: ignore
                )
            )
        )
        == 11
    )
    assert (
        len(
            list(
                domain.outcome_space.enumerate_or_sample_rational(
                    preferences=domain.ufuns, aggregator=any
                )
            )
        )
        == n_above
    )


def test_load_geniusweb_example_reserved_outcome():
    domain = GENIUSWEB_FOLDERS / "Fitness"
    scenario = Scenario.from_geniusweb_folder(domain, use_reserved_outcome=True)
    assert scenario is not None
    assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
    assert len(scenario.outcome_space.issues) == 5
    for i, (name, vals) in enumerate(
        (("type", 5), ("duration", 4), ("distance", 4), ("intensity", 4), ("price", 4))
    ):
        assert scenario.outcome_space.issues[i].name == name
        assert scenario.outcome_space.issues[i].cardinality == vals
    assert scenario.outcome_space.name == "fitness"
    assert len(scenario.ufuns) == 2
    assert all(isinstance(_, LinearAdditiveUtilityFunction) for _ in scenario.ufuns)
    assert all(
        (_.reserved_value is None or _.reserved_value == float("-inf"))
        and _.reserved_outcome is not None
        for _ in scenario.ufuns
    )
    assert scenario.ufuns[0].name == "fitness1"
    assert scenario.ufuns[1].name == "fitness2"


def test_load_geniusweb_example_reserved_value():
    domain = GENIUSWEB_FOLDERS / "Fitness"
    scenario = Scenario.from_geniusweb_folder(domain, use_reserved_outcome=False)
    assert scenario is not None
    assert isinstance(scenario.outcome_space, DiscreteCartesianOutcomeSpace)
    assert len(scenario.outcome_space.issues) == 5
    for i, (name, vals) in enumerate(
        (("type", 5), ("duration", 4), ("distance", 4), ("intensity", 4), ("price", 4))
    ):
        assert scenario.outcome_space.issues[i].name == name
        assert scenario.outcome_space.issues[i].cardinality == vals
    assert scenario.outcome_space.name == "fitness"
    assert len(scenario.ufuns) == 2
    assert all(isinstance(_, LinearAdditiveUtilityFunction) for _ in scenario.ufuns)
    assert all(
        _.reserved_value is not None and _.reserved_outcome is None
        for _ in scenario.ufuns
    )
    assert scenario.ufuns[0].name == "fitness1"
    assert scenario.ufuns[1].name == "fitness2"


# @pytest.mark.parametrize("folder_name", get_all_scenarios())
# def test_importing_all_single_issue_without_exceptions(folder_name):
#     # def test_encoding_decoding_all_without_discounting():
#     # folder_name = "/Users/yasser/code/projects/negmas/tests/data/scenarios/anac/y2012/FitnessC"
#     d2 = Domain.from_genius_folder(folder_name, safe_parsing=False)
#     d2.discretize()
#     assert isinstance(d2.outcome_space, DiscreteCartesianOutcomeSpace)
#     n = d2.outcome_space.cardinality
#     if n < 10_000:
#         d2.to_single_issue()
#         assert d2.outcome_space.cardinality == n or d2.outcome_space.cardinality == float("inf")


def test_scenario_calc_stats():
    """Test that calc_stats computes and stores stats correctly."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    assert scenario.stats is None

    stats = scenario.calc_stats()

    assert stats is not None
    assert isinstance(stats, ScenarioStats)
    assert scenario.stats is stats
    assert stats.opposition is not None
    assert len(stats.utility_ranges) == len(scenario.ufuns)
    assert len(stats.pareto_utils) > 0
    assert len(stats.pareto_outcomes) > 0


def test_scenario_save_and_load_stats(tmp_path):
    """Test that stats can be saved via dumpas and loaded back."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats
    stats = scenario.calc_stats()
    assert stats is not None

    # Save the scenario with stats
    scenario.dumpas(tmp_path, type="yml")

    # Verify stats file was created with correct name
    stats_file = tmp_path / STATS_FILE_NAME
    assert stats_file.exists(), f"Stats file {stats_file} should exist"

    # Load stats back
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    assert scenario2.stats is None

    scenario2.load_stats(tmp_path)
    assert scenario2.stats is not None

    # Verify loaded stats match original
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)
    assert len(scenario2.stats.pareto_utils) == len(stats.pareto_utils)


def test_scenario_load_stats_file(tmp_path):
    """Test load_stats_file method."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate and save stats
    stats = scenario.calc_stats()
    scenario.dumpas(tmp_path, type="yml")

    stats_file = tmp_path / STATS_FILE_NAME
    assert stats_file.exists()

    # Load stats from specific file
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2.stats is None

    scenario2.load_stats_file(stats_file)
    assert scenario2.stats is not None
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)


def test_scenario_load_stats_nonexistent(tmp_path):
    """Test that load_stats handles missing files gracefully."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    assert scenario.stats is None

    # Load from folder without stats file
    scenario.load_stats(tmp_path)
    assert scenario.stats is None

    # Load from nonexistent file
    scenario.load_stats_file(tmp_path / "nonexistent.yaml")
    assert scenario.stats is None


def test_scenario_load_with_stats(tmp_path):
    """Test that Scenario.load loads stats when available."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))

    # First create a scenario with stats and save it
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    scenario.calc_stats()
    scenario.dumpas(tmp_path, type="yml")

    # Now load from the saved location (which has stats)
    scenario_with_stats = Scenario.from_genius_folder(Path(file_name))
    assert scenario_with_stats is not None
    scenario_with_stats.load_stats(tmp_path)
    assert scenario_with_stats.stats is not None

    # Load without loading stats
    scenario_no_stats = Scenario.from_genius_folder(Path(file_name))
    assert scenario_no_stats is not None
    assert scenario_no_stats.stats is None


def test_scenario_calc_stats_attributes():
    """Test that calc_stats computes all expected attributes."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()

    # Check all ScenarioStats attributes are populated
    assert isinstance(stats.opposition, float)
    assert 0.0 <= stats.opposition <= 1.0

    assert isinstance(stats.utility_ranges, list)
    for r in stats.utility_ranges:
        assert len(r) == 2
        assert r[0] <= r[1]

    assert isinstance(stats.pareto_utils, tuple)
    assert isinstance(stats.pareto_outcomes, list)
    assert len(stats.pareto_utils) == len(stats.pareto_outcomes)

    assert isinstance(stats.nash_utils, list)
    assert isinstance(stats.nash_outcomes, list)

    assert isinstance(stats.kalai_utils, list)
    assert isinstance(stats.kalai_outcomes, list)

    assert isinstance(stats.max_welfare_utils, list)
    assert isinstance(stats.max_welfare_outcomes, list)


def test_scenario_stats_to_dict_exclude_pareto():
    """Test that to_dict can exclude pareto frontier data."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()
    assert stats is not None
    assert len(stats.pareto_utils) > 0
    assert len(stats.pareto_outcomes) > 0

    # Export with pareto
    d_with_pareto = stats.to_dict(include_pareto_frontier=True)
    assert len(d_with_pareto["pareto_utils"]) > 0
    assert len(d_with_pareto["pareto_outcomes"]) > 0

    # Export without pareto
    d_without_pareto = stats.to_dict(include_pareto_frontier=False)
    assert len(d_without_pareto["pareto_utils"]) == 0
    assert len(d_without_pareto["pareto_outcomes"]) == 0


def test_scenario_stats_from_dict_missing_pareto():
    """Test that from_dict handles missing pareto fields gracefully."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()
    assert stats is not None

    # Create dict without pareto fields
    d = stats.to_dict(include_pareto_frontier=False)
    assert len(d["pareto_utils"]) == 0
    assert len(d["pareto_outcomes"]) == 0

    # Reconstruct from dict without calculating pareto
    stats2 = ScenarioStats.from_dict(d)
    assert stats2.opposition == pytest.approx(stats.opposition, rel=1e-6)
    assert not stats2.has_pareto_frontier
    assert len(stats2.pareto_utils) == 0
    assert len(stats2.pareto_outcomes) == 0


def test_scenario_stats_from_dict_calc_pareto_if_missing():
    """Test that from_dict can calculate pareto frontier if missing."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()
    assert stats is not None
    original_pareto_count = len(stats.pareto_utils)
    assert original_pareto_count > 0

    # Create dict without pareto fields
    d = stats.to_dict(include_pareto_frontier=False)

    # Reconstruct from dict and calculate pareto
    stats2 = ScenarioStats.from_dict(
        d, ufuns=scenario.ufuns, calc_pareto_if_missing=True
    )
    assert stats2.has_pareto_frontier
    assert len(stats2.pareto_utils) == original_pareto_count
    assert len(stats2.pareto_outcomes) == original_pareto_count


def test_scenario_stats_from_dict_calc_pareto_requires_ufuns():
    """Test that from_dict raises error if calc_pareto_if_missing but no ufuns."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    stats = scenario.calc_stats()
    d = stats.to_dict(include_pareto_frontier=False)

    with pytest.raises(ValueError, match="ufuns must be provided"):
        ScenarioStats.from_dict(d, calc_pareto_if_missing=True)


def test_scenario_load_stats_calc_pareto_if_missing(tmp_path):
    """Test that load_stats can calculate pareto if missing."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats and save without pareto
    stats = scenario.calc_stats()
    original_pareto_count = len(stats.pareto_utils)
    assert original_pareto_count > 0

    scenario.dumpas(tmp_path, type="yml", include_pareto_frontier=False)

    # Verify stats file exists but has no pareto
    stats_file = tmp_path / STATS_FILE_NAME
    assert stats_file.exists()

    # Load without calculating pareto
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    scenario2.load_stats(tmp_path, calc_pareto_if_missing=False)
    assert scenario2.stats is not None
    assert not scenario2.stats.has_pareto_frontier

    # Load with calculating pareto
    scenario3 = Scenario.from_genius_folder(Path(file_name))
    assert scenario3 is not None
    scenario3.load_stats(tmp_path, calc_pareto_if_missing=True)
    assert scenario3.stats is not None
    assert scenario3.stats.has_pareto_frontier
    assert len(scenario3.stats.pareto_utils) == original_pareto_count


def test_scenario_load_stats_file_calc_pareto_if_missing(tmp_path):
    """Test that load_stats_file can calculate pareto if missing."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats and save without pareto
    stats = scenario.calc_stats()
    original_pareto_count = len(stats.pareto_utils)

    scenario.dumpas(tmp_path, type="yml", include_pareto_frontier=False)
    stats_file = tmp_path / STATS_FILE_NAME

    # Load with calculating pareto
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    scenario2.load_stats_file(stats_file, calc_pareto_if_missing=True)
    assert scenario2.stats is not None
    assert scenario2.stats.has_pareto_frontier
    assert len(scenario2.stats.pareto_utils) == original_pareto_count


def test_scenario_load_stats_legacy_json(tmp_path):
    """Test backward compatibility: load_stats can load legacy stats.json files."""
    from negmas.helpers.inout import dump

    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats
    stats = scenario.calc_stats()
    assert stats is not None

    # Save stats to legacy stats.json location (simulating old cartesian_tournament behavior)
    legacy_stats_file = tmp_path / "stats.json"
    dump(stats.to_dict(), legacy_stats_file)

    # Verify _stats.yaml does NOT exist (to ensure we're testing the fallback)
    new_stats_file = tmp_path / STATS_FILE_NAME
    assert not new_stats_file.exists()
    assert legacy_stats_file.exists()

    # Load stats - should fall back to stats.json
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    assert scenario2.stats is None

    scenario2.load_stats(tmp_path)
    assert scenario2.stats is not None
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)
    assert len(scenario2.stats.pareto_utils) == len(stats.pareto_utils)


def test_scenario_load_stats_prefers_new_format(tmp_path):
    """Test that load_stats prefers _stats.yaml over legacy stats.json."""
    from negmas.helpers.inout import dump

    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate stats
    stats = scenario.calc_stats()
    assert stats is not None

    # Save stats to new _stats.yaml with a distinctive opposition value
    scenario.dumpas(tmp_path, type="yml")

    # Also create a legacy stats.json with a DIFFERENT opposition value
    legacy_stats_file = tmp_path / "stats.json"
    modified_stats = stats.to_dict()
    modified_stats["opposition"] = 0.12345  # Distinctive value
    dump(modified_stats, legacy_stats_file)

    # Both files should exist
    new_stats_file = tmp_path / STATS_FILE_NAME
    assert new_stats_file.exists()
    assert legacy_stats_file.exists()

    # Load stats - should prefer _stats.yaml (original opposition value)
    scenario2 = Scenario.from_genius_folder(Path(file_name))
    assert scenario2 is not None
    scenario2.load_stats(tmp_path)
    assert scenario2.stats is not None

    # Should have loaded from _stats.yaml (original value), not stats.json (0.12345)
    assert scenario2.stats.opposition == pytest.approx(stats.opposition, rel=1e-6)
    assert scenario2.stats.opposition != pytest.approx(0.12345, rel=1e-6)


# ===== save_table Tests =====


def test_save_table_list_of_dicts_csv(tmp_path):
    """Test saving a list of dicts as CSV."""
    from negmas.helpers.inout import save_table

    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = save_table(data, tmp_path / "test.csv")

    assert path.exists()
    assert path.suffix == ".csv"
    with open(path) as f:
        content = f.read()
    assert "a,b" in content
    assert "1,2" in content
    assert "3,4" in content


def test_save_table_dataframe_csv(tmp_path):
    """Test saving a DataFrame as CSV."""
    import pandas as pd

    from negmas.helpers.inout import save_table

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = save_table(df, tmp_path / "test.csv")

    assert path.exists()
    with open(path) as f:
        content = f.read()
    assert "x,y" in content


def test_save_table_with_index(tmp_path):
    """Test saving with index enabled."""
    import pandas as pd

    from negmas.helpers.inout import save_table

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    path = save_table(df, tmp_path / "test.csv", index=True, index_label="idx")

    with open(path) as f:
        content = f.read()
    assert "idx" in content


def test_save_table_list_of_tuples(tmp_path):
    """Test saving a list of tuples with columns."""
    from negmas.helpers.inout import save_table

    data = [(1, 2), (3, 4)]
    path = save_table(data, tmp_path / "test.csv", columns=["col1", "col2"])

    with open(path) as f:
        content = f.read()
    assert "col1,col2" in content


def test_save_table_list_of_tuples_requires_columns(tmp_path):
    """Test that list of tuples requires columns parameter."""
    from negmas.helpers.inout import save_table

    data = [(1, 2), (3, 4)]
    with pytest.raises(ValueError, match="columns parameter is required"):
        save_table(data, tmp_path / "test.csv")


def test_save_table_gzip_format(tmp_path):
    """Test saving as gzip-compressed CSV."""
    import gzip

    from negmas.helpers.inout import save_table

    data = [{"a": 1, "b": 2}]
    path = save_table(data, tmp_path / "test.csv", storage_format="gzip")

    assert str(path).endswith(".gz")
    assert path.exists()

    # Verify it's valid gzip
    with gzip.open(path, "rt") as f:
        content = f.read()
    assert "a,b" in content


def test_save_table_parquet_format(tmp_path):
    """Test saving as Parquet."""
    import pandas as pd

    from negmas.helpers.inout import save_table

    data = [{"a": 1, "b": 2}]
    path = save_table(data, tmp_path / "test.csv", storage_format="parquet")

    assert path.suffix == ".parquet"
    assert path.exists()

    # Verify it's valid parquet
    df = pd.read_parquet(path)
    assert list(df.columns) == ["a", "b"]
    assert df["a"].tolist() == [1]


def test_save_table_empty_list(tmp_path):
    """Test saving an empty list."""
    from negmas.helpers.inout import save_table

    data = []
    path = save_table(data, tmp_path / "test.csv")

    assert path.exists()


def test_save_table_empty_list_with_columns(tmp_path):
    """Test saving an empty list with columns specified."""
    import pandas as pd

    from negmas.helpers.inout import save_table

    data = []
    path = save_table(data, tmp_path / "test.csv", columns=["a", "b"])

    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 0


def test_save_table_creates_parent_dirs(tmp_path):
    """Test that save_table creates parent directories."""
    from negmas.helpers.inout import save_table

    data = [{"a": 1}]
    path = save_table(data, tmp_path / "nested" / "dir" / "test.csv")

    assert path.exists()
    assert path.parent.exists()


def test_save_table_default_format():
    """Test that DEFAULT_TABLE_STORAGE_FORMAT is csv for backward compatibility."""
    from negmas.helpers.inout import DEFAULT_TABLE_STORAGE_FORMAT

    assert DEFAULT_TABLE_STORAGE_FORMAT == "csv"


def test_save_table_unsupported_data_type(tmp_path):
    """Test that unsupported data types raise ValueError."""
    from negmas.helpers.inout import save_table

    with pytest.raises(ValueError, match="Unsupported data type"):
        save_table("not a valid type", tmp_path / "test.csv")  # type: ignore


def test_save_table_unsupported_list_element(tmp_path):
    """Test that unsupported list element types raise ValueError."""
    from negmas.helpers.inout import save_table

    with pytest.raises(ValueError, match="Unsupported list element type"):
        save_table([1, 2, 3], tmp_path / "test.csv")  # type: ignore


# ===== recalculate_stats Tests =====


def test_scale_min_recalculate_stats_true():
    """Test scale_min with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate initial stats
    scenario.calc_stats()
    assert scenario.stats is not None

    # Scale with recalculate_stats=True (default)
    scenario.scale_min(to=0.0, recalculate_stats=True)

    assert scenario.stats is not None
    # Stats should be recalculated (may or may not change depending on scaling)
    assert isinstance(scenario.stats.opposition, float)


def test_scale_min_recalculate_stats_false():
    """Test scale_min with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Calculate initial stats
    scenario.calc_stats()
    assert scenario.stats is not None

    # Scale with recalculate_stats=False
    scenario.scale_min(to=0.0, recalculate_stats=False)

    # Stats should be invalidated
    assert scenario.stats is None


def test_scale_min_no_stats_does_nothing():
    """Test scale_min when no stats exist doesn't create them."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None
    assert scenario.stats is None

    # Scale without existing stats
    scenario.scale_min(to=0.0)

    # Stats should still be None (not created)
    assert scenario.stats is None


def test_scale_max_recalculate_stats_true():
    """Test scale_max with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.scale_max(to=1.0, recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_scale_max_recalculate_stats_false():
    """Test scale_max with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.scale_max(to=1.0, recalculate_stats=False)

    assert scenario.stats is None


def test_normalize_recalculate_stats_true():
    """Test normalize with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.normalize(to=(0.0, 1.0), recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_normalize_recalculate_stats_false():
    """Test normalize with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.normalize(to=(0.0, 1.0), recalculate_stats=False)

    assert scenario.stats is None


def test_discretize_recalculate_stats_true():
    """Test discretize with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.discretize(levels=5, recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_discretize_recalculate_stats_false():
    """Test discretize with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.discretize(levels=5, recalculate_stats=False)

    assert scenario.stats is None


def test_discretize_updates_ufun_outcome_spaces():
    """Test that discretize updates outcome_space for all ufuns."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.discretize(levels=5)

    # Verify all ufuns have the same outcome_space as the scenario
    for ufun in scenario.ufuns:
        assert ufun.outcome_space is scenario.outcome_space


def test_remove_discounting_recalculate_stats_true():
    """Test remove_discounting with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.remove_discounting(recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_remove_discounting_recalculate_stats_false():
    """Test remove_discounting with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.remove_discounting(recalculate_stats=False)

    assert scenario.stats is None


def test_remove_reserved_values_recalculate_stats_true():
    """Test remove_reserved_values with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Set some reserved values first
    for u in scenario.ufuns:
        u.reserved_value = 0.5

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.remove_reserved_values(r=float("-inf"), recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_remove_reserved_values_recalculate_stats_false():
    """Test remove_reserved_values with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.remove_reserved_values(r=0.0, recalculate_stats=False)

    assert scenario.stats is None


def test_to_single_issue_recalculate_stats_true():
    """Test to_single_issue with recalculate_stats=True recalculates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.to_single_issue(recalculate_stats=True)

    assert scenario.stats is not None
    assert isinstance(scenario.stats.opposition, float)


def test_to_single_issue_recalculate_stats_false():
    """Test to_single_issue with recalculate_stats=False invalidates stats."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    scenario.to_single_issue(recalculate_stats=False)

    assert scenario.stats is None


def test_to_single_issue_updates_ufun_outcome_spaces():
    """Test that to_single_issue creates ufuns with correct outcome_space."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    # Verify we have multiple issues initially
    assert hasattr(scenario.outcome_space, "issues")
    assert len(scenario.outcome_space.issues) > 1

    scenario.to_single_issue()

    # Verify single issue now
    assert len(scenario.outcome_space.issues) == 1

    # Verify all ufuns have outcome_space set (they are newly created ufuns)
    for ufun in scenario.ufuns:
        assert ufun.outcome_space is not None


def test_chained_operations_with_recalculate_stats():
    """Test chaining multiple operations with recalculate_stats parameter."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    # Chain operations - only recalculate at the end
    scenario.scale_max(recalculate_stats=False)
    assert scenario.stats is None

    # Stats are gone, so subsequent operations won't recreate them
    scenario.scale_min(recalculate_stats=True)
    assert scenario.stats is None  # No stats to recalculate

    # Recalculate explicitly
    scenario.calc_stats()
    assert scenario.stats is not None

    # Now chain with recalculate at each step
    scenario.normalize(recalculate_stats=True)
    assert scenario.stats is not None


def test_recalculate_stats_default_is_true():
    """Test that recalculate_stats defaults to True for all methods."""
    file_name = str(files("negmas").joinpath("tests/data/Laptop"))
    scenario = Scenario.from_genius_folder(Path(file_name))
    assert scenario is not None

    scenario.calc_stats()
    assert scenario.stats is not None

    # Without explicit parameter, stats should be recalculated (not invalidated)
    scenario.scale_max()  # Uses default recalculate_stats=True
    assert scenario.stats is not None
