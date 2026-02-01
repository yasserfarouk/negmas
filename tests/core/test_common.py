from __future__ import annotations
import random
import time

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings
import pytest

from negmas import NamedObject
from negmas.helpers import unique_name
from negmas.sao import SAOMechanism, RandomNegotiator, AspirationNegotiator
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction

random.seed(time.perf_counter())

good_attribs = ["current_step", "_current_step", "_Entity__current_step", "_step"]

bad_attribs = ["sdfds", "ewre"]


class WithStep(NamedObject):
    _step = 3


class MyEntity(NamedObject):
    pass


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    exist_ok=st.booleans(),
    single_checkpoint=st.booleans(),
    with_name=st.booleans(),
    with_info=st.booleans(),
    step_attribs=st.tuples(st.sampled_from(good_attribs + bad_attribs)),
)
def test_checkpoint(
    tmp_path, exist_ok, with_name, with_info, single_checkpoint, step_attribs
):
    x = WithStep()

    fname = unique_name("abc", rand_digits=10, add_time=True, sep=".")
    try:
        file_name = x.checkpoint(
            path=tmp_path,
            file_name=fname if with_name else None,
            info={"r": 3} if with_info else None,
            exist_ok=exist_ok,
            single_checkpoint=single_checkpoint,
            step_attribs=step_attribs,
        )
        assert (
            file_name.name.split(".")[0].isnumeric()
            or single_checkpoint
            or all(_ in bad_attribs for _ in set(step_attribs))
            or not any(hasattr(x, _) for _ in step_attribs)
        )
    except ValueError as e:
        if "exist_ok" in str(e):
            assert not exist_ok
        else:
            raise e

    x = MyEntity()

    fname = unique_name("abc", rand_digits=10, add_time=True, sep=".")
    try:
        file_name = x.checkpoint(
            path=tmp_path,
            file_name=fname if with_name else None,
            info={"r": 3} if with_info else None,
            exist_ok=exist_ok,
            single_checkpoint=single_checkpoint,
            step_attribs=step_attribs,
        )
        assert (
            file_name.name.split(".")[0].isnumeric()
            or single_checkpoint
            or all(_ in bad_attribs for _ in set(step_attribs))
            or not any(hasattr(x, _) for _ in step_attribs)
        )
    except ValueError as e:
        if "exist_ok" in str(e):
            assert not exist_ok
        else:
            raise e


# Tests for Mechanism.save()
class TestMechanismSave:
    """Tests for Mechanism.save() method."""

    def test_save_single_file_csv(self, tmp_path):
        """Test saving to a single CSV file."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(tmp_path, "test_single", single_file=True, storage_format="csv")

        assert result.exists()
        assert result.suffix == ".csv"
        assert result.stat().st_size > 0

    def test_save_single_file_gzip(self, tmp_path):
        """Test saving to a gzip-compressed CSV file."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(tmp_path, "test_gzip", single_file=True, storage_format="gzip")

        assert result.exists()
        assert str(result).endswith(".csv.gz")

    def test_save_directory_structure(self, tmp_path):
        """Test saving creates proper directory structure."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(tmp_path, "test_dir", single_file=False, storage_format="csv")

        assert result.is_dir()
        assert (result / "trace.csv").exists()
        assert (result / "config.yaml").exists()
        assert (result / "outcome_stats.yaml").exists()

    def test_save_with_preferences(self, tmp_path):
        """Test saving with negotiator preferences saves scenario."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        result = m.save(tmp_path, "test_prefs", single_file=False, save_scenario=True)

        assert result.is_dir()
        scenario_dir = result / "scenario"
        assert scenario_dir.exists()
        assert any(scenario_dir.iterdir())  # Has files

    def test_save_with_scenario_stats(self, tmp_path):
        """Test saving with scenario statistics."""
        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        result = m.save(
            tmp_path,
            "test_stats",
            single_file=False,
            save_scenario=True,
            save_scenario_stats=True,
        )

        scenario_dir = result / "scenario"
        assert scenario_dir.exists()
        # Stats file should exist
        assert (scenario_dir / "_stats.yaml").exists()

    def test_save_full_trace_source(self, tmp_path):
        """Test saving with full_trace as source."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(
            tmp_path, "test_full_trace", single_file=True, source="full_trace"
        )

        assert result.exists()
        # Read and check columns using pandas to handle parquet format
        import pandas as pd

        df = pd.read_parquet(result)
        columns = list(df.columns)
        assert "time" in columns
        assert "negotiator" in columns
        assert "offer" in columns

    def test_save_with_metadata(self, tmp_path):
        """Test saving with custom metadata."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        metadata = {"experiment": "test", "version": 1}
        result = m.save(tmp_path, "test_meta", single_file=False, metadata=metadata)

        assert (result / "metadata.yaml").exists()

    def test_save_outcome_stats_contains_agreement(self, tmp_path):
        """Test that outcome_stats contains agreement info."""
        import yaml

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(tmp_path, "test_outcome", single_file=False)

        with open(result / "outcome_stats.yaml") as f:
            stats = yaml.safe_load(f)

        assert "agreement" in stats
        assert "broken" in stats
        assert "timedout" in stats
        assert "utilities" in stats

    def test_save_config_contains_mechanism_info(self, tmp_path):
        """Test that config.yaml contains mechanism info."""
        import yaml

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        result = m.save(tmp_path, "test_config", single_file=False)

        with open(result / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert "mechanism_type" in config
        assert "n_negotiators" in config
        assert config["n_negotiators"] == 2
        assert "negotiator_names" in config

    def test_save_overwrite_false(self, tmp_path):
        """Test that overwrite=False doesn't overwrite existing files."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # First save (use csv format for predictable extension)
        m.save(tmp_path, "test_no_overwrite", single_file=True, storage_format="csv")
        first_size = (tmp_path / "test_no_overwrite.csv").stat().st_size

        # Second save should not overwrite
        m.save(
            tmp_path,
            "test_no_overwrite",
            single_file=True,
            storage_format="csv",
            overwrite=False,
            warn_if_existing=False,
        )
        second_size = (tmp_path / "test_no_overwrite.csv").stat().st_size

        assert first_size == second_size

    def test_save_returns_correct_path(self, tmp_path):
        """Test that save returns the correct path."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Single file (use csv format for predictable extension)
        result = m.save(tmp_path, "test_path", single_file=True, storage_format="csv")
        assert result == tmp_path / "test_path.csv"

        # Directory
        result = m.save(tmp_path, "test_dir_path", single_file=False)
        assert result == tmp_path / "test_dir_path"


# Tests for CompletedRun
class TestCompletedRun:
    """Tests for CompletedRun save and load functionality."""

    def test_to_completed_run_basic(self, tmp_path):
        """Test creating a CompletedRun from a mechanism."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # With source=None (default), auto-detects best available source
        completed = m.to_completed_run()

        # SAOMechanism has full_trace_with_utils, so that's what we get
        assert completed.history_type == "full_trace_with_utils"
        assert len(completed.history) > 0
        assert completed.config["mechanism_type"] == "SAOMechanism"
        assert completed.config["n_negotiators"] == 2

    def test_to_completed_run_explicit_history(self, tmp_path):
        """Test creating a CompletedRun with explicit history source."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Explicit source="history" uses the history attribute
        completed = m.to_completed_run(source="history")

        assert completed.history_type == "history"
        assert len(completed.history) > 0
        assert completed.config["mechanism_type"] == "SAOMechanism"
        assert completed.config["n_negotiators"] == 2

    def test_to_completed_run_full_trace(self, tmp_path):
        """Test creating a CompletedRun with full_trace source."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run(source="full_trace")

        assert completed.history_type == "full_trace"
        assert len(completed.history) > 0

    def test_to_completed_run_with_metadata(self, tmp_path):
        """Test creating a CompletedRun with metadata."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        metadata = {"experiment": "test", "version": 1}
        completed = m.to_completed_run(metadata=metadata)

        assert completed.metadata == metadata

    def test_completed_run_save_single_file(self, tmp_path):
        """Test saving a CompletedRun to a single file."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        result = completed.save(tmp_path, "test_single", single_file=True)

        assert result.exists()
        # Default format is now parquet, but check for any valid format
        assert result.suffix in (".csv", ".parquet", ".gz")

    def test_completed_run_save_directory(self, tmp_path):
        """Test saving a CompletedRun to a directory."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        result = completed.save(tmp_path, "test_dir", single_file=False)

        assert result.is_dir()
        # Check for any valid trace format
        trace_exists = (
            (result / "trace.csv").exists()
            or (result / "trace.parquet").exists()
            or (result / "trace.csv.gz").exists()
        )
        assert trace_exists
        assert (result / "run_info.yaml").exists()
        assert (result / "config.yaml").exists()

    def test_completed_run_save_load_roundtrip_single_file(self, tmp_path):
        """Test save and load roundtrip for single file mode."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        save_path = completed.save(tmp_path, "test_roundtrip", single_file=True)

        loaded = CompletedRun.load(save_path)

        # Single file mode has limited info
        assert len(loaded.history) == len(completed.history)

    def test_completed_run_save_load_roundtrip_directory(self, tmp_path):
        """Test save and load roundtrip for directory mode."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        save_path = completed.save(tmp_path, "test_roundtrip_dir", single_file=False)

        loaded = CompletedRun.load(save_path)

        assert loaded.history_type == completed.history_type
        assert loaded.config["mechanism_type"] == completed.config["mechanism_type"]
        assert loaded.config["n_negotiators"] == completed.config["n_negotiators"]

    def test_completed_run_save_load_gzip(self, tmp_path):
        """Test save and load with gzip format."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        save_path = completed.save(
            tmp_path, "test_gzip", single_file=False, storage_format="gzip"
        )

        loaded = CompletedRun.load(save_path)

        assert loaded.history_type == completed.history_type

    def test_completed_run_save_load_parquet(self, tmp_path):
        """Test save and load with parquet format."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        completed = m.to_completed_run()
        save_path = completed.save(
            tmp_path, "test_parquet", single_file=False, storage_format="parquet"
        )

        loaded = CompletedRun.load(save_path)

        assert loaded.history_type == completed.history_type

    def test_completed_run_with_scenario(self, tmp_path):
        """Test CompletedRun with scenario information."""
        from negmas.mechanisms import CompletedRun

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        completed = m.to_completed_run()

        assert completed.scenario is not None

        # Save and load with scenario
        save_path = completed.save(
            tmp_path, "test_scenario", single_file=False, save_scenario=True
        )

        loaded = CompletedRun.load(save_path)

        assert loaded.scenario is not None

    def test_completed_run_load_nonexistent_raises(self, tmp_path):
        """Test that loading from nonexistent path raises FileNotFoundError."""
        from negmas.mechanisms import CompletedRun

        with pytest.raises(FileNotFoundError):
            CompletedRun.load(tmp_path / "nonexistent")

    def test_infer_source_history(self, tmp_path):
        """Test inferring 'history' source type from saved file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with history source
        completed = m.to_completed_run(source="history")
        save_path = completed.save(tmp_path, "test_history", single_file=False)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "history"

    def test_infer_source_trace(self, tmp_path):
        """Test inferring 'trace' source type from saved file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with trace source
        completed = m.to_completed_run(source="trace")
        save_path = completed.save(tmp_path, "test_trace", single_file=False)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "trace"

    def test_infer_source_extended_trace(self, tmp_path):
        """Test inferring 'extended_trace' source type from saved file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with extended_trace source
        completed = m.to_completed_run(source="extended_trace")
        save_path = completed.save(tmp_path, "test_extended", single_file=False)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "extended_trace"

    def test_infer_source_full_trace(self, tmp_path):
        """Test inferring 'full_trace' source type from saved file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with full_trace source
        completed = m.to_completed_run(source="full_trace")
        save_path = completed.save(tmp_path, "test_full_trace", single_file=False)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "full_trace"

    def test_infer_source_full_trace_with_utils(self, tmp_path):
        """Test inferring 'full_trace_with_utils' source type from saved file."""
        from negmas.mechanisms import CompletedRun

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        # Save with full_trace_with_utils source
        completed = m.to_completed_run(source="full_trace_with_utils")
        save_path = completed.save(tmp_path, "test_full_trace_utils", single_file=False)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "full_trace_with_utils"

    def test_infer_source_single_file(self, tmp_path):
        """Test inferring source type from single file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with full_trace source as single file
        completed = m.to_completed_run(source="full_trace")
        save_path = completed.save(tmp_path, "test_single", single_file=True)

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "full_trace"

    def test_infer_source_parquet(self, tmp_path):
        """Test inferring source type from parquet file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with trace source as parquet
        completed = m.to_completed_run(source="trace")
        save_path = completed.save(
            tmp_path, "test_parquet", single_file=False, storage_format="parquet"
        )

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "trace"

    def test_infer_source_gzip(self, tmp_path):
        """Test inferring source type from gzip file."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Save with extended_trace source as gzip
        completed = m.to_completed_run(source="extended_trace")
        save_path = completed.save(
            tmp_path, "test_gzip", single_file=False, storage_format="gzip"
        )

        inferred = CompletedRun.infer_source(save_path)
        assert inferred == "extended_trace"

    def test_infer_source_nonexistent_raises(self, tmp_path):
        """Test that inferring from nonexistent path raises FileNotFoundError."""
        from negmas.mechanisms import CompletedRun

        with pytest.raises(FileNotFoundError):
            CompletedRun.infer_source(tmp_path / "nonexistent")

    def test_save_includes_optimality_stats_in_outcome_stats(self, tmp_path):
        """Test that save() merges agreement_stats into outcome_stats.yaml."""
        from negmas.preferences.ops import OutcomeOptimality
        from negmas.helpers.inout import load

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        # Create completed run with agreement_stats
        agreement_stats = OutcomeOptimality(
            pareto_optimality=0.85,
            nash_optimality=0.76,
            kalai_optimality=0.80,
            modified_kalai_optimality=0.81,
            max_welfare_optimality=0.74,
            ks_optimality=0.79,
            modified_ks_optimality=0.78,
        )

        completed = m.to_completed_run(agreement_stats=agreement_stats)
        save_path = completed.save(
            tmp_path, "test_optimality", single_file=False, save_agreement_stats=True
        )

        # Verify outcome_stats.yaml contains optimality stats
        outcome_stats = load(save_path / "outcome_stats.yaml")
        assert "pareto_optimality" in outcome_stats
        assert "nash_optimality" in outcome_stats
        assert outcome_stats["pareto_optimality"] == 0.85
        assert outcome_stats["nash_optimality"] == 0.76

    def test_load_extracts_agreement_from_outcome_stats(self, tmp_path):
        """Test that load() can get agreement from outcome_stats when not in run_info."""
        from negmas.mechanisms import CompletedRun
        from negmas.helpers.inout import load, dump

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        completed = m.to_completed_run()
        save_path = completed.save(tmp_path, "test_agreement", single_file=False)

        # Modify run_info.yaml to remove agreement (simulating old format)
        run_info = load(save_path / "run_info.yaml")
        original_agreement = run_info.get("agreement")
        if "agreement" in run_info:
            del run_info["agreement"]
        dump(run_info, save_path / "run_info.yaml")

        # Make sure outcome_stats has the agreement
        outcome_stats = load(save_path / "outcome_stats.yaml")
        outcome_stats["agreement"] = original_agreement
        dump(outcome_stats, save_path / "outcome_stats.yaml")

        # Load and verify agreement is retrieved from outcome_stats
        loaded = CompletedRun.load(save_path)
        assert loaded.agreement == original_agreement

    def test_load_extracts_agreement_stats_from_outcome_stats(self, tmp_path):
        """Test that load() extracts agreement_stats from outcome_stats.yaml."""
        from negmas.mechanisms import CompletedRun
        from negmas.preferences.ops import OutcomeOptimality

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        # Create completed run with agreement_stats
        agreement_stats = OutcomeOptimality(
            pareto_optimality=0.85,
            nash_optimality=0.76,
            kalai_optimality=0.80,
            modified_kalai_optimality=0.81,
            max_welfare_optimality=0.74,
            ks_optimality=0.79,
            modified_ks_optimality=0.78,
        )

        completed = m.to_completed_run(agreement_stats=agreement_stats)
        save_path = completed.save(
            tmp_path,
            "test_stats_from_outcome",
            single_file=False,
            save_agreement_stats=True,
        )

        # Load and verify agreement_stats is extracted from outcome_stats
        loaded = CompletedRun.load(save_path)
        assert loaded.agreement_stats is not None
        assert abs(loaded.agreement_stats.pareto_optimality - 0.85) < 0.001
        assert abs(loaded.agreement_stats.nash_optimality - 0.76) < 0.001

    def test_save_does_not_create_agreement_stats_yaml(self, tmp_path):
        """Test that save() does NOT create a separate agreement_stats.yaml file."""
        from negmas.preferences.ops import OutcomeOptimality

        issues = [make_issue(10, "price")]
        os = make_os(issues)
        m = SAOMechanism(issues=issues, n_steps=20)

        u1 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)

        m.add(AspirationNegotiator(name="buyer", preferences=u1))
        m.add(AspirationNegotiator(name="seller", preferences=u2))
        m.run()

        # Create completed run with agreement_stats
        agreement_stats = OutcomeOptimality(
            pareto_optimality=0.85,
            nash_optimality=0.76,
            kalai_optimality=0.80,
            modified_kalai_optimality=0.81,
            max_welfare_optimality=0.74,
            ks_optimality=0.79,
            modified_ks_optimality=0.78,
        )

        completed = m.to_completed_run(agreement_stats=agreement_stats)
        save_path = completed.save(
            tmp_path,
            "test_no_agreement_stats_yaml",
            single_file=False,
            save_agreement_stats=True,
        )

        # agreement_stats.yaml should NOT exist (all stats are in outcome_stats.yaml)
        assert not (save_path / "agreement_stats.yaml").exists()
        # But outcome_stats.yaml should exist and contain the optimality stats
        assert (save_path / "outcome_stats.yaml").exists()


class TestCompletedRunConvert:
    """Tests for CompletedRun.convert() method."""

    def test_convert_full_trace_to_trace(self, tmp_path):
        """Test converting from full_trace to trace format."""

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="full_trace")
        assert run.history_type == "full_trace"

        converted = run.convert("trace")
        assert converted.history_type == "trace"
        assert len(converted.history) == len(run.history)
        # trace format is (negotiator, offer)
        for entry in converted.history:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_convert_full_trace_to_extended_trace(self, tmp_path):
        """Test converting from full_trace to extended_trace format."""

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="full_trace")
        converted = run.convert("extended_trace")

        assert converted.history_type == "extended_trace"
        assert len(converted.history) == len(run.history)
        # extended_trace format is (step, negotiator, offer)
        for entry in converted.history:
            assert isinstance(entry, tuple)
            assert len(entry) == 3

    def test_convert_trace_to_full_trace(self, tmp_path):
        """Test converting from trace to full_trace (with None values)."""

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="trace")
        converted = run.convert("full_trace")

        assert converted.history_type == "full_trace"
        assert len(converted.history) == len(run.history)
        # Missing fields should be None
        for entry in converted.history:
            assert entry.time is None
            assert entry.relative_time is None
            assert entry.negotiator is not None  # This was preserved
            assert entry.offer is not None  # This was preserved

    def test_convert_to_full_trace_with_utils_requires_scenario(self, tmp_path):
        """Test that converting to full_trace_with_utils requires a scenario."""
        from negmas.mechanisms import CompletedRun

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="trace")
        # Clear scenario to simulate no utility functions
        run = CompletedRun(
            history=run.history,
            history_type=run.history_type,
            scenario=None,
            agreement=run.agreement,
            agreement_stats=run.agreement_stats,
            outcome_stats=run.outcome_stats,
            config=run.config,
            metadata=run.metadata,
        )

        with pytest.raises(ValueError, match="Cannot convert to full_trace_with_utils"):
            run.convert("full_trace_with_utils")

    def test_convert_to_full_trace_with_utils_with_scenario(self, tmp_path):
        """Test converting to full_trace_with_utils with a scenario."""
        from negmas.preferences import LinearAdditiveUtilityFunction

        issues = [make_issue(10, "price")]
        m = SAOMechanism(issues=issues, n_steps=10)
        u1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        m.add(RandomNegotiator(name="n1"), ufun=u1)
        m.add(RandomNegotiator(name="n2"), ufun=u2)
        m.run()

        run = m.to_completed_run(source="full_trace")
        converted = run.convert("full_trace_with_utils")

        assert converted.history_type == "full_trace_with_utils"
        # Should have 9 TraceElement fields + 2 utility values
        if converted.history:
            assert len(converted.history[0]) == 11

    def test_convert_with_external_scenario(self, tmp_path):
        """Test converting with an externally provided scenario."""
        from negmas.preferences import LinearAdditiveUtilityFunction
        from negmas.inout import Scenario

        issues = [make_issue(10, "price")]
        m = SAOMechanism(issues=issues, n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        # Run without ufuns
        run = m.to_completed_run(source="trace")
        assert run.scenario is None or not run.scenario.ufuns

        # Create external scenario with ufuns
        u1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        scenario = Scenario(outcome_space=m.outcome_space, ufuns=(u1, u2))

        # Convert with external scenario
        converted = run.convert("full_trace_with_utils", scenario=scenario)
        assert converted.history_type == "full_trace_with_utils"

    def test_convert_same_format_returns_copy(self, tmp_path):
        """Test that converting to same format returns a copy."""

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="full_trace")
        converted = run.convert("full_trace")

        assert converted.history_type == run.history_type
        assert converted is not run  # Should be a copy
        assert converted.history is not run.history

    def test_convert_invalid_target_raises(self, tmp_path):
        """Test that invalid target format raises ValueError."""

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        run = m.to_completed_run(source="full_trace")

        with pytest.raises(ValueError, match="Invalid target format"):
            run.convert("invalid_format")

    def test_convert_preserves_scenario(self, tmp_path):
        """Test that convert preserves the scenario."""
        from negmas.preferences import LinearAdditiveUtilityFunction

        issues = [make_issue(10, "price")]
        m = SAOMechanism(issues=issues, n_steps=10)
        u1 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        u2 = LinearAdditiveUtilityFunction.random(issues=issues, reserved_value=0.0)
        m.add(RandomNegotiator(name="n1"), ufun=u1)
        m.add(RandomNegotiator(name="n2"), ufun=u2)
        m.run()

        run = m.to_completed_run(source="full_trace")
        assert run.scenario is not None

        converted = run.convert("trace")
        assert converted.scenario is not None
        assert converted.scenario is run.scenario  # Same scenario object


# Tests for load_table
class TestLoadTable:
    """Tests for load_table function."""

    def test_load_table_csv(self, tmp_path):
        """Test loading a CSV file."""
        from negmas.helpers.inout import save_table, load_table

        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = tmp_path / "test.csv"
        save_table(data, path, storage_format="csv")

        result = load_table(path)

        assert len(result) == 2
        assert list(result.columns) == ["a", "b"]

    def test_load_table_csv_as_records(self, tmp_path):
        """Test loading a CSV file as list of dicts."""
        from negmas.helpers.inout import save_table, load_table

        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = tmp_path / "test.csv"
        save_table(data, path, storage_format="csv")

        result = load_table(path, as_dataframe=False)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_load_table_gzip(self, tmp_path):
        """Test loading a gzip-compressed CSV file."""
        from negmas.helpers.inout import save_table, load_table

        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = tmp_path / "test.csv"
        actual_path = save_table(data, path, storage_format="gzip")

        result = load_table(actual_path)

        assert len(result) == 2

    def test_load_table_parquet(self, tmp_path):
        """Test loading a parquet file."""
        from negmas.helpers.inout import save_table, load_table

        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = tmp_path / "test.csv"
        actual_path = save_table(data, path, storage_format="parquet")

        result = load_table(actual_path)

        assert len(result) == 2

    def test_load_table_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent file raises FileNotFoundError."""
        from negmas.helpers.inout import load_table

        with pytest.raises(FileNotFoundError):
            load_table(tmp_path / "nonexistent.csv")

    def test_load_table_unsupported_extension_raises(self, tmp_path):
        """Test that loading unsupported extension raises ValueError."""
        from negmas.helpers.inout import load_table

        # Create a file with unsupported extension
        path = tmp_path / "test.xyz"
        path.write_text("data")

        with pytest.raises(ValueError):
            load_table(path)


# Tests for per-negotiator saving deprecation
class TestPerNegotiatorSaving:
    """Tests for per-negotiator saving mode deprecation."""

    def test_per_negotiator_emits_deprecation_warning(self, tmp_path):
        """Test that per_negotiator=True emits a deprecation warning."""
        import warnings

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="buyer"))
        m.add(RandomNegotiator(name="seller"))
        m.run()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.save(tmp_path, "test_per_neg", per_negotiator=True, source="full_trace")

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                x for x in w if "per_negotiator" in str(x.message).lower()
            ]
            assert len(deprecation_warnings) >= 1, (
                "Expected deprecation warning for per_negotiator"
            )

    def test_per_negotiator_does_not_create_directory(self, tmp_path):
        """Test that per_negotiator=True no longer creates negotiator_behavior directory."""
        import warnings

        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="buyer"))
        m.add(RandomNegotiator(name="seller"))
        m.run()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress the deprecation warning
            result = m.save(
                tmp_path, "test_per_neg", per_negotiator=True, source="full_trace"
            )

        # negotiator_behavior directory should NOT be created
        assert not (result / "negotiator_behavior").exists()


class TestStringHelpers:
    """Tests for string helper functions."""

    def test_shorten_keys_basic(self):
        """Test shorten_keys with basic dict."""
        from negmas.helpers.strings import shorten_keys

        d = {"sdfsd": 463, "dfsdf": 4, "sdfds": "sdfwer"}
        result = shorten_keys(d)

        # All keys should be unique
        assert len(set(result.keys())) == len(result)
        # Values should be preserved
        assert set(result.values()) == set(d.values())

    def test_shorten_keys_empty(self):
        """Test shorten_keys with empty dict."""
        from negmas.helpers.strings import shorten_keys

        assert shorten_keys({}) == {}
        assert shorten_keys(None) == {}

    def test_shorten_keys_single_item(self):
        """Test shorten_keys with single item dict."""
        from negmas.helpers.strings import shorten_keys

        d = {"longkeyname": 42}
        result = shorten_keys(d)

        assert len(result) == 1
        assert 42 in result.values()

    def test_shorten_keys_in_string_basic(self):
        """Test shorten_keys_in_string with basic string."""
        from negmas.helpers.strings import shorten_keys_in_string

        s = "[sdfsd=463,dfsdf=4,sdfds=sdfwer]"
        result = shorten_keys_in_string(s)

        # Result should be shorter or equal length
        assert len(result) <= len(s)
        # Should still have bracket format
        assert result.startswith("[")
        assert result.endswith("]")

    def test_shorten_keys_in_string_empty(self):
        """Test shorten_keys_in_string with empty/no-match string."""
        from negmas.helpers.strings import shorten_keys_in_string

        assert shorten_keys_in_string("") == ""
        assert shorten_keys_in_string("no matches here") == "no matches here"

    def test_shorten_keys_in_string_max_compression_true(self):
        """Test shorten_keys_in_string with max_compression=True (default)."""
        from negmas.helpers.strings import shorten_keys_in_string

        s = "[first_name=John,last_name=Doe]"
        result = shorten_keys_in_string(s, max_compression=True)

        # With max compression, keys should be as short as possible
        assert result.startswith("[")
        assert result.endswith("]")
        assert len(result) < len(s)
        # Values should be preserved
        assert "John" in result
        assert "Doe" in result

    def test_shorten_keys_in_string_max_compression_false(self):
        """Test shorten_keys_in_string with max_compression=False."""
        from negmas.helpers.strings import shorten_keys_in_string

        s = "[first_name=John,last_name=Doe]"
        result_max = shorten_keys_in_string(s, max_compression=True)
        result_min = shorten_keys_in_string(s, max_compression=False)

        # With minimal compression, result should be longer than max compression
        assert len(result_min) >= len(result_max)
        # Values should still be preserved
        assert "John" in result_min
        assert "Doe" in result_min
        assert result_min.startswith("[")
        assert result_min.endswith("]")

    def test_shorten_keys_in_string_max_compression_none_short(self):
        """Test shorten_keys_in_string with max_compression=None and short result."""
        from negmas.helpers.strings import shorten_keys_in_string

        # Short input that won't exceed max_length with minimal compression
        s = "[a=1,b=2]"
        result = shorten_keys_in_string(s, max_compression=None, max_length=100)

        # Should use minimal compression since result is short
        assert result.startswith("[")
        assert result.endswith("]")
        assert "1" in result
        assert "2" in result

    def test_shorten_keys_in_string_max_compression_none_long(self):
        """Test shorten_keys_in_string with max_compression=None and long result."""
        from negmas.helpers.strings import shorten_keys_in_string

        # Longer input with similar prefixes
        s = "[parameter_alpha=0.1,parameter_beta=0.2,parameter_gamma=0.3]"

        # With a very small max_length, should fall back to max compression
        result_adaptive = shorten_keys_in_string(s, max_compression=None, max_length=20)
        result_max = shorten_keys_in_string(s, max_compression=True)

        # Adaptive should produce same result as max compression when exceeding max_length
        assert result_adaptive == result_max

    def test_shorten_keys_in_string_max_compression_none_threshold(self):
        """Test shorten_keys_in_string adaptive behavior at threshold."""
        from negmas.helpers.strings import shorten_keys_in_string

        s = "[first_name=John,last_name=Doe]"
        result_min = shorten_keys_in_string(s, max_compression=False)
        result_max = shorten_keys_in_string(s, max_compression=True)

        # With max_length larger than minimal result, should use minimal compression
        result_large_threshold = shorten_keys_in_string(
            s, max_compression=None, max_length=len(result_min) + 10
        )
        assert result_large_threshold == result_min

        # With max_length smaller than minimal result, should use max compression
        result_small_threshold = shorten_keys_in_string(
            s, max_compression=None, max_length=len(result_min) - 1
        )
        assert result_small_threshold == result_max

    def test_encode_params_basic(self):
        """Test encode_params with basic dict."""
        from negmas.helpers.strings import encode_params

        params = {"key1": "val1", "key2": 42}
        result = encode_params(params)

        assert result.startswith("[")
        assert result.endswith("]")
        assert "key1=val1" in result
        assert "key2=42" in result

    def test_encode_params_empty(self):
        """Test encode_params with empty/None."""
        from negmas.helpers.strings import encode_params

        assert encode_params({}) == ""
        assert encode_params(None) == ""

    def test_shortest_unique_names_with_params(self):
        """Test that shortest_unique_names works when names differ only by params."""
        from negmas.helpers.strings import shortest_unique_names, encode_params

        # Same class name but different params
        names = [
            "MyNegotiator" + encode_params({"alpha": 0.1}),
            "MyNegotiator" + encode_params({"alpha": 0.5}),
            "MyNegotiator" + encode_params({"alpha": 0.9}),
        ]
        result = shortest_unique_names(names)

        # All results should be unique
        assert len(set(result)) == len(result)


class TestTraceElementTextData:
    """Tests for text and data fields in TraceElement."""

    def test_full_trace_has_text_and_data_fields(self):
        """Test that full_trace elements have text and data fields."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        trace = m.full_trace
        assert len(trace) > 0

        # Check that TraceElement has text and data
        first_element = trace[0]
        assert hasattr(first_element, "text")
        assert hasattr(first_element, "data")

    def test_negotiator_full_trace_has_text_and_data(self):
        """Test that negotiator_full_trace returns text and data."""
        m = SAOMechanism(outcomes=[(i,) for i in range(10)], n_steps=10)
        m.add(RandomNegotiator(name="n1"))
        m.add(RandomNegotiator(name="n2"))
        m.run()

        neg_trace = m.negotiator_full_trace(m.negotiators[0].id)

        if len(neg_trace) > 0:
            # Each element should be (time, relative_time, step, offer, response, text, data)
            first_element = neg_trace[0]
            assert len(first_element) == 7

    def test_trace_element_members_includes_text_data(self):
        """Test that TRACE_ELEMENT_MEMBERS includes text and data."""
        from negmas.common import TRACE_ELEMENT_MEMBERS

        assert "text" in TRACE_ELEMENT_MEMBERS
        assert "data" in TRACE_ELEMENT_MEMBERS
