"""Tests for the new mechanism.save() based saving in cartesian tournaments."""

from __future__ import annotations


from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.sao import AspirationNegotiator
from negmas.tournaments.neg.simple.cartesian import run_negotiation


class TestCartesianMechanismSave:
    """Test suite for mechanism.save() based negotiation saving."""

    def test_save_uses_mechanism_save(self, tmp_path):
        """Test that negotiations are saved using mechanism.save()."""
        # Create a simple scenario
        issues = [make_issue([0, 1, 2], "price")]
        os = make_os(issues)
        ufuns = (
            MappingUtilityFunction(lambda x: x[0], outcome_space=os),
            MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os),
        )
        scenario = Scenario(outcome_space=os, ufuns=ufuns)

        # Run a negotiation with saving
        run_negotiation(
            s=scenario,
            partners=(AspirationNegotiator, AspirationNegotiator),
            run_id=1,
            path=tmp_path,
            mechanism_params=dict(n_steps=10),
            storage_optimization="space",
            storage_format="csv",
        )

        # Check that negotiations directory was created
        negotiations_dir = tmp_path / "negotiations"
        assert negotiations_dir.exists()

        # Check that a negotiation file was created
        neg_files = list(negotiations_dir.glob("*.csv"))
        assert len(neg_files) > 0, "No negotiation files found"

        # Check that results directory was created
        results_dir = tmp_path / "results"
        assert results_dir.exists()

        # Check that a result JSON was created
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) > 0, "No result files found"

    def test_save_with_parquet_format(self, tmp_path):
        """Test that negotiations can be saved in parquet format."""
        # Create a simple scenario
        issues = [make_issue([0, 1, 2], "price")]
        os = make_os(issues)
        ufuns = (
            MappingUtilityFunction(lambda x: x[0], outcome_space=os),
            MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os),
        )
        scenario = Scenario(outcome_space=os, ufuns=ufuns)

        # Run a negotiation with parquet format
        run_negotiation(
            s=scenario,
            partners=(AspirationNegotiator, AspirationNegotiator),
            run_id=1,
            path=tmp_path,
            mechanism_params=dict(n_steps=10),
            storage_optimization="space",
            storage_format="parquet",
        )

        # Check that negotiations directory was created
        negotiations_dir = tmp_path / "negotiations"
        assert negotiations_dir.exists()

        # Check that a parquet file was created
        parquet_files = list(negotiations_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No parquet files found"

    def test_save_with_gzip_format(self, tmp_path):
        """Test that negotiations can be saved in gzip format."""
        # Create a simple scenario
        issues = [make_issue([0, 1, 2], "price")]
        os = make_os(issues)
        ufuns = (
            MappingUtilityFunction(lambda x: x[0], outcome_space=os),
            MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os),
        )
        scenario = Scenario(outcome_space=os, ufuns=ufuns)

        # Run a negotiation with gzip format
        run_negotiation(
            s=scenario,
            partners=(AspirationNegotiator, AspirationNegotiator),
            run_id=1,
            path=tmp_path,
            mechanism_params=dict(n_steps=10),
            storage_optimization="balanced",
            storage_format="gzip",
        )

        # Check that negotiations directory was created
        negotiations_dir = tmp_path / "negotiations"
        assert negotiations_dir.exists()

        # Check that a gzip file was created
        gzip_files = list(negotiations_dir.glob("*.csv.gz"))
        assert len(gzip_files) > 0, "No gzip files found"

    def test_save_no_path_does_not_crash(self):
        """Test that negotiations without a path don't crash."""
        # Create a simple scenario
        issues = [make_issue([0, 1, 2], "price")]
        os = make_os(issues)
        ufuns = (
            MappingUtilityFunction(lambda x: x[0], outcome_space=os),
            MappingUtilityFunction(lambda x: 2 - x[0], outcome_space=os),
        )
        scenario = Scenario(outcome_space=os, ufuns=ufuns)

        # Run a negotiation without a path (should not crash)
        record = run_negotiation(
            s=scenario,
            partners=(AspirationNegotiator, AspirationNegotiator),
            run_id=1,
            path=None,
            mechanism_params=dict(n_steps=10),
        )

        assert record is not None
        assert "agreement" in record
