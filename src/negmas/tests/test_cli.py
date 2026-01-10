from __future__ import annotations
import functools
from unittest import mock

import pytest
from click.testing import CliRunner
from typer.testing import CliRunner as TyperRunner
from pathlib import Path

from negmas.scripts.app import cli as main
from negmas.scripts.negotiate import (
    app,
    GENIUSMARKER,
    ANLMARKER,
    BOAMARKER,
    MAPMARKER,
    LLMMARKER,
    NEGOLOGMARKER,
    GAMARKER,
    get_negotiator,
    make_llm_negotiator,
    make_negolog_negotiator,
    make_ga_negotiator,
)


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert len(result.output) > 0
    # Exit code 2 is expected when no subcommand is provided (shows usage)
    assert result.exit_code == 2


def test_negotiate_app_nstpes():
    runner = TyperRunner()
    result = runner.invoke(
        app, ["--no-stats", "--no-plot", "--no-rank-stats", "-s", "20"]
    )
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert "Agreement" in result.output
    assert "Utilities" in result.output
    assert "Advantages" in result.output


def test_negotiate_app_tlimit():
    runner = TyperRunner()
    result = runner.invoke(
        app, ["--no-stats", "--no-plot", "--no-rank-stats", "-t", "3"]
    )
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert "Agreement" in result.output
    assert "Utilities" in result.output
    assert "Advantages" in result.output


def test_negotiate_app_rank_stats():
    runner = TyperRunner()
    result = runner.invoke(app, ["--no-stats", "--no-plot", "--rank-stats", "-s", "10"])
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert "Agreement" in result.output
    assert "Utilities" in result.output
    assert "Advantages" in result.output
    assert "Ordinal Nash Distance" in result.output
    assert "Ordinal Kalai Distance" in result.output
    assert "Ordinal Modified Kalai Distance" in result.output
    assert "Ordinal Max Welfare Distance" in result.output


def test_negotiate_app_stats():
    runner = TyperRunner()
    result = runner.invoke(app, ["--stats", "--no-plot", "--no-rank-stats", "-s", "10"])
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert "Agreement" in result.output
    assert "Utilities" in result.output
    assert "Advantages" in result.output
    assert "Nash Distance" in result.output
    assert "Kalai Distance" in result.output
    assert "Modified Kalai Distance" in result.output
    assert "Max Welfare Distance" in result.output


def test_negotiate_app_save(tmp_path):
    base: Path = tmp_path / "tstneg"
    runner = TyperRunner()
    result = runner.invoke(
        app,
        [
            "--stats",
            "--no-plot-show",
            "--rank-stats",
            "-s",
            "10",
            "--save-path",
            str(base),
        ],
    )
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert "Agreement" in result.output
    assert "Utilities" in result.output
    assert "Advantages" in result.output
    assert "Ordinal Nash Distance" in result.output
    assert "Ordinal Kalai Distance" in result.output
    assert "Ordinal Modified Kalai Distance" in result.output
    assert "Ordinal Max Welfare Distance" in result.output
    assert "Nash Distance" in result.output
    assert "Kalai Distance" in result.output
    assert "Modified Kalai Distance" in result.output
    assert "Max Welfare Distance" in result.output
    for fname in (
        "history.csv",
        "domain.yml",
        "u1.yml",
        "u2.yml",
        "session.json",
        "stats.json",
    ):
        assert (base / fname).exists()
        assert (base / fname).is_file()
        assert (base / fname).stat().st_size > 10


# ============================================================================
# Tests for negotiate script markers and get_negotiator function
# ============================================================================


class TestNegotiatorMarkers:
    """Test that all marker constants are correctly defined."""

    def test_genius_marker(self):
        assert GENIUSMARKER == "genius"

    def test_anl_marker(self):
        assert ANLMARKER == "anl"

    def test_boa_marker(self):
        assert BOAMARKER == "boa"

    def test_map_marker(self):
        assert MAPMARKER == "map"

    def test_llm_marker(self):
        assert LLMMARKER == "llm"

    def test_negolog_marker(self):
        assert NEGOLOGMARKER == "negolog"

    def test_ga_marker(self):
        assert GAMARKER == "ga"


class TestGetNegotiator:
    """Test the get_negotiator function with various prefixes."""

    def test_get_negotiator_returns_class_for_builtin(self):
        """Test that get_negotiator returns a class for built-in negotiators."""
        result = get_negotiator("AspirationNegotiator")
        # Should return a class or callable
        assert callable(result)

    def test_get_negotiator_returns_class_for_full_path(self):
        """Test that get_negotiator works with fully qualified class names."""
        result = get_negotiator("negmas.sao.negotiators.AspirationNegotiator")
        assert callable(result)

    def test_get_negotiator_with_adapter(self):
        """Test that get_negotiator works with adapter syntax (adapter/negotiator)."""
        # TAUNegotiatorAdapter is the correct adapter class name
        result = get_negotiator("TAUNegotiatorAdapter/AspirationNegotiator")
        assert callable(result)
        # Should be a partial function that creates an adapter
        assert isinstance(result, functools.partial)

    def test_get_negotiator_boa_prefix(self):
        """Test that get_negotiator works with boa: prefix."""
        # Test with a minimal BOA spec
        result = get_negotiator("boa:offering=GTimeDependentOffering")
        assert callable(result)
        assert isinstance(result, functools.partial)

    def test_get_negotiator_map_prefix(self):
        """Test that get_negotiator works with map: prefix."""
        # Test with a minimal MAP spec
        result = get_negotiator("map:offering=GTimeDependentOffering")
        assert callable(result)
        assert isinstance(result, functools.partial)

    def test_get_negotiator_anl_prefix(self):
        """Test that get_negotiator returns partial for anl: prefix."""
        result = get_negotiator("anl.SomeAgent")
        assert callable(result)
        assert isinstance(result, functools.partial)

    def test_get_negotiator_llm_prefix(self):
        """Test that get_negotiator returns partial for llm: prefix."""
        result = get_negotiator("llm:SomeNegotiator")
        assert callable(result)
        assert isinstance(result, functools.partial)

    def test_get_negotiator_negolog_prefix(self):
        """Test that get_negotiator returns partial for negolog: prefix."""
        result = get_negotiator("negolog:SomeNegotiator")
        assert callable(result)
        assert isinstance(result, functools.partial)

    def test_get_negotiator_ga_prefix(self):
        """Test that get_negotiator returns partial for ga: prefix."""
        result = get_negotiator("ga:SomeAgent")
        assert callable(result)
        assert isinstance(result, functools.partial)


class TestExternalPackageNegotiators:
    """Test external package negotiator factory functions."""

    def test_make_llm_negotiator_missing_package(self):
        """Test that make_llm_negotiator shows helpful error when package missing."""
        # Mock builtins.__import__ to raise ImportError for negmas_llm
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "negmas_llm":
                raise ImportError("No module named 'negmas_llm'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                make_llm_negotiator("SomeClass")
            assert exc_info.value.code == 1

    def test_make_negolog_negotiator_missing_package(self):
        """Test that make_negolog_negotiator shows helpful error when package missing."""
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "negmas_negolog":
                raise ImportError("No module named 'negmas_negolog'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                make_negolog_negotiator("SomeClass")
            assert exc_info.value.code == 1

    def test_make_ga_negotiator_missing_package(self):
        """Test that make_ga_negotiator shows helpful error when package missing."""
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "genius_agents":
                raise ImportError("No module named 'genius_agents'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                make_ga_negotiator("SomeClass")
            assert exc_info.value.code == 1


class TestNegotiateAppWithPrefixes:
    """Test negotiate CLI with various negotiator prefixes."""

    def test_negotiate_with_specific_negotiators(self):
        """Test running negotiate with specific negotiator types."""
        runner = TyperRunner()
        result = runner.invoke(
            app,
            [
                "-n",
                "AspirationNegotiator",
                "-n",
                "NaiveTitForTatNegotiator",
                "--no-stats",
                "--no-plot",
                "--no-rank-stats",
                "-s",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "Agreement" in result.output or "Disagreement" in result.output

    def test_negotiate_help_shows_all_prefixes(self):
        """Test that the help text mentions all available prefixes."""
        runner = TyperRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Verify key prefixes are mentioned
        assert "genius:" in result.output
        assert "anl:" in result.output
        assert "llm:" in result.output
        assert "negolog:" in result.output
        assert "ga:" in result.output
