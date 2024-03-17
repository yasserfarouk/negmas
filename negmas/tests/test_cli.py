from __future__ import annotations
from click.testing import CliRunner
from typer.testing import CliRunner as TyperRunner
from pathlib import Path

from negmas.scripts.app import cli as main
from negmas.scripts.negotiate import app


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert len(result.output) > 0
    assert result.exit_code == 0


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
