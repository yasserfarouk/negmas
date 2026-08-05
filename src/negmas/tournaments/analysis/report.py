"""One-call orchestration: run every post-tournament analysis method in this
package over a tournament result and optionally save the results (and plots)
to disk.

This is the single entry point meant for both:
    - automatic use from ``cartesian_tournament(..., run_analysis=True)``
      (and, where wired, situated tournament runners), and
    - fully post-hoc use on any tournament result you already have, whether
      in memory (a ``SimpleTournamentResults``/situated ``TournamentResults``),
      on disk (a path to a saved cartesian tournament), or as an
      already-normalized DataFrame -- see
      :func:`negmas.tournaments.analysis.loaders.load_records` for exactly
      what is accepted.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from attrs import define

from .dynamics import ReplicatorTrajectory, symmetric_replicator_dynamics
from .equilibria import PureNashResult, egta_evaluation, nash_averaging_ranking
from .loaders import load_records
from .payoff import PayoffTable, build_payoff_table, iterated_elimination
from .ranking import sequential_elimination_ranking, tournament_evaluation_ranking
from .significance import apply_corrections, normality_tests, pairwise_tests

__all__ = ["TournamentAnalysisReport", "analyze_tournament"]


@define
class TournamentAnalysisReport:
    """Bundles every result computed by :func:`analyze_tournament`."""

    records: pd.DataFrame
    """The common analysis DataFrame the report was built from."""
    payoff: PayoffTable
    """The empirical payoff matrix."""
    has_self_play: bool
    """Whether the payoff matrix diagonal (self-play) was available."""
    tournament_ranking: pd.DataFrame
    """Baseline Tournament Evaluation ranking (always available)."""
    sequential_elimination_ranking: pd.DataFrame | None
    """Sequential Elimination Ranking; None if there was no self-play data."""
    egta: PureNashResult | None
    """EGTA pure symmetric Nash evaluation; None if there was no self-play data."""
    nash_averaging: pd.DataFrame | None
    """NE-regret/Nash-averaging ranking; None if unavailable (no self-play,
    or no pure symmetric Nash equilibrium to use as the reference profile)."""
    iterated_elimination_survivors: list[str] | None
    """Strategies surviving iterated elimination of strictly dominated
    strategies; None if there was no self-play data."""
    replicator_dynamics: ReplicatorTrajectory | None
    """Symmetric replicator dynamics trajectory; None if there was no
    self-play data."""
    significance_unpaired: pd.DataFrame
    """Pairwise significance test (unpaired) with Bonferroni/Holm/BH corrections."""
    significance_paired: pd.DataFrame | None
    """Pairwise significance test (paired by scenario); None if no two
    strategies shared any scenario."""
    normality: pd.DataFrame
    """Per-strategy normality diagnostics."""
    output_dir: Path | None = None
    """Directory results (and plots) were saved to, if any."""


def _default_original_scores(source: Any) -> pd.DataFrame | None:
    """Best-effort extraction of a plain strategy/score table straight from
    ``source``, for the "raw tournament result" bar-chart plot."""
    from negmas.tournaments.neg.simple.cartesian import SimpleTournamentResults
    from negmas.tournaments.tournaments import TournamentResults

    try:
        if isinstance(source, (str, Path)):
            source = SimpleTournamentResults.load(Path(source))
        if isinstance(source, SimpleTournamentResults):
            fs = source.final_scores
            if fs is not None and len(fs) and "strategy" in fs.columns:
                return fs
        elif isinstance(source, TournamentResults):
            ts = source.total_scores
            if ts is None or len(ts) == 0:
                return None
            df = ts.reset_index()
            if df.shape[1] == 2:
                df.columns = ["strategy", "score"]
                return df
    except Exception:
        return None
    return None


def analyze_tournament(
    source: Any,
    output_dir: str | Path | None = None,
    metric: str = "advantage",
    stat: str = "mean",
    alpha: float = 0.05,
    significance_test: str = "ttest",
    make_plots: bool = False,
    original_scores: pd.DataFrame | None = None,
    replicator_t_max: float = 100.0,
    replicator_n_steps: int = 200,
) -> TournamentAnalysisReport:
    """Runs every post-tournament analysis method in this package -- EGTA
    pure-Nash evaluation, Sequential Elimination Ranking, Tournament
    Evaluation, iterated dominance elimination, replicator dynamics, pairwise
    significance testing and normality diagnostics -- over a tournament
    result, and optionally saves everything (as CSVs, plus PNG plots) to
    ``output_dir``.

    Args:
        source: anything accepted by
            :func:`negmas.tournaments.analysis.loaders.load_records`: an
            in-memory ``SimpleTournamentResults`` or situated
            ``TournamentResults``, a path to a saved cartesian tournament, or
            an already-normalized DataFrame.
        output_dir: if given, every result is saved as a CSV (plus PNG plots
            under ``output_dir/plots`` if ``make_plots``) into this directory.
        metric: which column to build the payoff matrix and run the
            significance tests over (typically ``"advantage"`` or ``"utility"``).
        stat: aggregation statistic for the payoff matrix (see
            :func:`negmas.tournaments.analysis.payoff.build_payoff_table`).
        alpha: significance level used by the pairwise tests' corrections.
        significance_test: one of ``"ttest"``, ``"ranksum"``, ``"ks"`` (see
            :func:`negmas.tournaments.analysis.significance.pairwise_tests`).
        make_plots: if True (and ``output_dir`` is given), saves a PNG for
            every applicable plot, including a bar chart of the raw
            tournament scores (see ``original_scores``).
        original_scores: an optional name/score DataFrame (e.g.
            ``SimpleTournamentResults.final_scores`` or a reset-index'd
            ``TournamentResults.total_scores``) to plot as a plain bar chart
            of the raw tournament result, for comparison against the
            game-theoretic rankings. If not given, it is auto-detected from
            ``source`` when possible.
        replicator_t_max: integration horizon for replicator dynamics.
        replicator_n_steps: number of time points for replicator dynamics.

    Returns:
        A :class:`TournamentAnalysisReport`. Methods that require the payoff
        matrix diagonal (self-play) are set to None (with a warning) if the
        tournament has no self-play data.
    """
    records = load_records(source)
    if original_scores is None:
        original_scores = _default_original_scores(source)

    try:
        payoff = build_payoff_table(
            records, metric=metric, stat=stat, require_self_play=True
        )
        has_self_play = True
    except ValueError as e:
        warnings.warn(
            f"{e} Skipping EGTA evaluation, Sequential Elimination Ranking, "
            "Nash-averaging, iterated elimination and replicator dynamics "
            "(all require self-play data); only Tournament Evaluation and "
            "the statistical tests will be computed.",
            stacklevel=2,
        )
        payoff = build_payoff_table(
            records, metric=metric, stat=stat, require_self_play=False
        )
        has_self_play = False

    tournament_ranking = tournament_evaluation_ranking(payoff)

    ser = egta = nash_avg = traj = survivors = None
    if has_self_play:
        ser = sequential_elimination_ranking(payoff)
        egta = egta_evaluation(payoff)
        try:
            nash_avg = nash_averaging_ranking(payoff)
        except ValueError:
            nash_avg = None
        survivors = iterated_elimination(payoff).strategies
        traj = symmetric_replicator_dynamics(
            payoff, t_max=replicator_t_max, n_steps=replicator_n_steps
        )

    sig_unpaired = apply_corrections(
        pairwise_tests(records, metric=metric, test=significance_test, paired=False),
        alpha=alpha,
    )
    try:
        paired_df = pairwise_tests(
            records, metric=metric, test=significance_test, paired=True
        )
        sig_paired = (
            apply_corrections(paired_df, alpha=alpha) if len(paired_df) else None
        )
    except ValueError:
        sig_paired = None
    normality = normality_tests(records, metric=metric)

    report = TournamentAnalysisReport(
        records=records,
        payoff=payoff,
        has_self_play=has_self_play,
        tournament_ranking=tournament_ranking,
        sequential_elimination_ranking=ser,
        egta=egta,
        nash_averaging=nash_avg,
        iterated_elimination_survivors=survivors,
        replicator_dynamics=traj,
        significance_unpaired=sig_unpaired,
        significance_paired=sig_paired,
        normality=normality,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    if output_dir is not None:
        _save_report(report, Path(output_dir), original_scores, make_plots)
    return report


def _save_report(
    report: TournamentAnalysisReport,
    output_dir: Path,
    original_scores: pd.DataFrame | None,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.payoff.to_frame().to_csv(output_dir / "payoff_matrix.csv")
    report.tournament_ranking.to_csv(
        output_dir / "tournament_evaluation_ranking.csv", index=False
    )
    if report.sequential_elimination_ranking is not None:
        report.sequential_elimination_ranking.to_csv(
            output_dir / "sequential_elimination_ranking.csv", index=False
        )
    if report.egta is not None:
        pd.DataFrame(
            dict(
                candidates=[";".join(report.egta.candidates)],
                best=[report.egta.best],
                unique=[report.egta.unique],
            )
        ).to_csv(output_dir / "egta_evaluation.csv", index=False)
    if report.nash_averaging is not None:
        report.nash_averaging.to_csv(
            output_dir / "nash_averaging_ranking.csv", index=False
        )
    if report.iterated_elimination_survivors is not None:
        pd.DataFrame({"strategy": report.iterated_elimination_survivors}).to_csv(
            output_dir / "iterated_elimination_survivors.csv", index=False
        )
    if report.replicator_dynamics is not None:
        traj = report.replicator_dynamics
        traj_df = pd.DataFrame(traj.x, columns=traj.strategies_x)
        traj_df.insert(0, "time", traj.times)
        traj_df.to_csv(output_dir / "replicator_dynamics.csv", index=False)
    report.significance_unpaired.to_csv(
        output_dir / "significance_unpaired.csv", index=False
    )
    if report.significance_paired is not None:
        report.significance_paired.to_csv(
            output_dir / "significance_paired.csv", index=False
        )
    report.normality.to_csv(output_dir / "normality.csv", index=False)

    if make_plots:
        _save_plots(report, output_dir / "plots", original_scores)


def _save_plots(
    report: TournamentAnalysisReport,
    plots_dir: Path,
    original_scores: pd.DataFrame | None,
) -> None:
    import matplotlib.pyplot as plt

    from . import plotting
    from . import significance as sig_mod

    plots_dir.mkdir(parents=True, exist_ok=True)

    def _save(ax, name: str) -> None:
        ax.figure.savefig(plots_dir / name, bbox_inches="tight")
        plt.close(ax.figure)

    if original_scores is not None and len(original_scores):
        name_col = (
            "strategy"
            if "strategy" in original_scores.columns
            else original_scores.columns[0]
        )
        value_col = (
            "score"
            if "score" in original_scores.columns
            else original_scores.columns[-1]
        )
        _save(
            plotting.plot_scores_bar(
                original_scores,
                name_col=name_col,
                value_col=value_col,
                title="Tournament final scores",
            ),
            "tournament_scores.png",
        )

    _save(plotting.plot_payoff_heatmap(report.payoff), "payoff_heatmap.png")
    _save(
        plotting.plot_ranking(report.tournament_ranking),
        "tournament_evaluation_ranking.png",
    )
    if report.sequential_elimination_ranking is not None:
        _save(
            plotting.plot_ranking(report.sequential_elimination_ranking),
            "sequential_elimination_ranking.png",
        )
    if report.nash_averaging is not None:
        _save(
            plotting.plot_ranking(report.nash_averaging), "nash_averaging_ranking.png"
        )
    if report.has_self_play:
        _save(plotting.plot_deviation_graph(report.payoff), "deviation_graph.png")
    if report.replicator_dynamics is not None:
        _save(
            plotting.plot_replicator_dynamics(report.replicator_dynamics),
            "replicator_dynamics.png",
        )
    _save(
        sig_mod.plot_significance_heatmap(report.significance_unpaired),
        "significance_heatmap_unpaired.png",
    )
    _save(
        sig_mod.plot_significance_marks(report.significance_unpaired),
        "significance_marks_unpaired.png",
    )
    if report.significance_paired is not None:
        _save(
            sig_mod.plot_significance_heatmap(report.significance_paired),
            "significance_heatmap_paired.png",
        )
        _save(
            sig_mod.plot_significance_marks(report.significance_paired),
            "significance_marks_paired.png",
        )
