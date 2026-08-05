"""Tests for negmas.tournaments.analysis: loaders, payoff matrix
construction, EGTA pure-Nash evaluation, Sequential Elimination Ranking, and
replicator dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.tournaments.analysis import (
    analyze_tournament,
    build_payoff_table,
    dynamics,
    egta_evaluation,
    iterated_elimination,
    load_records,
    mixed_nash_equilibria,
    nash_averaging_ranking,
    pure_symmetric_nash_equilibria,
    records_from_path,
    records_from_simple_results,
    regret,
    sequential_elimination_ranking,
    tournament_evaluation_ranking,
)
from negmas.tournaments.neg import cartesian_tournament


def _records_from_matrix(strategies, matrix) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(strategies):
        for j, p in enumerate(strategies):
            rows.append(
                dict(
                    strategy=s,
                    partner=p,
                    scenario="s0",
                    utility=matrix[i, j],
                    reserved_value=0.0,
                    advantage=matrix[i, j],
                )
            )
    return pd.DataFrame(rows)


class TestPayoffTable:
    def test_build_payoff_table_basic(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.5], [0.7, 0.3]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records, metric="advantage", stat="mean")
        assert payoff.strategies == strategies
        assert np.allclose(payoff.matrix, matrix)

    def test_missing_self_play_raises_by_default(self):
        records = pd.DataFrame(
            [
                dict(
                    strategy="A",
                    partner="B",
                    scenario="s0",
                    utility=0.5,
                    reserved_value=0.0,
                    advantage=0.5,
                )
            ]
        )
        with pytest.raises(ValueError, match="self-play"):
            build_payoff_table(records)

    def test_missing_self_play_allowed_when_disabled(self):
        records = pd.DataFrame(
            [
                dict(
                    strategy="A",
                    partner="B",
                    scenario="s0",
                    utility=0.5,
                    reserved_value=0.0,
                    advantage=0.5,
                )
            ]
        )
        payoff = build_payoff_table(records, require_self_play=False)
        assert np.isnan(payoff.matrix[payoff.index("A"), payoff.index("A")])


class TestIteratedElimination:
    def test_strictly_dominated_strategy_removed(self):
        # A strictly dominates both B and C against every opponent (including
        # self), so only A should survive.
        strategies = ["A", "B", "C"]
        matrix = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        reduced = iterated_elimination(payoff)
        assert reduced.strategies == ["A"]

    def test_partially_dominated_strategy_removed(self):
        # C is strictly dominated by A, but B is not dominated by anyone.
        strategies = ["A", "B", "C"]
        matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.1, -1.0, 0.1]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        reduced = iterated_elimination(payoff)
        assert set(reduced.strategies) == {"A", "B"}


class TestEGTAEvaluation:
    def test_finds_pure_symmetric_nash(self):
        # Column A's max (0.5) is at row A -> A is a best response to itself.
        # Column B's max (0.8) is at row B -> B is a best response to itself.
        # Column C's max (0.9) is at row A, not C -> C is not a pure sym. NE.
        # Among {A, B}, B has the higher diagonal payoff (0.8 > 0.5) -> best.
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        result = egta_evaluation(payoff)
        assert set(result.candidates) == {"A", "B"}
        assert result.best == "B"
        assert result.unique

    def test_no_pure_symmetric_nash(self):
        # A rock-paper-scissors-like matrix has no pure symmetric NE: the
        # column-best-response never lands back on the diagonal.
        strategies = ["rock", "paper", "scissors"]
        matrix = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        result = egta_evaluation(payoff)
        assert result.candidates == []
        assert result.best is None

    def test_tied_pure_symmetric_nash(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        result = egta_evaluation(payoff)
        assert set(result.candidates) == {"A", "B"}
        assert not result.unique
        assert result.best == "A"  # deterministic alphabetical tie-break

    def test_pure_symmetric_nash_helper_matches_evaluation(self):
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        assert (
            pure_symmetric_nash_equilibria(payoff) == egta_evaluation(payoff).candidates
        )


class TestRegretAndNashAveraging:
    def test_regret_of_pure_nash_profile_is_zero(self):
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        # B is a pure symmetric NE (see TestEGTAEvaluation): no deviation
        # from a population entirely playing B can gain anything.
        assert regret(payoff, {"B": 1.0}) == pytest.approx(0.0, abs=1e-9)

    def test_regret_of_non_equilibrium_profile_matches_known_deviation_gain(self):
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        # Against an all-C population, the best deviation is to A (col C max
        # is M[A, C] = 0.9), current payoff of C against itself is M[C, C]=0.6.
        assert regret(payoff, {"C": 1.0}) == pytest.approx(0.9 - 0.6)

    def test_regret_rejects_profile_not_summing_to_one(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        with pytest.raises(ValueError, match="sum to 1"):
            regret(payoff, {"A": 0.5})

    def test_nash_averaging_ranking_defaults_to_egta_best(self):
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        ranking = nash_averaging_ranking(payoff)
        assert list(ranking["rank"]) == [1, 2, 3]
        # B is the reference equilibrium -> its own regret against itself is 0.
        assert ranking.loc[
            ranking["strategy"] == "B", "regret"
        ].item() == pytest.approx(0.0, abs=1e-9)

    def test_nash_averaging_ranking_raises_without_pure_nash_or_explicit_profile(self):
        strategies = ["rock", "paper", "scissors"]
        matrix = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        with pytest.raises(ValueError, match="No pure symmetric Nash"):
            nash_averaging_ranking(payoff)


class TestMixedNashEquilibria:
    def test_requires_nashpy(self):
        pytest.importorskip("nashpy")
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        equilibria = mixed_nash_equilibria(payoff)
        assert len(equilibria) > 0
        for eq in equilibria:
            assert set(eq) <= {"A", "B"}
            assert sum(eq.values()) == pytest.approx(1.0, abs=1e-6)

    def test_raises_informative_error_without_nashpy(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "nashpy":
                raise ImportError("simulated missing nashpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        with pytest.raises(ImportError, match="negmas\\[egta\\]"):
            mixed_nash_equilibria(payoff)


class TestSequentialEliminationRanking:
    def test_hand_computed_elimination_order(self):
        # Constructed so that elimination order differs from a plain
        # row-average (tournament evaluation) ranking, since the average is
        # recomputed over the shrinking remaining set each round.
        strategies = ["A", "B", "C", "D"]
        matrix = np.array(
            [
                [3.0, 3.0, 3.0, 0.0],  # A: great vs A/B/C, terrible vs D
                [3.0, 3.0, 3.0, 0.0],  # B: identical to A
                [3.0, 3.0, 3.0, 0.0],  # C: identical to A/B
                [10.0, 10.0, 10.0, 1.0],  # D: crushes everyone, weak vs self
            ]
        )
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)

        # Round 1 (all 4 remaining): U^4(A)=U^4(B)=U^4(C)=2.25, U^4(D)=7.75.
        # A/B/C tie for lowest -> alphabetically first (A) eliminated first.
        # Round 2 (B,C,D remaining): U^3(B)=U^3(C)=2.0, U^3(D)=7.0 -> B eliminated.
        # Round 3 (C,D remaining): U^2(C)=1.5, U^2(D)=5.5 -> C eliminated.
        # Round 4 (D remaining): D is rank 1.
        ranking = sequential_elimination_ranking(payoff)
        by_strategy = dict(zip(ranking["strategy"], ranking["rank"]))
        assert by_strategy == {"A": 4, "B": 3, "C": 2, "D": 1}

    def test_filler_agent_does_not_change_relative_ranking(self):
        """Injecting one uniformly-weak filler strategy must not change the
        relative order of the pre-existing strategies (de Jonge, chapter
        6.3): the filler is eliminated in round 1 and thereafter excluded
        from every remaining average."""
        rng = np.random.default_rng(0)
        strategies = ["A", "B", "C", "D"]
        base = rng.uniform(0.2, 1.0, size=(4, 4))
        records = _records_from_matrix(strategies, base)
        payoff = build_payoff_table(records)
        baseline = sequential_elimination_ranking(payoff)
        baseline_order = list(baseline.sort_values("rank")["strategy"])

        strategies_with_filler = strategies + ["filler"]
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = base
        augmented[4, :] = 0.0  # filler is worst against everyone, including itself
        augmented[:4, 4] = 0.0  # everyone crushes the filler
        records2 = _records_from_matrix(strategies_with_filler, augmented)
        payoff2 = build_payoff_table(records2)
        with_filler = sequential_elimination_ranking(payoff2)
        with_filler_order = list(
            with_filler[with_filler["strategy"] != "filler"].sort_values("rank")[
                "strategy"
            ]
        )

        assert with_filler_order == baseline_order
        assert with_filler.loc[with_filler["strategy"] == "filler", "rank"].item() == 5

    def test_full_ranking_covers_all_strategies_exactly_once(self):
        rng = np.random.default_rng(1)
        strategies = [f"s{i}" for i in range(6)]
        matrix = rng.uniform(0, 1, size=(6, 6))
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        ranking = sequential_elimination_ranking(payoff)
        assert sorted(ranking["strategy"]) == sorted(strategies)
        assert sorted(ranking["rank"]) == list(range(1, 7))


class TestTournamentEvaluationRanking:
    def test_ranks_by_row_mean(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 1.0], [0.0, 0.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        ranking = tournament_evaluation_ranking(payoff)
        assert list(ranking["strategy"]) == ["A", "B"]
        assert list(ranking["rank"]) == [1, 2]


class TestReplicatorDynamics:
    def test_symmetric_trajectory_normalizes_and_has_expected_shape(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.5], [0.7, 0.3]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        traj = dynamics.symmetric_replicator_dynamics(payoff, t_max=10.0, n_steps=20)
        assert traj.x.shape == (20, 2)
        assert np.allclose(traj.x.sum(axis=1), 1.0, atol=1e-6)
        assert traj.strategies_x == strategies

    def test_dominant_strategy_takes_over_population(self):
        # A strictly dominates B against every opponent -> population share
        # of A must (weakly) increase over time from an interior start.
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 1.0], [0.0, 0.0]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        traj = dynamics.symmetric_replicator_dynamics(
            payoff, x0={"A": 0.5, "B": 0.5}, t_max=50.0, n_steps=50
        )
        assert traj.x[-1, 0] > traj.x[0, 0]
        final = dynamics.final_mixed_strategy(traj, eps=0.5)
        assert final == {"A": pytest.approx(1.0, abs=1e-2)}

    def test_asymmetric_dynamics_defaults_to_transpose(self):
        strategies = ["A", "B"]
        matrix = np.array([[1.0, 0.5], [0.7, 0.3]])
        records = _records_from_matrix(strategies, matrix)
        payoff = build_payoff_table(records)
        traj = dynamics.asymmetric_replicator_dynamics(payoff, t_max=5.0, n_steps=10)
        assert traj.y is not None
        assert traj.x.shape == (10, 2)
        assert traj.y.shape == (10, 2)
        assert np.allclose(traj.y.sum(axis=1), 1.0, atol=1e-6)


def _small_scenarios():
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    return [
        Scenario(
            outcome_space=make_os(issues, name="S0"),
            ufuns=(
                U.random(issues=issues, reserved_value=0.0, normalized=False),
                U.random(issues=issues, reserved_value=0.0, normalized=False),
            ),
            mechanism_type=SAOMechanism,
            mechanism_params=dict(),
        )
    ]


class TestLoaders:
    def test_in_memory_and_from_disk_loaders_agree(self, tmp_path):
        results = cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            path=tmp_path / "tournament",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
        in_memory = records_from_simple_results(results)
        from_disk = records_from_path(tmp_path / "tournament")

        def _key(df):
            return set(zip(df["strategy"], df["partner"], df["scenario"]))

        assert _key(in_memory) == _key(from_disk)
        assert list(in_memory.columns) == list(from_disk.columns)

    def test_load_records_dispatches_on_type(self):
        results = cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            path=None,
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
        via_object = load_records(results)
        assert set(via_object.columns) >= {
            "strategy",
            "partner",
            "scenario",
            "utility",
            "reserved_value",
            "advantage",
        }
        payoff = build_payoff_table(via_object)
        assert set(payoff.strategies) == {
            "AspirationNegotiator",
            "NaiveTitForTatNegotiator",
        }


def _situated_results(params, rows):
    from negmas.tournaments.tournaments import TournamentResults

    scores = pd.DataFrame(rows, columns=["agent_type", "world", "score"])
    return TournamentResults(
        scores=scores,
        total_scores=pd.DataFrame(),
        winners=[],
        winners_scores=np.array([]),
        params=params,
    )


class TestSituatedLoader:
    def test_bilateral_worlds_are_decomposed(self):
        rows = [("A", "w0", 0.6), ("B", "w0", 0.4), ("A", "w1", 0.7), ("B", "w1", 0.3)]
        results = _situated_results({"n_competitors_per_world": 2}, rows)
        from negmas.tournaments.analysis import records_from_situated

        out = records_from_situated(results)
        assert set(zip(out["strategy"], out["partner"], out["scenario"])) == {
            ("A", "B", "w0"),
            ("B", "A", "w0"),
            ("A", "B", "w1"),
            ("B", "A", "w1"),
        }
        assert out.loc[
            (out["strategy"] == "A") & (out["scenario"] == "w0"), "utility"
        ].item() == pytest.approx(0.6)

    def test_non_competitors_excluded_from_bilateral_check(self):
        # 3 distinct agent_types per world, but only 2 are competitors
        # (n_competitors_per_world=2); the third is a non_competitor and must
        # not make the world look non-bilateral.
        rows = [("A", "w0", 0.6), ("B", "w0", 0.4), ("Bystander", "w0", 0.1)]
        results = _situated_results(
            {"n_competitors_per_world": 2, "non_competitors": ("Bystander",)}, rows
        )
        from negmas.tournaments.analysis import records_from_situated

        out = records_from_situated(results)
        assert set(out["strategy"]) == {"A", "B"}
        assert "Bystander" not in set(out["strategy"]) | set(out["partner"])

    def test_non_bilateral_tournament_returns_empty_with_warning(self):
        rows = [("A", "w0", 0.6), ("B", "w0", 0.3), ("C", "w0", 0.5)]
        results = _situated_results({"n_competitors_per_world": 3}, rows)
        from negmas.tournaments.analysis import records_from_situated

        with pytest.warns(UserWarning, match="n_competitors_per_world"):
            out = records_from_situated(results)
        assert len(out) == 0

    def test_world_with_wrong_distinct_competitor_count_is_skipped(self):
        # No explicit n_competitors_per_world in params: falls back to
        # per-world distinct-competitor-type counting; a 3-type world (after
        # excluding non-competitors, none here) is skipped with a warning.
        rows = [
            ("A", "w0", 0.6),
            ("B", "w0", 0.4),
            ("A", "w1", 0.5),
            ("B", "w1", 0.3),
            ("C", "w1", 0.2),
        ]
        results = _situated_results({}, rows)
        from negmas.tournaments.analysis import records_from_situated

        with pytest.warns(UserWarning, match="distinct"):
            out = records_from_situated(results)
        assert set(out["scenario"]) == {"w0"}


class TestAnalyzeTournament:
    def test_post_hoc_on_in_memory_results(self, tmp_path):
        results = cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=2,
            njobs=2,
            executor="thread",
            path=None,
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
        report = analyze_tournament(
            results, output_dir=tmp_path / "analysis", make_plots=True
        )
        assert report.has_self_play
        assert set(report.tournament_ranking["strategy"]) == {
            "AspirationNegotiator",
            "NaiveTitForTatNegotiator",
        }
        assert report.sequential_elimination_ranking is not None
        assert report.egta is not None
        assert report.replicator_dynamics is not None

        expected_csvs = {
            "payoff_matrix.csv",
            "tournament_evaluation_ranking.csv",
            "sequential_elimination_ranking.csv",
            "egta_evaluation.csv",
            "iterated_elimination_survivors.csv",
            "replicator_dynamics.csv",
            "significance_unpaired.csv",
            "normality.csv",
        }
        on_disk = {p.name for p in (tmp_path / "analysis").glob("*.csv")}
        assert expected_csvs <= on_disk

        plot_dir = tmp_path / "analysis" / "plots"
        assert plot_dir.exists()
        assert len(list(plot_dir.glob("*.png"))) > 0

    def test_post_hoc_on_path(self, tmp_path):
        cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            path=tmp_path / "tournament",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
        report = analyze_tournament(tmp_path / "tournament")
        assert report.has_self_play
        assert report.output_dir is None  # no output_dir given -> nothing saved

    def test_run_analysis_flag_on_cartesian_tournament(self, tmp_path):
        cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            path=tmp_path / "tournament",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
            run_analysis=True,
            analysis_plots=True,
        )
        analysis_dir = tmp_path / "tournament" / "analysis"
        assert analysis_dir.exists()
        assert (analysis_dir / "sequential_elimination_ranking.csv").exists()
        assert (analysis_dir / "plots").exists()
        assert len(list((analysis_dir / "plots").glob("*.png"))) > 0

    def test_run_analysis_false_by_default_no_analysis_dir(self, tmp_path):
        cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_small_scenarios(),
            n_steps=20,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            path=tmp_path / "tournament",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
        assert not (tmp_path / "tournament" / "analysis").exists()

    def test_no_self_play_degrades_gracefully(self):
        strategies = ["A", "B", "C"]
        matrix = np.array([[0.5, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.1, 0.6]])
        records = _records_from_matrix(strategies, matrix)
        # Drop the diagonal (self-play) rows.
        no_self_play = records[records["strategy"] != records["partner"]]
        with pytest.warns(UserWarning, match="self-play"):
            report = analyze_tournament(no_self_play)
        assert not report.has_self_play
        assert report.sequential_elimination_ranking is None
        assert report.egta is None
        assert report.replicator_dynamics is None
        assert len(report.tournament_ranking) == 3
