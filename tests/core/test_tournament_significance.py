"""Tests for negmas.tournaments.analysis.significance: pairwise hypothesis
tests, multiple-comparison corrections, and normality diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from negmas.tournaments.analysis import (
    apply_corrections,
    normality_tests,
    pairwise_tests,
    significance_matrix,
)


def _synthetic_records(seed=0) -> pd.DataFrame:
    """3 strategies x 6 scenarios x 4 reps; A and C share the same
    distribution (mean 0.5), B is shifted well above them (mean 0.9), so a
    real hypothesis test should find A-vs-C not significant and A-vs-B /
    B-vs-C significant."""
    rng = np.random.default_rng(seed)
    rows = []
    for strategy, mu in (("A", 0.5), ("B", 0.9), ("C", 0.5)):
        for scenario in [f"s{i}" for i in range(6)]:
            for _rep in range(4):
                rows.append(
                    dict(
                        strategy=strategy,
                        partner="X",
                        scenario=scenario,
                        utility=mu,
                        reserved_value=0.0,
                        advantage=rng.normal(mu, 0.05),
                    )
                )
    return pd.DataFrame(rows)


class TestPairwiseTests:
    @pytest.mark.parametrize("test", ["ttest", "ranksum", "ks"])
    def test_unpaired_detects_real_difference(self, test):
        records = _synthetic_records()
        df = pairwise_tests(records, test=test, paired=False)
        assert set(zip(df["strategy_a"], df["strategy_b"])) == {
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
        }
        row = df.set_index(["strategy_a", "strategy_b"])
        assert row.loc[("A", "B"), "p_value"] < 0.01
        assert row.loc[("B", "C"), "p_value"] < 0.01
        assert row.loc[("A", "C"), "p_value"] > 0.05

    @pytest.mark.parametrize("test", ["ttest", "ranksum"])
    def test_paired_detects_real_difference(self, test):
        records = _synthetic_records()
        df = pairwise_tests(records, test=test, paired=True)
        row = df.set_index(["strategy_a", "strategy_b"])
        # Wilcoxon signed-rank on 6 paired scenarios is an exact test whose
        # smallest achievable p-value is 1/32 = 0.03125 -- looser bound than
        # the t-test's, but both must still find the real A-vs-B difference.
        assert row.loc[("A", "B"), "p_value"] < 0.05
        assert row.loc[("A", "C"), "n_a"] == 6  # 6 shared scenarios

    def test_ks_rejects_paired(self):
        records = _synthetic_records()
        with pytest.raises(ValueError, match="does not support paired"):
            pairwise_tests(records, test="ks", paired=True)

    def test_unknown_test_name_rejected(self):
        records = _synthetic_records()
        with pytest.raises(ValueError, match="test must be one of"):
            pairwise_tests(records, test="bogus")

    def test_insufficient_samples_yields_nan_not_error(self):
        records = pd.DataFrame(
            [
                dict(
                    strategy="A",
                    partner="X",
                    scenario="s0",
                    utility=0.5,
                    reserved_value=0.0,
                    advantage=0.5,
                ),
                dict(
                    strategy="B",
                    partner="X",
                    scenario="s0",
                    utility=0.3,
                    reserved_value=0.0,
                    advantage=0.3,
                ),
            ]
        )
        df = pairwise_tests(records)
        assert len(df) == 1
        assert np.isnan(df["p_value"].iloc[0])
        assert np.isnan(df["statistic"].iloc[0])


class TestCorrections:
    def test_bonferroni_holm_bh_are_monotone_looser(self):
        # Bonferroni is always >= Holm >= raw p-value (Holm is uniformly more
        # powerful than Bonferroni by construction), and BH <= Bonferroni.
        records = _synthetic_records()
        df = apply_corrections(pairwise_tests(records), alpha=0.05)
        assert (df["p_holm"] <= df["p_bonferroni"] + 1e-12).all()
        assert (df["p_bh"] <= df["p_bonferroni"] + 1e-12).all()
        assert (df["p_value"] <= df["p_holm"] + 1e-12).all()

    def test_hand_computed_holm_and_bh(self):
        # 4 raw p-values with a known Holm/BH result (textbook example).
        df = pd.DataFrame(
            {
                "strategy_a": ["A", "A", "A", "B"],
                "strategy_b": ["B", "C", "D", "C"],
                "statistic": [0.0] * 4,
                "p_value": [0.01, 0.02, 0.03, 0.04],
                "n_a": [10] * 4,
                "n_b": [10] * 4,
            }
        )
        corrected = apply_corrections(df, alpha=0.05)
        # Holm: sorted p * (m - i), enforced non-decreasing, clipped to 1.
        # m=4: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06, 0.04*1=0.04 -> cummax -> [0.04,0.06,0.06,0.06]
        assert corrected["p_holm"].tolist() == pytest.approx([0.04, 0.06, 0.06, 0.06])
        # Bonferroni: p * m, clipped to 1.
        assert corrected["p_bonferroni"].tolist() == pytest.approx(
            [0.04, 0.08, 0.12, 0.16]
        )
        # BH: p_i * m / i (1-indexed by rank), enforced non-increasing from the top.
        # raw ranks (ascending): 0.01(1),0.02(2),0.03(3),0.04(4)
        # bh: 0.01*4/1=0.04, 0.02*4/2=0.04, 0.03*4/3=0.04, 0.04*4/4=0.04
        assert corrected["p_bh"].tolist() == pytest.approx([0.04, 0.04, 0.04, 0.04])

    def test_nan_pvalues_excluded_from_m_and_left_nan(self):
        df = pd.DataFrame(
            {
                "strategy_a": ["A", "A"],
                "strategy_b": ["B", "C"],
                "statistic": [1.0, float("nan")],
                "p_value": [0.03, float("nan")],
                "n_a": [10, 1],
                "n_b": [10, 1],
            }
        )
        corrected = apply_corrections(df, alpha=0.05)
        # m=1 (only the valid row counts) -> Bonferroni/Holm/BH == raw p-value.
        assert corrected["p_bonferroni"].iloc[0] == pytest.approx(0.03)
        assert np.isnan(corrected["p_bonferroni"].iloc[1])
        assert not corrected["significant_bonferroni"].iloc[1]


class TestSignificanceMatrix:
    def test_symmetric_and_diagonal_nan(self):
        records = _synthetic_records()
        df = apply_corrections(pairwise_tests(records), alpha=0.05)
        matrix = significance_matrix(df, value_col="p_value")
        assert matrix.loc["A", "B"] == matrix.loc["B", "A"]
        assert np.isnan(matrix.loc["A", "A"])


class TestNormalityTests:
    def test_returns_expected_columns_and_shapes(self):
        records = _synthetic_records()
        df = normality_tests(records)
        assert set(df["strategy"]) == {"A", "B", "C"}
        assert set(df.columns) == {
            "strategy",
            "n",
            "skewness",
            "kurtosis",
            "shapiro_w",
            "shapiro_p",
            "dagostino_p",
        }
        assert (df["n"] == 24).all()  # 6 scenarios x 4 reps

    def test_too_few_samples_yields_nan(self):
        records = pd.DataFrame(
            [
                dict(
                    strategy="A",
                    partner="X",
                    scenario="s0",
                    utility=0.5,
                    reserved_value=0.0,
                    advantage=0.5,
                ),
                dict(
                    strategy="A",
                    partner="X",
                    scenario="s1",
                    utility=0.6,
                    reserved_value=0.0,
                    advantage=0.6,
                ),
            ]
        )
        df = normality_tests(records)
        assert df["n"].iloc[0] == 2
        assert np.isnan(df["shapiro_w"].iloc[0])
        assert np.isnan(df["dagostino_p"].iloc[0])
