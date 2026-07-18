"""Issue-weight opponent models (survey §5.3.1).

Linear-additive opponent utility models that estimate the opponent's *issue
weights* from the way its offers change over time, following the "issue
preference order" family of Baarslag, Hendrikx, Hindriks & Jonker, *Learning
about the opponent in automated bilateral negotiation: a comprehensive survey of
opponent modeling techniques*, JAAMAS 30:849-898 (2016), §5.3.1.

All three models here assume the opponent uses a `LinearAdditiveUtilityFunction`
(a weighted sum of per-issue value utilities) and a time-dependent, concession
based bidding strategy — the assumption the survey attaches to this whole family.
They differ only in how they turn the *sequence of offers* into issue weights:

- `ConcessionRatioUFunModel` — Niemann & Lang [143]: an issue conceded on more
  often is less important, so ``weight = 1 - concession_ratio``.
- `ValueDifferenceUFunModel` — Carbonneau & Vahidov [37]: uses the *magnitude* of
  the change in each issue's value between sequential offers (normalized by the
  issue range), not merely whether it changed.
- `KDEWeightUFunModel` — Coehoorn & Jennings [50] (the most-established
  issue-weight method): estimates, via kernel density estimation, how much
  probability mass each issue's sequential-offer distances put near zero (an
  issue that barely moves is important).

In every case per-issue *value* utilities are estimated from offer frequency
(values offered more often are assumed more preferred), exactly as in
`FrequencyLinearUFunModel`, and the estimated utility of an outcome is the
weight-normalized sum of the per-issue value scores.

*AI Generated (issue-weight opponent models from Baarslag et al. 2016 §5.3.1).*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from .ufun import UFunModel, _bucket_value, _discretize_issues

if TYPE_CHECKING:
    from negmas import PreferencesChange, Value
    from negmas.outcomes import Outcome

__all__ = ["ConcessionRatioUFunModel", "ValueDifferenceUFunModel", "KDEWeightUFunModel"]


def _issue_span(discrete_issue) -> tuple[float, bool]:
    """Return ``(span, is_numeric)`` for normalizing value differences.

    For a numeric issue ``span`` is ``max - min`` over its (discretized) values;
    for a categorical issue it is ``1.0`` and ``is_numeric`` is ``False`` (so a
    change is scored as a 0/1 indicator). ``span`` is never zero.
    """
    try:
        vals = [float(v) for v in discrete_issue.all]
        span = max(vals) - min(vals)
        return (span if span > 0 else 1.0, True)
    except (TypeError, ValueError):
        return (1.0, False)


@define
class _SequentialWeightUFunModel(UFunModel):
    """Shared machinery for the §5.3.1 sequential-offer issue-weight models.

    Subclasses implement :meth:`_update_weights`, which is called with the
    previous and current *bucketed* offers whenever a new offer is observed, and
    should refresh ``self._weights`` (a ``{issue_name: weight}`` map). The base
    class tracks per-issue value-frequency counts (for value utilities) and the
    previous offer, and computes the linear-additive utility in :meth:`eval`.

    *AI Generated.*
    """

    above_reserve: bool = True
    levels: int = 10
    _counts: dict = field(init=False, factory=dict)
    _weights: dict = field(init=False, factory=dict)
    _total: int = field(init=False, default=0)
    _issues: list | None = field(init=False, default=None)
    _discrete_issues: list | None = field(init=False, default=None)
    _prev: tuple | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Set up the (discretized) issues and register the model."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        os_ = self.negotiator.ufun.outcome_space
        issues = getattr(os_, "issues", None) if os_ is not None else None
        self._issues = list(issues) if issues is not None else []
        self._discrete_issues = _discretize_issues(self._issues, self.levels)
        self._update_private_info()

    def before_responding(
        self, state, offer: Outcome | None, source: str | None = None
    ):
        """Learn from the offer the negotiator is about to respond to."""
        if offer is not None:
            self._observe(offer)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        """Learn from a partner proposal (only with ``enable_callbacks``)."""
        if offer is not None:
            self._observe(offer)

    def _bucketed(self, offer: Outcome) -> tuple:
        """Snap ``offer`` to its per-issue discretized values."""
        assert self._issues is not None and self._discrete_issues is not None
        return tuple(
            _bucket_value(issue, dissue, value)
            for issue, dissue, value in zip(self._issues, self._discrete_issues, offer)
        )

    def _observe(self, offer: Outcome) -> None:
        """Record an opponent offer: update value counts, then issue weights."""
        if not self._issues or not self._discrete_issues:
            return
        cur = self._bucketed(offer)
        for issue, bucket in zip(self._issues, cur):
            counts = self._counts.setdefault(issue.name, {})
            counts[bucket] = counts.get(bucket, 0) + 1
        self._total += 1
        if self._prev is not None:
            self._update_weights(self._prev, cur)
        self._prev = cur

    def _update_weights(self, prev: tuple, cur: tuple) -> None:  # pragma: no cover
        """Refresh ``self._weights`` from two sequential (bucketed) offers."""
        raise NotImplementedError()

    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's normalized utility of ``offer`` in ``[0, 1]``."""
        if (
            offer is None
            or not self._issues
            or not self._discrete_issues
            or self._total == 0
        ):
            return 0.5
        scores, weights = [], []
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            name = issue.name
            bucket = _bucket_value(issue, dissue, value)
            counts = self._counts.get(name, {})
            mx = max(counts.values()) if counts else 0
            scores.append((counts.get(bucket, 0) / mx) if mx > 0 else 0.0)
            weights.append(self._weights.get(name, 1.0))
        wsum = sum(weights)
        if wsum <= 0:
            return sum(scores) / len(scores) if scores else 0.5
        return sum(s * w for s, w in zip(scores, weights)) / wsum

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Eval normalized (the model already returns values in ``[0, 1]``)."""
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class ConcessionRatioUFunModel(_SequentialWeightUFunModel):
    """Issue weights from concession ratios — Niemann & Lang [143] (survey §5.3.1).

    For each issue a *concession ratio* ``c_i`` is the fraction of consecutive
    offer pairs in which the opponent changed that issue's value. The more an
    issue is conceded on, the less important it is assumed to be, so the issue
    weight is ``w_i = 1 - c_i`` (then normalized across issues). Per-issue value
    utilities are estimated from offer frequency.

    Remarks:
        - The survey's method updates a Bayesian posterior over weight
          hypotheses; here ``1 - c_i`` is used directly as a deterministic
          maximum-likelihood rendition (no explicit posterior), which keeps the
          model domain-agnostic and online.
        - Returns utilities already normalized to ``[0, 1]``; a neutral ``0.5``
          before any offer is observed.

    *AI Generated (Niemann & Lang concession-ratio issue weights).*
    """

    _change_counts: dict = field(init=False, factory=dict)
    _steps: int = field(init=False, default=0)

    def _update_weights(self, prev: tuple, cur: tuple) -> None:
        assert self._issues is not None
        self._steps += 1
        for issue, p, c in zip(self._issues, prev, cur):
            if p != c:
                self._change_counts[issue.name] = (
                    self._change_counts.get(issue.name, 0) + 1
                )
        if self._steps <= 0:
            return
        self._weights = {
            issue.name: 1.0 - (self._change_counts.get(issue.name, 0) / self._steps)
            for issue in self._issues
        }


@define
class ValueDifferenceUFunModel(_SequentialWeightUFunModel):
    """Issue weights from value-difference magnitudes — Carbonneau & Vahidov [37].

    Unlike `ConcessionRatioUFunModel`, which only counts *whether* an issue's
    value changed, this model uses the *magnitude* of the change between two
    sequential offers, normalized by the issue's range (survey §5.3.1). For a
    numeric issue the per-step difference is ``|v_t - v_{t-1}| / range_i``; for a
    categorical issue it is a ``0/1`` indicator. The running mean normalized
    difference ``d_i`` gives the issue weight ``w_i = 1 - d_i`` (then normalized).
    Per-issue value utilities are estimated from offer frequency.

    Remarks:
        - Larger (normalized) moves on an issue ⇒ more concession ⇒ lower weight.
        - Returns utilities already normalized to ``[0, 1]``; a neutral ``0.5``
          before any offer is observed.

    *AI Generated (Carbonneau & Vahidov value-difference issue weights).*
    """

    _diff_sum: dict = field(init=False, factory=dict)
    _steps: int = field(init=False, default=0)

    def _update_weights(self, prev: tuple, cur: tuple) -> None:
        assert self._issues is not None and self._discrete_issues is not None
        self._steps += 1
        for issue, dissue, p, c in zip(self._issues, self._discrete_issues, prev, cur):
            span, numeric = _issue_span(dissue)
            if numeric:
                try:
                    d = abs(float(c) - float(p)) / span
                except (TypeError, ValueError):
                    d = 0.0 if p == c else 1.0
            else:
                d = 0.0 if p == c else 1.0
            self._diff_sum[issue.name] = self._diff_sum.get(issue.name, 0.0) + min(
                1.0, d
            )
        if self._steps <= 0:
            return
        self._weights = {
            issue.name: 1.0 - (self._diff_sum.get(issue.name, 0.0) / self._steps)
            for issue in self._issues
        }


@define
class KDEWeightUFunModel(_SequentialWeightUFunModel):
    """Issue weights via kernel density estimation — Coehoorn & Jennings [50].

    The most-established issue-weight method in the survey (§5.3.1). For each
    issue it collects the sequence of normalized distances between sequential
    offers and fits a Gaussian kernel density estimate (`scipy.stats.gaussian_kde`)
    to that distance distribution. An issue whose distances concentrate near zero
    (the opponent barely moves it) is important, so the issue weight is the KDE
    probability mass in ``[0, threshold]`` — i.e. ``P(distance <= threshold)`` —
    then normalized across issues. Per-issue value utilities are estimated from
    offer frequency.

    Args:
        threshold: The distance below which a move counts as "held" when
            integrating the KDE (a fraction of the normalized ``[0, 1]`` range).

    Remarks:
        - The survey's full method maps distance to weight using a database of
          previous negotiations; this is a domain-agnostic *online* rendition
          that estimates the distance distribution from the running negotiation.
        - Falls back to the mean-distance estimate ``1 - mean_distance`` per issue
          until there are enough distinct distance samples to fit a KDE.
        - Returns utilities already normalized to ``[0, 1]``.

    *AI Generated (Coehoorn & Jennings KDE issue weights).*
    """

    threshold: float = 0.1
    _dists: dict = field(init=False, factory=dict)

    def _update_weights(self, prev: tuple, cur: tuple) -> None:
        assert self._issues is not None and self._discrete_issues is not None
        for issue, dissue, p, c in zip(self._issues, self._discrete_issues, prev, cur):
            span, numeric = _issue_span(dissue)
            if numeric:
                try:
                    d = min(1.0, abs(float(c) - float(p)) / span)
                except (TypeError, ValueError):
                    d = 0.0 if p == c else 1.0
            else:
                d = 0.0 if p == c else 1.0
            self._dists.setdefault(issue.name, []).append(d)
        self._recompute_kde_weights()

    def _recompute_kde_weights(self) -> None:
        assert self._issues is not None
        import numpy as np

        weights: dict = {}
        for issue in self._issues:
            dists = self._dists.get(issue.name, [])
            if not dists:
                weights[issue.name] = 1.0
                continue
            arr = np.asarray(dists, dtype=float)
            # KDE needs >=2 samples with non-zero variance; otherwise fall back.
            if arr.size >= 2 and float(arr.var()) > 1e-12:
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(arr)
                    mass = float(kde.integrate_box_1d(0.0, self.threshold))
                    weights[issue.name] = min(1.0, max(0.0, mass))
                    continue
                except Exception:  # pragma: no cover - defensive
                    pass
            weights[issue.name] = 1.0 - float(arr.mean())
        self._weights = weights
