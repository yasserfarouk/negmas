"""Utility function implementations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from attrs import define, field

from negmas.preferences import RankOnlyUtilityFunction
from negmas.preferences.base_ufun import BaseUtilityFunction

from ..base import GBComponent

if TYPE_CHECKING:
    from negmas import PreferencesChange, Value
    from negmas.outcomes import Outcome

__all__ = [
    "UFunModel",
    "FrequencyUFunModel",
    "FrequencyLinearUFunModel",
    "PeekingOpponentModel",
    "ZeroSumModel",
]

SENTINAL = object()


def _discretize_issues(issues, levels: int = 10):
    """Return a discrete counterpart of each issue via ``issue.to_discrete``.

    Already-discrete issues are returned unchanged; continuous issues are
    discretized to at most ``levels`` grid values (so frequency counts can be
    kept over a finite value set instead of ignoring continuous issues).
    """
    out = []
    for issue in issues:
        try:
            out.append(issue.to_discrete(n=levels))
        except Exception:  # pragma: no cover - defensive
            out.append(issue)
    return out


def _bucket_value(original_issue, discrete_issue, value):
    """Map an offered ``value`` to the discretized value it falls into.

    For already-discrete issues the value is returned unchanged. For continuous
    issues it is snapped to the nearest grid value of the discretized issue.
    """
    if getattr(original_issue, "is_discrete", lambda: False)():
        return value
    best = None
    best_d = None
    for d in discrete_issue.all:
        try:
            dd = abs(float(d) - float(value))
        except (TypeError, ValueError):
            if d == value:
                return d
            continue
        if best_d is None or dd < best_d:
            best_d = dd
            best = d
    return best if best is not None else value


class UFunModel(GBComponent, BaseUtilityFunction):
    """
    A `SAOComponent` that can model the opponent's utility function.

    Classes implementing this ufun-model, must implement the abstract `eval()`
    method to return the utility value of an outcome. They can use any callbacks
    available to `SAOComponent` to update the model.

    A `UFunModel` is a *full stand-in for a `BaseUtilityFunction`*: anywhere a
    ufun is accepted (e.g. `pareto_frontier`, the bargaining-solution
    calculators in `negmas.preferences.ops`, or a `ParetoSampler`) a `UFunModel`
    can be passed without errors. Concrete subclasses are typically declared
    with ``@define`` and therefore skip `BaseUtilityFunction.__init__`, so the
    state it would have set up (``_invalid_value``, ``_constraints``,
    ``_reserved_value`` and the caches) is initialised here in
    ``__attrs_post_init__`` instead.

    *AI supported (made into a full ufun stand-in for use as an opponent model).*
    """

    def __attrs_post_init__(self):
        """Initialize the `BaseUtilityFunction`-required state.

        ``@define`` subclasses generate their own ``__init__`` and so skip
        `BaseUtilityFunction.__init__` (and `Preferences.__init__`). We call it
        here so the model carries every piece of state a ufun needs —
        ``reserved_value``, ``_invalid_value``, ``_constraints``, ``_stability``,
        the caches, ``outcome_space`` plumbing, etc. — and is a *full stand-in*
        for a `BaseUtilityFunction`: anywhere a ufun is accepted
        (`pareto_frontier`, the bargaining-solution calculators, `ParetoSampler`,
        `minmax`, `invert`, …) a `UFunModel` works without errors.

        The default reserved value is ``0.0`` (the normalised disagreement point)
        so the model's reserve is finite and usable by the bargaining-solution
        calculators. This does not touch the ``@define``-declared fields
        (``above_reserve``, ``levels``, ``ufun``, ``_counts``, …), which are set
        by the attrs-generated ``__init__`` before this hook runs.
        """
        BaseUtilityFunction.__init__(self, reserved_value=0.0)

    @property
    def outcome_space(self):  # type: ignore[override]
        """The outcome space the model is defined over.

        A `UFunModel` models the opponent *over the same outcomes* the agent
        cares about, so its outcome space is the negotiator's ufun outcome
        space. Exposing it (rather than leaving it unset, as the ``@define``
        subclasses do by skipping `BaseUtilityFunction.__init__`) lets the model
        be used wherever a ufun is expected — e.g. `minmax`, `pareto_frontier`,
        and the bargaining-solution calculators all read ``outcome_space``.
        """
        negotiator = getattr(self, "negotiator", None)
        if negotiator is not None and getattr(negotiator, "ufun", None) is not None:
            return negotiator.ufun.outcome_space
        return getattr(self, "_outcome_space", None)

    @outcome_space.setter
    def outcome_space(self, value) -> None:
        self._outcome_space = value

    def _update_private_info(self, partner_id: str | None = None) -> None:
        """Update the negotiator's private_info with this model.

        For bilateral negotiations, sets private_info["opponent_ufun"].
        For multilateral negotiations, sets private_info["opponent_ufuns"][partner_id].

        Args:
            partner_id: The partner's ID for multilateral negotiations.
                       If None, assumes bilateral and uses "opponent_ufun".
        """
        if not self.negotiator:
            return

        # Ensure private_info exists
        if not hasattr(self.negotiator, "private_info"):
            return

        private_info = self.negotiator.private_info
        if private_info is None:
            return

        # Check if this is a multilateral negotiation
        nmi = self.negotiator.nmi
        is_multilateral = nmi is not None and nmi.n_negotiators > 2

        if is_multilateral and partner_id is not None:
            # Multilateral: store in opponent_ufuns dict
            if "opponent_ufuns" not in private_info:
                private_info["opponent_ufuns"] = {}
            private_info["opponent_ufuns"][partner_id] = self
        else:
            # Bilateral: store directly
            private_info["opponent_ufun"] = self


@define
class ZeroSumModel(UFunModel):
    """
    Assumes a zero-sum negotiation (i.e. $u_o$ = $-u_s$ )

    Remarks:

        - Because some negotiators do not work well with negative ufun values, we return (max - u(w)) instead of (- u(w))
    """

    above_reserve: bool = True
    rank_only: bool = False
    _effective_ufun: BaseUtilityFunction = field(init=False, default=SENTINAL)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """On preferences changed.

        Args:
            changes: Changes.
        """
        if not self.negotiator or not self.negotiator.ufun:
            raise ValueError("Negotiator or ufun are not known")
        self._effective_ufun = (
            self.negotiator.ufun
            if not self.rank_only
            else RankOnlyUtilityFunction(self.negotiator.ufun)
        )
        # Update private_info so negotiators can access this model
        self._update_private_info()

    def eval(self, offer: Outcome) -> Value:
        """Eval.

        Args:
            offer: Offer being considered.

        Returns:
            Value: The result.
        """
        uo = self._effective_ufun.eval_normalized(offer, self.above_reserve) * -1 + 1.0
        mn, mx = self._effective_ufun.minmax(above_reserve=False)
        return uo * (mx - mn) + mn

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Eval normalized.

        Args:
            offer: Offer being considered.
            above_reserve: Above reserve.
            expected_limits: Expected limits.

        Returns:
            Value: The result.
        """
        if offer is None:
            return 0.0
        return (
            self._effective_ufun.eval_normalized(offer, above_reserve, expected_limits)
            * -1
            + 1.0
        )


@define
class FrequencyUFunModel(UFunModel):
    """
    A frequency-based opponent model that makes **no assumption** about the form
    of the opponent's utility function (use this when the opponent's ufun is
    *not* known to be linear-additive).

    It counts how often the opponent has offered each complete outcome and
    estimates the opponent's utility of an outcome as its offer frequency
    normalized by the maximum observed frequency. Outcomes never offered score
    ``0``. Before any offer is observed, every outcome scores a neutral ``0.5``.

    The model is updated from every offer the negotiator responds to
    (``before_responding``) and, when the mechanism enables callbacks, from
    ``on_partner_proposal``.

    Remarks:
        - Returns utilities in ``[0, 1]`` (already normalized).
        - Sparse: only outcomes actually offered get a positive score, so this
          model is best suited to small/discrete outcome spaces. For larger
          spaces with an additive opponent ufun, prefer
          `FrequencyLinearUFunModel`.

    *AI Generated (frequency-based opponent model).*
    """

    above_reserve: bool = True
    levels: int = 10
    _counts: dict = field(init=False, factory=dict)
    _total: int = field(init=False, default=0)
    _issues: list | None = field(init=False, default=None)
    _discrete_issues: list | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Register the model and set up discretized issues.

        Args:
            changes: Changes.
        """
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
        """Learn from the offer the negotiator is about to respond to.

        Args:
            state: Current state.
            offer: The partner's offer being responded to.
            source: Source identifier.
        """
        if offer is not None:
            self._observe(offer)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        """Learn from a partner proposal (only called with ``enable_callbacks``).

        Args:
            state: State when the offer was proposed.
            partner_id: The ID of the agent who proposed.
            offer: The proposal.
        """
        if offer is not None:
            self._observe(offer)

    def _bucket_key(self, offer: Outcome) -> tuple:
        """Snap an offer to its discretized (hashable) key.

        Continuous issue values are snapped to the nearest discretization grid
        value, so repeated offers in the same region of a continuous issue count
        toward the same bucket instead of being treated as unique.
        """
        if not self._issues or not self._discrete_issues:
            try:
                hash(offer)
                return tuple(offer)  # type: ignore[return-value]
            except TypeError:
                return tuple(offer)  # type: ignore[return-value]
        return tuple(
            _bucket_value(issue, dissue, value)
            for issue, dissue, value in zip(self._issues, self._discrete_issues, offer)
        )

    def _observe(self, offer: Outcome) -> None:
        """Record a single opponent offer into the frequency table."""
        key = self._bucket_key(offer)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1

    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's (normalized) utility of ``offer``.

        Args:
            offer: Offer being considered.

        Returns:
            A value in ``[0, 1]``.
        """
        if offer is None or self._total == 0:
            return 0.5
        mx = max(self._counts.values()) if self._counts else 0
        if mx <= 0:
            return 0.0
        return self._counts.get(self._bucket_key(offer), 0) / mx

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Eval normalized (the model already returns values in ``[0, 1]``).

        Args:
            offer: Offer being considered.
            above_reserve: Unused (kept for interface compatibility).
            expected_limits: Unused (kept for interface compatibility).

        Returns:
            A value in ``[0, 1]``.
        """
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class FrequencyLinearUFunModel(UFunModel):
    """
    A frequency-based opponent model that **assumes the opponent's utility
    function is `LinearAdditiveUtilityFunction`** (a weighted sum of per-issue
    value utilities) — the assumption used by the Bayesian opponent model of
    Hindriks & Tykhonov (2008), and hence the default opponent model for the
    Nice Tit for Tat agent.

    It estimates the opponent's preference for an issue value from how often
    the opponent has *offered* that value (values offered more frequently are
    assumed more preferred) and combines issues with learned weights: an issue
    whose offered-value distribution is more concentrated receives a higher
    weight (``weight = 1 - normalized_entropy`` of the value counts). The
    estimated utility is the weighted sum of the per-issue value scores — a
    linear-additive aggregation.

    The model is updated from every offer the negotiator responds to
    (``before_responding``) and, when the mechanism enables callbacks, from
    ``on_partner_proposal``.

    Args:
        above_reserve: Kept for interface compatibility with `ZeroSumModel`.
        levels: Number of grid values used to discretize continuous issues (so
            they contribute frequency counts instead of being ignored).

    Remarks:
        - Continuous issues are discretized to ``levels`` grid values (via
          `Issue.to_discrete`); offered values are snapped to the nearest grid
          value for counting. Already-discrete issues are used as-is.
        - Returns utilities in ``[0, 1]`` (already normalized).
        - Before any opponent offer is observed, returns a neutral ``0.5`` for
          every outcome (so a Nice Tit for Tat offering strategy falls back to
          mirroring until the model has learned).

    *AI Generated (linear-additive frequency opponent model).*
    """

    above_reserve: bool = True
    levels: int = 10
    _counts: dict = field(init=False, factory=dict)
    _weights: dict = field(init=False, factory=dict)
    _total: int = field(init=False, default=0)
    _issues: list | None = field(init=False, default=None)
    _discrete_issues: list | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """On preferences changed.

        Args:
            changes: Changes.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return
        os_ = self.negotiator.ufun.outcome_space
        issues = getattr(os_, "issues", None) if os_ is not None else None
        self._issues = list(issues) if issues is not None else []
        # Note: we deliberately do NOT clear ``_counts`` here so that any
        # already-learned frequencies survive a repeated preference-change
        # broadcast (preferences only change at negotiation start in practice).
        self._discrete_issues = _discretize_issues(self._issues, self.levels)
        self._update_private_info()

    def before_responding(
        self, state, offer: Outcome | None, source: str | None = None
    ):
        """Learn from the offer the negotiator is about to respond to.

        Args:
            state: Current state.
            offer: The partner's offer being responded to.
            source: Source identifier.
        """
        if offer is not None:
            self._observe(offer)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        """Learn from a partner proposal (only called with ``enable_callbacks``).

        Args:
            state: State when the offer was proposed.
            partner_id: The ID of the agent who proposed.
            offer: The proposal.
        """
        if offer is not None:
            self._observe(offer)

    def _observe(self, offer: Outcome) -> None:
        """Record a single opponent offer into the per-issue frequency tables."""
        if not self._issues or not self._discrete_issues:
            return
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            name = issue.name
            bucket = _bucket_value(issue, dissue, value)
            counts = self._counts.setdefault(name, {})
            counts[bucket] = counts.get(bucket, 0) + 1
        self._total += 1
        # Recompute issue weights from the concentration of each value distribution.
        self._weights = {}
        for name, counts in self._counts.items():
            total = sum(counts.values())
            if total <= 0:
                continue
            entropy = -sum(
                (c / total) * math.log(c / total) for c in counts.values() if c > 0
            )
            nvals = len(counts)
            if nvals > 1:
                entropy /= math.log(nvals)
            self._weights[name] = 1.0 - entropy

    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's (normalized) utility of ``offer``.

        Args:
            offer: Offer being considered.

        Returns:
            A value in ``[0, 1]``.
        """
        if (
            offer is None
            or not self._issues
            or not self._discrete_issues
            or self._total == 0
        ):
            return 0.5
        scores = []
        weights = []
        for issue, dissue, value in zip(self._issues, self._discrete_issues, offer):
            name = issue.name
            bucket = _bucket_value(issue, dissue, value)
            counts = self._counts.get(name, {})
            mx = max(counts.values()) if counts else 0
            scores.append((counts.get(bucket, 0) / mx) if mx > 0 else 0.0)
            weights.append(self._weights.get(name, 0.0))
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
        """Eval normalized (the model already returns values in ``[0, 1]``).

        Args:
            offer: Offer being considered.
            above_reserve: Unused (kept for interface compatibility).
            expected_limits: Unused (kept for interface compatibility).

        Returns:
            A value in ``[0, 1]``.
        """
        if offer is None:
            return 0.0
        return float(self.eval(offer))


@define
class PeekingOpponentModel(UFunModel):
    """
    An *oracle* opponent model that wraps the opponent's true utility function.

    Intended for testing/analysis: the model "peeks" at the opponent's actual
    `BaseUtilityFunction` and delegates ``eval`` / ``eval_normalized`` to it.
    This lets a Nice Tit for Tat agent (or any consumer of ``opponent_model``)
    be tested against a *correct* opponent model, so its concession/Nash-aiming
    behaviour can be checked without the noise of a learned model.

    Args:
        ufun: The opponent's true utility function. May be set at construction
            or assigned later (``model.ufun = ...``) before the negotiation
            starts.

    Remarks:
        - Unlike `FrequencyLinearUFunModel` / `FrequencyUFunModel`, this model
          does not learn; it simply reads the opponent's true utility. Use it
          only in tests/simulation where the opponent's ufun is known.

    *AI Generated (oracle opponent model for testing).*
    """

    ufun: BaseUtilityFunction | None = None

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Register the model in the negotiator's private info.

        Args:
            changes: Changes.
        """
        self._update_private_info()

    def eval(self, offer: Outcome) -> Value:
        """Return the opponent's true (normalized) utility of ``offer``.

        Args:
            offer: Offer being considered.

        Returns:
            The opponent's normalized utility, or ``0.5`` if no ufun is set.
        """
        if self.ufun is None:
            return 0.5
        return float(self.ufun.eval_normalized(offer))

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> Value:
        """Return the opponent's true normalized utility of ``offer``.

        Args:
            offer: Offer being considered.
            above_reserve: Forwarded to the wrapped ufun.
            expected_limits: Forwarded to the wrapped ufun.

        Returns:
            The opponent's normalized utility, or ``0.5`` if no ufun is set.
        """
        if offer is None:
            return 0.0
        if self.ufun is None:
            return 0.5
        return float(self.ufun.eval_normalized(offer, above_reserve, expected_limits))
