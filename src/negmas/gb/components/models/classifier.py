"""Preference-profile classification opponent model (survey §5.3.2).

An opponent model that treats preference learning as a *classification* problem:
given a finite set of candidate preference profiles (utility functions), it
decides — via Bayesian learning — which one the opponent is most likely using,
following Lin et al. [123,124] as described by Baarslag, Hendrikx, Hindriks &
Jonker, *Learning about the opponent in automated bilateral negotiation: a
comprehensive survey of opponent modeling techniques*, JAAMAS 30:849-898 (2016),
§5.3.2.

*AI Generated (Lin et al. Luce-number preference-profile classifier).*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.preferences.base_ufun import BaseUtilityFunction

from .ufun import UFunModel

if TYPE_CHECKING:
    from negmas import PreferencesChange, Value
    from negmas.outcomes import Outcome

__all__ = ["LuceProfileClassifierModel"]


@define
class LuceProfileClassifierModel(UFunModel):
    """Classifies the opponent among candidate profiles via Luce numbers.

    The model is given a finite set of candidate opponent utility functions
    (``profiles``). Following Lin et al. [123,124], a rational opponent using
    profile ``t`` is assumed to offer outcome ``o`` with probability proportional
    to its *Luce number* ``L_t(o) = u_t(o) / Σ_{o'} u_t(o')`` — the outcome's
    utility divided by the sum of utilities over the outcome space. Each observed
    opponent offer therefore updates a Bayesian posterior over the candidate
    profiles (accumulated in log-space for numerical stability).

    The estimated opponent utility of an outcome is the posterior-weighted average
    of the candidate utilities (or, if ``use_map`` is set, the utility under the
    single most-probable profile — the choice the survey describes).

    Args:
        profiles: The candidate opponent utility functions to classify among.
        use_map: If ``True``, evaluate using only the maximum-a-posteriori
            profile; otherwise use the posterior-weighted average of all profiles.
        max_outcomes: Cap on the number of outcomes sampled to compute the Luce
            denominators (for large/continuous spaces).

    Remarks:
        - The Luce denominators and combination use each candidate's *normalized*
          utility (``eval_normalized``), so utilities are non-negative and
          comparable across profiles.
        - Before any offer is observed the posterior is uniform, so ``eval``
          returns the plain average of the candidate utilities.

    *AI Generated (Lin et al. Luce-number classifier).*
    """

    profiles: list[BaseUtilityFunction] = field(factory=list)
    use_map: bool = False
    max_outcomes: int = 10000
    _logpost: list[float] = field(init=False, factory=list)
    _logz: list[float] = field(init=False, factory=list)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Compute Luce denominators and initialize a uniform log-posterior."""
        import math

        n = len(self.profiles)
        self._logpost = [0.0] * n  # uniform prior (equal, unnormalized in log-space)
        self._logz = [0.0] * n
        outcomes = self._outcomes()
        for i, profile in enumerate(self.profiles):
            z = 0.0
            for o in outcomes:
                z += max(0.0, float(profile.eval_normalized(o)))
            self._logz[i] = math.log(z) if z > 0 else 0.0
        self._update_private_info()

    def _outcomes(self) -> list[Outcome]:
        if not self.negotiator or not self.negotiator.ufun:
            return []
        os_ = self.negotiator.ufun.outcome_space
        if os_ is None:
            return []
        try:
            return list(os_.enumerate_or_sample(max_cardinality=self.max_outcomes))
        except Exception:  # pragma: no cover - defensive
            return []

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

    def _observe(self, offer: Outcome) -> None:
        """Update the log-posterior with the Luce likelihood of ``offer``."""
        import math

        if not self.profiles or not self._logpost:
            return
        for i, profile in enumerate(self.profiles):
            u = max(0.0, float(profile.eval_normalized(offer)))
            # log Luce number = log u - log Z ; add tiny epsilon to avoid log(0).
            self._logpost[i] += math.log(u + 1e-12) - self._logz[i]

    def _posterior(self) -> list[float]:
        """Return the normalized posterior over profiles (softmax of log-posterior)."""
        import math

        if not self._logpost:
            return []
        m = max(self._logpost)
        exps = [math.exp(lp - m) for lp in self._logpost]
        s = sum(exps)
        if s <= 0:
            return [1.0 / len(exps)] * len(exps)
        return [e / s for e in exps]

    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's normalized utility of ``offer`` in ``[0, 1]``."""
        if offer is None or not self.profiles:
            return 0.5
        post = self._posterior()
        if self.use_map:
            best = max(range(len(post)), key=lambda i: post[i])
            return float(self.profiles[best].eval_normalized(offer))
        return float(
            sum(
                p * float(prof.eval_normalized(offer))
                for p, prof in zip(post, self.profiles)
            )
        )

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
