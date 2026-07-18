"""Logical-reasoning / heuristic opponent model (survey §5.3.4).

An ordinal opponent model that learns the region of *acceptable* offers for the
opponent by inductive reasoning over the negotiation trace, following the
"candidate elimination" family of Baarslag, Hendrikx, Hindriks & Jonker,
*Learning about the opponent in automated bilateral negotiation: a comprehensive
survey of opponent modeling techniques*, JAAMAS 30:849-898 (2016), §5.3.4.

*AI Generated (candidate-elimination acceptable-offer model).*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from .ufun import UFunModel

if TYPE_CHECKING:
    from negmas import PreferencesChange, Value
    from negmas.gb import GBState
    from negmas.outcomes import ExtendedOutcome, Outcome

__all__ = ["CandidateEliminationModel"]


@define
class CandidateEliminationModel(UFunModel):
    """Ordinal opponent model via candidate elimination (survey §5.3.4).

    Candidate elimination [8,9] is an inductive-learning algorithm that assumes
    only that the opponent's preferences do not change during the negotiation.
    Each offer the opponent *sends* is a positive instance (its values are
    acceptable to the opponent); each of our own offers that the opponent
    *rejects* (i.e. counters instead of accepting) is a negative instance
    (something in it is unacceptable).

    A full version space over whole offers is exponential, so — as the survey
    notes such models "only learn part of the relationships" — this is a
    per-issue rendition: for each issue it keeps the set of values seen in the
    opponent's offers (acceptable) and the set seen only in rejected offers of
    ours (suspect). The estimated ordinal utility of an offer is the mean
    per-issue score, where a value is scored ``1`` if confirmed acceptable, ``0``
    if only ever seen in a rejected offer, and ``0.5`` if not yet seen (the
    general-boundary default: unseen values may still be acceptable).

    Negatives are derived automatically: whenever the opponent makes a new offer,
    our most recent proposal is treated as rejected. They can also be supplied
    explicitly via :meth:`note_rejected`.

    Remarks:
        - This is an *ordinal* model — the values it returns rank outcomes but are
          not calibrated cardinal utilities.
        - Returns ``0.5`` for every outcome until some evidence is gathered.

    *AI Generated (candidate-elimination acceptable-offer model).*
    """

    _pos: dict = field(init=False, factory=dict)
    _neg: dict = field(init=False, factory=dict)
    _issues: list | None = field(init=False, default=None)
    _last_proposed: Outcome | None = field(init=False, default=None)

    def on_preferences_changed(self, changes: list[PreferencesChange]):
        """Set up the issue list and register the model."""
        if not self.negotiator or not self.negotiator.ufun:
            return
        os_ = self.negotiator.ufun.outcome_space
        issues = getattr(os_, "issues", None) if os_ is not None else None
        self._issues = list(issues) if issues is not None else []
        self._update_private_info()

    def after_proposing(
        self, state: GBState, offer: Outcome | ExtendedOutcome | None, dest=None
    ):
        """Remember our own proposal so a later opponent offer marks it rejected."""
        out = getattr(offer, "outcome", offer)
        if out is not None:
            self._last_proposed = out  # type: ignore[assignment]

    def before_responding(
        self, state, offer: Outcome | None, source: str | None = None
    ):
        """Learn from the opponent offer we are about to respond to."""
        if offer is not None:
            self._observe_positive(offer)

    def on_partner_proposal(self, state, partner_id: str, offer: Outcome) -> None:
        """Learn from a partner proposal (only with ``enable_callbacks``)."""
        if offer is not None:
            self._observe_positive(offer)

    def _observe_positive(self, offer: Outcome) -> None:
        """Record the opponent's offer as a positive (acceptable) instance.

        Also derives a negative instance: the opponent making this offer means it
        rejected our most recent proposal.
        """
        if self._last_proposed is not None and self._last_proposed != offer:
            self.note_rejected(self._last_proposed)
            self._last_proposed = None
        if not self._issues:
            return
        for issue, value in zip(self._issues, offer):
            self._pos.setdefault(issue.name, set()).add(value)

    def note_rejected(self, offer: Outcome) -> None:
        """Record ``offer`` as a negative (rejected-by-opponent) instance."""
        if not self._issues or offer is None:
            return
        for issue, value in zip(self._issues, offer):
            self._neg.setdefault(issue.name, set()).add(value)

    def eval(self, offer: Outcome) -> Value:
        """Estimate the opponent's ordinal utility of ``offer`` in ``[0, 1]``."""
        if offer is None or not self._issues:
            return 0.5
        if not self._pos and not self._neg:
            return 0.5
        scores = []
        for issue, value in zip(self._issues, offer):
            name = issue.name
            if value in self._pos.get(name, set()):
                scores.append(1.0)
            elif value in self._neg.get(name, set()):
                scores.append(0.0)
            else:
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 0.5

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
