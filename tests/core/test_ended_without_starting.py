from __future__ import annotations

from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LU
from negmas.sao import AspirationNegotiator, SAOMechanism


def _issues():
    return [make_issue([f"v{i}" for i in range(5)], "i0")]


def _mechanism(dynamic_entry: bool, n_negotiators: int, **kwargs) -> SAOMechanism:
    issues = _issues()
    m = SAOMechanism(issues=issues, dynamic_entry=dynamic_entry, **kwargs)
    for i in range(n_negotiators):
        m.add(
            AspirationNegotiator(name=f"a{i}"),
            preferences=LU.random(issues=issues, reserved_value=0.0),
        )
    return m


def test_ended_without_starting_is_reported_as_ended():
    """A mechanism that can never start must report `ended`.

    Without this, `not state.ended` stays True forever and callers that poll it to
    decide whether to keep stepping (``World._step_negotiations``) never stop.
    """
    m = _mechanism(dynamic_entry=False, n_negotiators=1, n_steps=10)
    for _ in range(5):
        state = m.step()
        assert not state.started
        assert state.ended_without_starting
        assert state.ended_or_never_started
        assert not state.ended, "`ended` must keep implying `started`"
        assert not state.running


def test_dynamic_entry_mechanism_stays_open_before_starting():
    """The regression guard for the fix that was NOT taken.

    ``is_running = state.running`` would have read False here and closed a
    negotiation that is deliberately waiting for another negotiator to join.
    """
    m = _mechanism(dynamic_entry=True, n_negotiators=1, n_steps=10)
    for _ in range(5):
        state = m.step()
        assert not state.started
        assert not state.ended_without_starting
        assert not state.ended_or_never_started, (
            "a mechanism still accepting negotiators is not done"
        )
        assert not state.ended


def test_fresh_state_has_not_ended():
    m = _mechanism(dynamic_entry=False, n_negotiators=2, n_steps=10)
    assert not m.state.ended
    assert not m.state.ended_or_never_started
    assert not m.state.ended_without_starting


def test_normal_negotiation_still_reports_ended_on_agreement():
    m = _mechanism(dynamic_entry=False, n_negotiators=2, n_steps=10)
    state = m.run()
    assert state.ended
    assert state.ended_or_never_started
    assert state.started
    assert not state.ended_without_starting, (
        "a negotiation that ran is not `concluded`-without-start"
    )


def test_timeout_before_starting_is_reported_as_ended():
    """``n_steps``/``time_limit`` set ``timedout`` before the negotiation starts.

    ``ended`` requires ``started`` and so reads False for these -- by design, kept
    that way -- which is why the stepping predicate is ``finished``.
    """
    m = _mechanism(dynamic_entry=False, n_negotiators=2, n_steps=10)
    m._current_state.timedout = True
    assert not m.state.started
    assert m.state.ended_or_never_started
    assert not m.state.ended, "`ended` must keep implying `started`"


def test_ended_still_implies_started():
    """`ended` is a narrower predicate than `finished` and must stay that way.

    Downstream code (scml, anl) reads `ended` as "ran and then concluded"; widening
    it would silently change that meaning everywhere.
    """
    for dynamic_entry in (False, True):
        for n in (1, 2):
            m = _mechanism(dynamic_entry=dynamic_entry, n_negotiators=n, n_steps=5)
            for _ in range(8):
                st = m.step()
                assert not st.ended or st.started
                assert not st.ended or st.ended_or_never_started
