from __future__ import annotations

from negmas.mechanisms import Mechanism
from negmas.outcomes import Issue


class MyMechanism(Mechanism):
    def round():
        pass


def test_imap_issues():
    issues = [make_issue(5, f"i{i}") for i in range(5)]
    m = MyMechanism(issues=issues)
    assert len(m.nmi.imap) == 10
    for i in range(5):
        assert m.nmi.imap[i] == f"i{i}"
        assert m.nmi.imap[f"i{i}"] == i


def test_imap_outcomes():
    m = MyMechanism(outcomes=10)
    assert len(m.nmi.imap) == 2
    assert m.nmi.imap[0] == f"0"
    assert m.nmi.imap[f"0"] == 0
