from negmas.mechanisms import Mechanism
from negmas.outcomes import Issue


class MyMechanism(Mechanism):
    def round():
        pass


def test_imap_issues():
    issues = [Issue(5, f"i{i}") for i in range(5)]
    m = MyMechanism(issues=issues)
    assert len(m.ami.imap) == 10
    for i in range(5):
        assert m.ami.imap[i] == f"i{i}"
        assert m.ami.imap[f"i{i}"] == i


def test_imap_outcomes():
    m = MyMechanism(outcomes=10)
    assert len(m.ami.imap) == 2
    assert m.ami.imap[0] == f"0"
    assert m.ami.imap[f"0"] == 0
