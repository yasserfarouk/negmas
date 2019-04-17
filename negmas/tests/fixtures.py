from pytest import fixture
from negmas.outcomes import Issue


@fixture()
def bissue():
    return Issue(["be", "not b"], name="binary")


@fixture()
def hamlet():
    return Issue(["val {}".format(_) for _ in range(5)], name="THE problem")


@fixture()
def uissue():
    return Issue(["val {}".format(_) for _ in range(5)])


@fixture()
def cissue():
    return Issue((0.0, 1.0), name="c")


@fixture()
def dissue():
    return Issue(10, name="d")


@fixture()
def sissue():
    return Issue(["val {}".format(_) for _ in range(5)], name="s")


@fixture()
def issues(cissue, dissue, sissue):
    return [cissue, dissue, sissue]


@fixture()
def int_issues():
    return Issue.generate([5], [10])


@fixture()
def valid_outcome_dict(issues):
    outcome = {}
    for issue in issues:
        outcome[issue.name] = issue.rand()
    return outcome


@fixture()
def invalid_outcome_dict(issues):
    outcome = {}
    for issue in issues:
        outcome[issue.name] = issue.rand_invalid()
    return outcome


@fixture()
def valid_outcome_list(int_issues):
    outcome = []
    for issue in int_issues:
        outcome.append(issue.rand())
    return outcome


@fixture()
def invalid_outcome_list(int_issues):
    outcome = []
    for issue in int_issues:
        outcome.append(issue.rand_invalid())
    return outcome
