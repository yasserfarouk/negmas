from __future__ import annotations

from pytest import fixture

from negmas.outcomes import generate_issues, make_issue


@fixture()
def bissue():
    return make_issue(["be", "not b"], name="binary")


@fixture()
def hamlet():
    return make_issue([f"val {_}" for _ in range(5)], name="THE problem")


@fixture()
def uissue():
    return make_issue([f"val {_}" for _ in range(5)])


@fixture()
def cissue():
    return make_issue((0.0, 1.0), name="c")


@fixture()
def dissue():
    return make_issue(10, name="d")


@fixture()
def sissue():
    return make_issue([f"val {_}" for _ in range(5)], name="s")


@fixture()
def int_issues():
    return generate_issues([5], [10])


@fixture()
def issues(cissue, dissue, sissue):
    return [cissue, dissue, sissue]


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
