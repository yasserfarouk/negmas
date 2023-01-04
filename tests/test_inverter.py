from negmas import PresortingInverseUtilityFunction, make_issue, make_os
from negmas.preferences.crisp.mapping import MappingUtilityFunction


def test_inv_simple_case_sort_all():
    os = make_os([make_issue(10)])
    outcomes = list(os.enumerate_or_sample())
    ufun = MappingUtilityFunction(
        dict(zip(outcomes, range(len(outcomes)), strict=True)),
        reserved_value=4,
        outcome_space=os,
    )
    inverter = PresortingInverseUtilityFunction(ufun, sort_rational_only=False)
    inverter.init()
    inverted_outcomes = [outcomes[-i] for i in range(1, len(outcomes) + 1)]
    for i in range(len(outcomes)):
        assert (
            inverter.outcome_at(i) == inverted_outcomes[i]
        ), f"{i}: found {inverter.outcome_at(i)} expected {inverted_outcomes[i]}"


def test_inv_simple_case_sort_rational():
    os = make_os([make_issue(10)])
    outcomes = list(os.enumerate_or_sample())
    ufun = MappingUtilityFunction(
        dict(zip(outcomes, range(len(outcomes)), strict=True)),
        reserved_value=4,
        outcome_space=os,
    )
    inverter = PresortingInverseUtilityFunction(ufun, sort_rational_only=True)
    inverter.init()
    inverted_outcomes = [outcomes[-i] for i in range(1, len(outcomes) + 1)]
    for i in range(5):
        assert (
            inverter.outcome_at(i) == inverted_outcomes[i]
        ), f"{i}: found {inverter.outcome_at(i)} expected {inverted_outcomes[i]}"
    for i in range(6, 10):
        assert (
            inverter.outcome_at(i) == outcomes[i - 6]
        ), f"{i}: found {inverter.outcome_at(i)} expected {outcomes[i]}"
