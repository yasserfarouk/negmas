from negmas.gb.negotiators.timebased import AspirationNegotiator
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.ops import compare_ufuns
from negmas.preferences.value_fun import AffineFun, LinearFun
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators.modular import BOANegotiator
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.components.acceptance import ACNext
from negmas.gb.components.genius.models import GSmithFrequencyModel


def test_gsmith_frequency_model_initializes_base_ufun_attributes():
    """Test that GSmithFrequencyModel properly initializes BaseUtilityFunction attributes.

    This ensures that the attrs-based GeniusOpponentModel correctly calls
    BaseUtilityFunction.__init__ via __attrs_post_init__.
    """
    model = GSmithFrequencyModel()

    # Verify all attributes set by BaseUtilityFunction.__init__ are present and have correct defaults
    assert hasattr(model, "_reserved_value"), "Missing _reserved_value attribute"
    assert hasattr(model, "_invalid_value"), "Missing _invalid_value attribute"
    assert hasattr(model, "_cached_inverse"), "Missing _cached_inverse attribute"
    assert hasattr(model, "_cached_inverse_type"), (
        "Missing _cached_inverse_type attribute"
    )

    # Check default values match BaseUtilityFunction defaults
    assert model._reserved_value == float("-inf"), (
        f"Expected -inf, got {model._reserved_value}"
    )
    assert model._invalid_value is None, f"Expected None, got {model._invalid_value}"
    assert model._cached_inverse is None, f"Expected None, got {model._cached_inverse}"
    assert model._cached_inverse_type is None, (
        f"Expected None, got {model._cached_inverse_type}"
    )

    # Verify the model is an instance of BaseUtilityFunction
    assert isinstance(model, BaseUtilityFunction)

    # Verify the reserved_value property works (uses _reserved_value internally)
    assert model.reserved_value == float("-inf")


def calc_scores(m: SAOMechanism) -> dict[str, dict[str, float]]:
    """Compute scores for the given agreement according the ANL 2026 rules."""

    # extract the agreement
    agreement = m.agreement

    # extract negotiator names
    negotiators = [_.__class__.__name__ for _ in m.negotiators]

    # find advantages (utility above reserved value)
    advantages = [
        float(_.ufun(agreement)) - float(_.ufun.reserved_value) if _.ufun else 0.0
        for _ in m.negotiators
    ]

    # calculate modeling accuracies
    ufuns = [_.ufun for _ in m.negotiators]
    models = [_.opponent_ufun for _ in m.negotiators]
    models.reverse()
    accuracies = [
        (1 + compare_ufuns(u, model, method="kendall", outcome_space=m.outcome_space))
        / 2
        for u, model in zip(ufuns, models)
    ]

    # normalize accuracies so that we divide one point among all negotiators with
    # negotiators with higher accuracy getting higher part of this point.
    accsum = sum(accuracies)
    if accsum > 0:
        accuracies = [_ / accsum for _ in accuracies]
    else:
        accuracies = [0] * len(negotiators)
    accuracies.reverse()
    # return final scores. You can improve your score in one of three ways:
    # 1. Increase your advantage (negotiating a better deal for yourself)
    # 2. Increase your modeling accuracy (better opponent modeling)
    # 3. Decrease your opponent's accuracy (confuse their opponent modeling)
    return dict(
        zip(
            negotiators,
            (
                dict(Advavntage=adv, Accuracy=acc, Score=adv + acc)
                for adv, acc in zip(advantages, accuracies)
            ),
        )
    )


class BOANeg(BOANegotiator):
    def __init__(self, *args, **kwargs):
        offering = TimeBasedOfferingPolicy()
        kwargs |= dict(
            acceptance=ACNext(offering), offering=offering, model=GSmithFrequencyModel()
        )
        super().__init__(*args, **kwargs)


def test_smith_frequency_model():
    os = make_os([make_issue(10, "i1"), make_issue(10, "i2")])
    m = SAOMechanism(
        n_steps=100,
        outcome_space=os,
        ignore_negotiator_exceptions=False,
        one_offer_per_step=True,
    )
    m.add(
        BOANeg(
            ufun=LinearAdditiveUtilityFunction(
                values=[LinearFun(slope=0.1), LinearFun(slope=0.1)],
                weights=[0.5, 0.5],
                outcome_space=os,
            ),
            id="boa",
        )
    )
    m.add(
        AspirationNegotiator(
            ufun=LinearAdditiveUtilityFunction(
                values=[AffineFun(slope=-0.1, bias=10), LinearFun(slope=0.1)],
                weights=[0.8, 0.2],
                outcome_space=os,
            ),
            id="asp",
        )
    )
    m.run()
    print(calc_scores(m))
    trace = m.extended_trace
    assert len(trace) > 2, f"{trace}"
