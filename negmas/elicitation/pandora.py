from __future__ import annotations

import functools
import random
import time
from heapq import heapify, heappop
from math import sqrt
from typing import Callable

import numpy as np
import scipy.optimize as opt

from ..common import MechanismState, Value
from ..models.acceptance import AdaptiveDiscreteAcceptanceModel
from ..negotiators.helpers import PolyAspiration
from ..outcomes import Outcome
from ..sao import AspirationNegotiator, SAONegotiator
from .base import BaseElicitor
from .common import _loc, _scale
from .expectors import (
    AspiringExpector,
    BalancedExpector,
    MaxExpector,
    MeanExpector,
    MinExpector,
)

__all__ = [
    "BasePandoraElicitor",
    "PandoraElicitor",
    "OptimalIncrementalElicitor",
    "FullElicitor",
    "RandomElicitor",
    "weitzman_index_uniform",
    "FastElicitor",
    "MeanElicitor",
    "BalancedElicitor",
    "AspiringElicitor",
    "PessimisticElicitor",
    "OptimisticElicitor",
]


def weitzman_index_uniform(
    loc: float, scale: float, cost: float, time_discount: float = 1.0
) -> float:
    """Implements Weitzman's 1979 Bandora's Box index calculation.

    Args:
        loc: Loc of the uniform distribution to update
        scale: Scale of the uniform distribution to update
        cost: cost of opening a box
        time_discount: time discount. Assumed unity (no discounting) for
                       negotiation.

    Returns:
        The index value of this distribution.

    """
    # assume zi < l
    end = loc + scale
    z = time_discount * (loc + end) / 2.0 - cost

    b = -2 * (end + scale * (1.0 - time_discount) / time_discount)
    c = end * end - 2 * scale * cost / time_discount

    d = b * b - 4 * c
    if d < 0:
        z1 = z2 = -1.0
    else:
        d = sqrt(d)
        z1 = (d - b) / 2.0
        z2 = -(d + b) / 2.0

    if z <= loc and not loc < z1 <= end and not loc < z2 <= end:
        return z
    if z > loc and loc < z1 <= end and not loc < z2 <= end:
        return z1
    if z > loc and not loc < z1 <= end and loc < z2 <= end:
        return z2

    if z <= loc or (z - loc) < 1e-5:
        return z
    elif loc < z1 <= end:
        return z1
    elif loc < z2 <= end:
        return z2
    for _ in (z1, z2):
        if abs(_ - loc) < 1e-5 or abs(_ - end) < 1e-3:
            return _
    print(
        "No solutions are found for (l={}, s={}, c={}, time_discount={}) [{}, {}, {}]".format(
            loc, scale, cost, time_discount, z, z1, z2
        )
    )
    return 0.0


class BasePandoraElicitor(BaseElicitor):
    """
    The base class of all Pandora's box based algorithms.

    Args:
        user: A `User` object that can be elicited
        strategy: An elicitation strategy that determines the questions to
                  be asked and their order.
        base_negotiator: The base negotiator used for responding and proposing
        deep_elicitation: If True, a single elicitation act (`elicit_single`)
                          will elicit the utility value of an outcome until
                          its uncertainty is under `epsilon`.
        opponent_model_factory: A callable to create opponent models. An opponent model is
                                used to estimate acceptance probability of outcomes.
        expector_factory: A callable to create an `Expector`
        single_elicitation_per_round: If True, only a single elicitation act (call to
                                      `elicit_single` is allowed per negotiation round.
        continue_eliciting_past_reserved_val: If True, continue elicitation even if the
                                              estimated utility is under the reserved value.
        epsilon: A small number to represent small uncertainty after which no more elicitation
                 is done.
        true_utility_on_zero_cost: If `True`, zero cost will force the final elicited
                                   value of any outcome to exactly match the utility
                                   function. If `False`, the final utility value after
                                   elicitation may be within `epsilon` from the true
                                   value.
        assume_uniform: If True, assume that all utility distributions are uniform
        user_model_in_index: Use the user model in the index
        precalculated_index: The index is calculated once and never updated
                             when the opponent model probabilities of acceptance
                             change.
        incremental: Only valid if `precalculated_index` is not set. If set,
                     only the outcomes for which the opponemt model acceptance
                     probability have changed will be updated whenever such
                     a change occures. Otherwise, all outcomes will be updated
                     on every change.
        max_aspiration: Maximum aspiration.
        aspiration_type: Aspiration type. Can be "boulware", "linear", "conceder" or a numric
                         exponent.
    """

    def __init__(
        self,
        user: User,
        strategy: EStrategy,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool,
        opponent_model_factory: None
        | (
            Callable[[NegotiatorMechanismInterface], DiscreteAcceptanceModel]
        ) = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(nmi=x),
        expector_factory: Expector | Callable[[], Expector] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
        incremental=True,
        max_aspiration=0.99,
        aspiration_type="boulware",
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        self.add_capabilities(
            {
                "propose": True,
                "respond": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.my_last_proposals: Outcome | None = None
        self.deep_elicitation = deep_elicitation
        self.elicitation_history = []
        self.cutoff_utility = None
        self.opponent_model = None
        self._elicitation_time = None
        self.offerable_outcomes = (
            []
        )  # will contain outcomes with known or at least elicited utilities
        self.cutoff_utility = None
        self.unknown = None
        self.assume_uniform = assume_uniform
        self.user_model_in_index = user_model_in_index
        self.precalculated_index = precalculated_index
        self.incremental = incremental
        self.__asp = PolyAspiration(max_aspiration, aspiration_type)

    def utility_at(self, x):
        return self.__asp.utility_at(x)

    def utility_on_rejection(self, outcome: Outcome, state: MechanismState) -> Value:
        """Uses the aspiration level as the utility of rejection.

        Remarks:
            - This does not depend on using `AspirationNegotiator` as the
              base_negotiator.
        """
        return self.utility_at(state.relative_time)

    def update_cutoff_utility(self) -> None:
        r"""
        Updates the cutoff utility under which no elicitation is done.

        Remarks:
            - By default it uses :math:`max_{o \in \Omega \cup \phi}{u(o)}` for
              all outcomes :math:`o \in \Omega` and the None outcome representing
              reserved value
        """
        self.cutoff_utility = self.reserved_value
        expected_utilities = [
            self.user_preferences(outcome) for outcome in self.offerable_outcomes
        ]
        if len(expected_utilities) > 0:
            self.cutoff_utility = max(expected_utilities)

    def do_elicit(self, outcome: Outcome, state: MechanismState) -> Value:
        """
        Does a real elicitation step.

        Args:
            outcome: The outcome to elicit
            state: The state at which elicitation is happening

        Remarks:
            - If `deep_elicitation` is set, the strategy is applied until the
              uncertainty in the utility value for `outcome` is less than the
              accuracty limit otherwise, apply it once.
        """
        if not self.deep_elicitation:
            return self.strategy.apply(user=self.user, outcome=outcome)[0]

        while True:
            u, _ = self.strategy.apply(user=self.user, outcome=outcome)
            if isinstance(u, float):
                break
            if u.scale < self.acc_limit:
                u = float(u)
                break

        # we do that after the normal elicitation so that the elicitation time
        # is recorded correctly
        # TODO find a way to avoid calling deep elicitation in this case
        # for speedup. The problem is that I need to calcualate the elicitation
        # time correctly (at least estimate it)
        if self.user.cost_of_asking() == 0.0 and self.true_utility_on_zero_cost:
            return self.user.ufun(outcome)
        return u

    def z_index(self, updated_outcomes: list[Outcome] | None = None):
        """
        Update the internal z-index or create it if needed.

        Args:
            updated_outcomes: A list of the outcomes with updated utility values.
        """
        n_outcomes = self._nmi.n_outcomes
        outcomes = self._nmi.outcomes
        unknown = self.unknown
        if unknown is None:
            unknown = [(-self.reserved_value, _) for _ in range(n_outcomes)]
        else:
            unknown = [_ for _ in unknown if _[1] is not None]

        xw = list(self.preferences.distributions.values())
        if len(unknown) == 0:
            return
        unknown_indices = [_[1] for _ in unknown if _[1] is not None]
        if updated_outcomes is None:
            updated_outcomes = unknown_indices
        else:
            updated_outcomes = [self.indices[_] for _ in updated_outcomes]
            updated_outcomes = list(set(unknown_indices).intersection(updated_outcomes))
        if len(updated_outcomes) == 0:
            return unknown
        z = unknown
        if self.assume_uniform:
            for j, (u, i) in enumerate(unknown):
                if i is None or i not in updated_outcomes:
                    continue
                loc = xw[i].loc if not isinstance(xw[i], float) else xw[i]
                scale = xw[i].scale if not isinstance(xw[i], float) else 0.0
                if self.user_model_in_index:
                    p = self.opponent_model.probability_of_acceptance(outcomes[i])
                    current_loc = loc
                    loc = p * loc + (1 - p) * self.reserved_value
                    scale = (
                        p * (current_loc + scale) + (1 - p) * self.reserved_value - loc
                    )
                cost = self.user.cost_of_asking()
                z[j] = (-weitzman_index_uniform(loc, scale, cost=cost), i)
        else:

            def qualityfun(z, distribution, cost):
                c_estimate = distribution.expect(lambda x: x - z, lb=z, ub=1.0)
                if self.user_model_in_index:
                    p = self.opponent_model.probability_of_acceptance(outcomes[i])
                    c_estimate = p * c_estimate + (1 - p) * self.reserved_value
                return sqrt(c_estimate - cost)

            for j, (u, i) in enumerate(unknown):
                if i is None or i not in updated_outcomes:
                    continue
                cost = self.user.cost_of_asking()
                f = functools.partial(qualityfun, distribution=xw[i], cost=cost)
                z[j] = (
                    -opt.minimize(
                        f, x0=np.asarray([u]), bounds=[(0.0, 1.0)], method="L-BFGS-B"
                    ).x[0],
                    i,
                )
                # we always push the reserved value for the outcome None representing breaking
        heapify(z)
        return z

    def init_unknowns(self):
        """
        Initializes the unknowns list which is a list of Tuples [-u(o), o] for o in outcomes.
        """
        self.unknown = self.z_index(updated_outcomes=None)

    def offer_to_elicit(self) -> tuple[float, int | None]:
        """
        Returns the maximum estimaged utility and the corresponding outocme


        Remarks:
            - This method assumes that each element of the unkown list is a
              `tuple` of negative the utility value and the outcome.
            - It also assumest that the first item in the unknown list contains
              the outocme with maximum estimated utility and the negative
              of the corresponding estimated utility.
        """
        if self.unknown is None:
            self.init_unknowns()
        unknowns = self.unknown
        if len(unknowns) > 0:
            return -unknowns[0][0], unknowns[0][1]
        return self.reserved_value, None

    def update_best_offer_utility(self, outcome: Outcome, u: Value):
        """
        Updates the unknown list (and makes sure it is a heap) given the given utility
        value for the given outcome.
        """
        if self.unknown is None:
            self.init_unknowns()
        self.unknown[0] = (
            -weitzman_index_uniform(_loc(u), _scale(u), self.user.cost_of_asking()),
            self.unknown[0][1],
        )
        heapify(self.unknown)

    def remove_best_offer_from_unknown_list(self) -> tuple[float, int]:
        """
        Removes the best offer from the unkonwn list and returns it.
        """
        if self.unknown is None:
            self.init_unknowns()
        return heappop(self.unknown)

    def elicit_single(self, state: MechanismState):
        """
        Does a single elicitation act

        Args:
            state: mechanism state

        Remarks:
            The process goes as follows:

            1. Find the offer to elicit using `offer_to_elicit`.
            2. If the utility of that offer is less than the cutoff utility,
               or the best offer to elicit is `None`, stop eliciting and return.
            3. call `do_elicit` once.
            4. update the distribution of the elicited outcome
            5. add the elicited outcome to the offerable outcomes
            6. If the utility of the best outcome is a number, remove it
               from the unknown outcomes list otherwise jsut update it
            7. Update the cutoff utility and elicitation history.
        """
        z, best_index = self.offer_to_elicit()
        if z < self.cutoff_utility:
            return False
        if best_index is None:
            return self.continue_eliciting_past_reserved_val
        outcome = self._nmi.outcomes[best_index]
        u = self.do_elicit(outcome, None)
        self.preferences.distributions[outcome] = u
        expected_value = self.offering_utility(outcome, state=state)
        # TODO confirm that offerable_outcomes need not be unique or use
        # a set for it.
        self.offerable_outcomes.append(outcome)
        if isinstance(u, float):
            self.remove_best_offer_from_unknown_list()
        else:
            self.update_best_offer_utility(outcome, u)
        self.cutoff_utility = max(
            (self.cutoff_utility, self.expect(expected_value, state=state))
        )
        self.elicitation_history.append((outcome, u, state.step))
        return True

    def init_elicitation(
        self,
        preferences: IPUtilityFunction | Distribution | None,
        **kwargs,
    ):
        """
        Initializes the elicitation process (called only once).

        Remarks:

            - After calling the parent's `init_elicitation`, this method calls
              `init_unknowns` to initialize the unknown list.
        """
        super().init_elicitation(preferences=preferences, **kwargs)
        strt_time = time.perf_counter()
        self.cutoff_utility = self.reserved_value
        self.unknown = None  # needed as init_unknowns uses unknown
        self.init_unknowns()
        self._elicitation_time += time.perf_counter() - strt_time

    def before_eliciting(self):
        """Called before starting elicitation at every negotiation round.

        Remarks:
            - It just updates cutoff utility
        """
        self.update_cutoff_utility()

    def can_elicit(self) -> bool:
        """
        Checks whether there are any unknowns in the unknowns list.
        """
        if self.unknown is None:
            self.init_unknowns()
        return self.unknown and len(self.unknown) != 0

    def on_opponent_model_updated(
        self, outcomes: list[Outcome], old: list[float], new: list[float]
    ) -> None:
        """
        Called when the opponent model is updated.

        Args:
            outcomes: changed outcomes
            old: old probabilities of acceptance
            new: new probabilities of acceptance

        Remarks:
            Updates the unknown list only if precalculated_index was not set.
        """
        if not self.precalculated_index:
            self.unknown = self.z_index(
                updated_outcomes=outcomes if self.incremental else None
            )


class FullElicitor(BasePandoraElicitor):
    """
    Does full deep elicitation in the first call to `elicit`.
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        **kwargs,
    ) -> None:
        kwargs["deep_elicitation"] = True
        super().__init__(
            strategy=strategy,
            user=user,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
            **kwargs,
        )
        self.elicited = {}

    def update_best_offer_utility(self, outcome: Outcome, u: Value):
        pass

    def init_elicitation(
        self,
        preferences: IPUtilityFunction | Distribution | None,
        **kwargs,
    ):
        super().init_elicitation(preferences=preferences)
        strt_time = time.perf_counter()
        self.elicited = False
        self._elicitation_time += time.perf_counter() - strt_time

    def elicit(self, state: MechanismState):
        if not self.elicited:
            outcomes = self._nmi.outcomes
            utilities = [
                self.expect(self.do_elicit(outcome, state=state), state=state)
                for outcome in self._nmi.outcomes
            ]
            self.offerable_outcomes = list(outcomes)
            self.elicitation_history = [zip(outcomes, utilities)]
            self.elicited = True

    def init_unknowns(self) -> list[tuple[float, int]]:
        self.unknown = []


class RandomElicitor(BasePandoraElicitor):
    """
    Uses a random index instead of the optimal z-index used by the Pandora's
    box solution.
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        deep_elicitation=True,
        true_utility_on_zero_cost=False,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        opponent_model_factory: None
        | (
            Callable[[NegotiatorMechanismInterface], DiscreteAcceptanceModel]
        ) = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(nmi=x),
        single_elicitation_per_round=False,
        **kwargs,
    ) -> None:
        kwargs["epsilon"] = 0.001
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            opponent_model_factory=opponent_model_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            base_negotiator=base_negotiator,
            **kwargs,
        )

    def init_unknowns(self) -> None:
        n = self._nmi.n_outcomes
        z: list[tuple[float, int | None]] = list(
            zip((-random.random() for _ in range(n + 1)), range(n + 1))
        )
        z[-1] = (z[-1][0], None)
        heapify(z)
        self.unknown = z

    def update_best_offer_utility(self, outcome: Outcome, u: Value):
        pass


class PandoraElicitor(BasePandoraElicitor):
    """
    Implements the original [Baarslag and Gerding]_'s Pandora's box based elicitation
    algorithm (when used with the default parameters).


    Args:
        strategy: The elicitation strategy
        user: The user to elicit


    .. [Baarslag and Gerding] Tim Baarslag and Enrico H. Gerding. 2015.
       Optimal incremental preference elicitation during negotiation.
       In Proceedings of the 24th International Conference on Artificial
       Intelligence (IJCAI’15). AAAI Press, 3–9
       (https://dl.acm.org/doi/10.5555/2832249.2832250).
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(
            dict(
                base_negotiator=AspirationNegotiator(),
                opponent_model_factory=lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(
                    nmi=x
                ),
                expector_factory=MeanExpector,
                deep_elicitation=True,
                single_elicitation_per_round=False,
                continue_eliciting_past_reserved_val=False,
                epsilon=0.001,
                assume_uniform=True,
                user_model_in_index=True,
                precalculated_index=False,
                incremental=True,
                true_utility_on_zero_cost=False,
            )
        )
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )


class FastElicitor(PandoraElicitor):
    """
    Same as `PandoraElicitor` but does not use deep elicitation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["deep_elicitation"] = False
        super().__init__(*args, **kwargs)

    def update_best_offer_utility(self, outcome: Outcome, u: Value):
        """We need not do anything here as we will remove the outcome anyway to the known list"""

    def do_elicit(self, outcome: Outcome, state: MechanismState):
        return self.expect(super().do_elicit(outcome, state), state=state)


class OptimalIncrementalElicitor(FastElicitor):
    """
    Same as `FastElicitor` but uses incremental elicitation which simply
    means that it only updates the index for outcomes that are affected
    by changes in the opponent model.
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(dict(incremental=True))
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )


class MeanElicitor(OptimalIncrementalElicitor):
    """Same as `OptimalIncrementalElicitor` using `MeanExpector` for
    estimating utilities"""

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(dict(expector_factory=MeanExpector))
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )


class BalancedElicitor(OptimalIncrementalElicitor):
    """Same as `OptimalIncrementalElicitor` using `MeanExpector` for
    estimating utilities"""

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(dict(expector_factory=BalancedExpector))
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )


class AspiringElicitor(OptimalIncrementalElicitor):
    """
    Same as `OptimalIncrementalElicitor` using aspiration level for
    estimating utilities.
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        max_aspiration: float = 1.0,
        aspiration_type: float | str = "linear",
        **kwargs,
    ) -> None:
        kwargs.update(
            dict(
                expector_factory=lambda: AspiringExpector(
                    max_aspiration=max_aspiration,
                    aspiration_type=aspiration_type,
                    nmi=self._nmi,
                )
            )
        )
        super().__init__(
            strategy=strategy,
            user=user,
            max_aspiration=max_aspiration,
            aspiration_type=aspiration_type,
            **kwargs,
        )


class PessimisticElicitor(OptimalIncrementalElicitor):
    """Same as `OptimalIncrementalElicitor` using the minimum to estimate
    utilities."""

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(dict(expector_factory=MinExpector))
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )


class OptimisticElicitor(OptimalIncrementalElicitor):
    """Same as `OptimalIncrementalElicitor` using the maximum to estimate
    utilities."""

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        **kwargs,
    ) -> None:
        kwargs.update(dict(expector_factory=MaxExpector))
        super().__init__(
            strategy=strategy,
            user=user,
            **kwargs,
        )
