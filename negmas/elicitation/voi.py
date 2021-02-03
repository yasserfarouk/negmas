"""
Implements all Value-of-Information based elicitation methods.

"""
import copy
import time
from abc import abstractmethod
from collections import defaultdict
from heapq import heapify, heappop, heappush
from warnings import warn
import numpy as np

try:
    from blist import sortedlist
except ImportError:
    warn(
        "blist is not found. VOI based elicitation methods will not work. You can install"
        " blist by running:"
        ""
        ">> pip install blist"
        ""
        "or "
        ""
        ">> pip install negmas[elicitation]",
        ImportWarning,
    )
from typing import Callable, List, Optional, Tuple, Union

from .base import BaseElicitor
from .common import _scale, argmax
from .expectors import Expector, MeanExpector
from .queries import Query, Answer, RangeConstraint
from .strategy import EStrategy
from ..common import MechanismState
from ..modeling import AdaptiveDiscreteAcceptanceModel
from ..outcomes import Outcome
from ..sao import (
    AspirationNegotiator,
    SAONegotiator,
)
from ..utilities import UtilityValue

__all__ = [
    "BaseVOIElicitor",
    "VOIElicitor",
    "VOIFastElicitor",
    "VOINoUncertaintyElicitor",
    "VOIOptimalElicitor",
    "OQA",
]


class BaseVOIElicitor(BaseElicitor):
    """
    Base class for all value of information (VOI) elicitation algorithms

    Args:
        strategy: The elicitation strategy. It is only used if `dynamic_query_set`
                  is set. In that case, the strategy is used to compile
                  the set of all possible queries during construction. If
                  using `dynamic_query_set` pass `None` for the strategy.
        user: The `User` to elicit.
        base_negotiator: The base negotiator used for proposing and responding.
        dynamic_query_set: If given, the user of the object is supposed to
                           manage the `queries` manually and the strategy is
                           not used.
        queries: An optinal list of queries to use.
        adaptive_answer_probabilities: If `True`, answer probabilities will not
                                       be considered equal for all possible
                                       answers. The probability of getting an
                                       answer will be based on the current
                                       estimate of the utility value distribution.
        expector_factory: A `Callable` used to estimate real-valued utilities given
                          a distribution.
        opponent_model_factory: A `Callable` used to construct the opponent model.
        single_elicitation_per_round: If set, a single query is allowed per round.
        continue_eliciting_past_reserved_val: If set, elicition will continue
                                              even if the estimated utility
                                              of an outcome is less than the
                                              reserved value.
        epsilon: A small number used to stop elicitation when the uncertainty
                 in the utility value is within it.
        true_utility_on_zero_cost: If set, the true utility will be elicited
                                   for outcomes if the elicitation cost is zero.
        each_outcome_once: If set, each outcome is to be offered exactly once.
        update_related_queries: If set, queries that are related to one that
                                was asked and answered will get updated based
                                on the answer.
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: "User",
        *,
        dynamic_query_set=False,
        queries=None,
        adaptive_answer_probabilities=True,
        each_outcome_once=False,
        update_related_queries=True,
        **kwargs,
    ) -> None:
        super().__init__(
            strategy=strategy, user=user, **kwargs,
        )
        self.eeu_query = None
        self.query_index_of_outcome = None
        self.dynamic_query_set = dynamic_query_set
        self.adaptive_answer_probabilities = adaptive_answer_probabilities
        self.current_eeu = None
        self.eus = None
        self.queries = queries if queries is not None else []
        self.outcome_in_policy = None
        self.each_outcome_once = each_outcome_once
        self.queries_of_outcome = None
        self.update_related_queries = update_related_queries
        self.total_voi = 0.0

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        queries: Optional[List[Query]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the elicitation process once.

        Remarks:
            - After calling parent, it checks that `dynamic_query_set`, `queries`
              and `strategy` settings are consistent.
            - It then calls, `init_optimal_policy` to initialize the optimal
              policy
            - The set of queries is updated from the strategy if needed and
              a mapping from outcomes to their queries is created if `update_related_queries`
              is set to be used for updating related queries later.
            - It then calls `init_query_eeus` to initialize the EEU of all
              queries.
        """
        super().init_elicitation(ufun=ufun)
        strt_time = time.perf_counter()
        ami = self._ami
        self.eus = np.array([_.mean() for _ in self.utility_distributions()])
        self.offerable_outcomes = ami.outcomes
        if self.dynamic_query_set and not isinstance(self.strategy, EStrategy):
            raise ValueError("The strategy must be a EStrategy for VOIElicitor")
        if not self.dynamic_query_set and self.strategy is not None:
            raise ValueError(
                "If you are not using a dynamic query set, then you cannot pass a strategy. It will not be used"
            )
        if not self.dynamic_query_set and self.queries is None and queries is None:
            raise ValueError(
                "If you are not using a dynamic query set then you must pass a set of queries"
            )
        if self.dynamic_query_set and queries is not None:
            raise ValueError(
                "You cannot pass a set of queries if you use dynamic ask sets"
            )
        if not self.dynamic_query_set and queries is not None:
            self.queries += queries
        self.init_optimal_policy()
        if self.dynamic_query_set:
            self.queries = [
                (outcome, self.strategy.next_query(outcome), 0.0)
                for outcome in ami.outcomes
            ]
        else:
            if self.update_related_queries:
                queries_of_outcome = defaultdict(list)
                for i, (_o, _q, _c) in enumerate(self.queries):
                    queries_of_outcome[_o].append(i)
                self.queries_of_outcome = queries_of_outcome
        self.init_query_eeus()
        self._elicitation_time += time.perf_counter() - strt_time

    def best_offer(self, state: MechanismState) -> Tuple[Optional["Outcome"], float]:
        """
        The best offer and its corresponding utility

        Args:
            state: The mechanism state

        Remarks:
            - It will return (`None`, reserved-value) if the best outcome has
              a utility less than the reserved value.
            - It uses the internal eu_policy heap to find the best outcome.
            - If each-outcome-once is set, the best outcome is popped from the
              heap which prevents it from ever being selected again.

        """
        if self.each_outcome_once:
            # TODO this needs correction. When I opp from the eu_policy, all eeu_query become wrong
            if len(self.eu_policy) < 1:
                self.init_optimal_policy()
            _, outcome_index = self.eu_policy.pop()
        else:
            outcome_index = self.eu_policy[0][1]
        if self.eus[outcome_index] < self.reserved_value:
            return None, self.reserved_value
        return (
            self._ami.outcomes[outcome_index],
            self.expect(
                self.utility_function(self._ami.outcomes[outcome_index]), state=state
            ),
        )

    def can_elicit(self) -> bool:
        """Always can elicit"""
        return True

    def best_offers(self, n: int) -> List[Tuple[Optional["Outcome"], float]]:
        """Returns the best offer repeated n times"""
        return [self.best_offer()] * n

    def before_eliciting(self):
        """Called every round before trying to elicit. Does nothing"""

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        """
        Called whenever the opponent model is updated.

        Args:
            outcomes: The updated outomes. None means all outcomes
            old: The old acceptance probabilities
            new: The new acceptance probabilities

        Remarks:
            It calls `init_optimal_policy` and `init_query_eeus` if any old
            value is not equal to a new value.
        """
        if any(o != n for o, n in zip(old, new)):
            self.init_optimal_policy()
            self.init_query_eeus()

    def update_optimal_policy(
        self, index: int, outcome: "Outcome", oldu: float, newu: float
    ):
        """Updates the optimal policy after a change to the utility value
        of some outcome.

        Args:
            outcome: The outcome whose utiltiy have changed
            oldu: The old utility
            newu: The new utility

        Remarks:
            It just calls `update_optimal_policy`

        """
        if oldu != newu:
            self.init_optimal_policy()

    def elicit_single(self, state: MechanismState):
        """
        Called to conduct a single eliciataion act.

        Args:
            state: The mechanism state

        Remarks:
            - It returns False ending eliciatation if eeu_query is empty or
              `can_elicit` returns False
            - The algorithm outline is as follows:

                1. Pops the top query with its EEU from the heap
                2. elicitation is stopped if the top query is None, the eeu
                   is less than the current EEU, or the EEU after asking will
                   be less than the reserved value.
                3. If dynamic_query_set, the strategy is invoked to get
                   the next query, otherwise, the user is asked the top
                   query and the related queries are updated.
                4. The expected utility is updated base on the answer received
                   from the user and `update_optimal_policy` is called followed
                   by `init_query_eeus`.
        """
        if self.eeu_query is not None and len(self.eeu_query) < 1:
            return False
        if not self.can_elicit():
            return False
        eeu, q = heappop(self.eeu_query)
        if q is None or -eeu <= self.current_eeu:
            return False
        if (not self.continue_eliciting_past_reserved_val) and (
            -eeu - (self.user.cost_of_asking() + self.elicitation_cost)
            < self.reserved_value
        ):
            return False
        outcome, query, cost = self.queries[q]
        if query is None:
            return False
        self.queries[q] = (None, None, None)
        oldu = self.utility_function.distributions[outcome]
        if _scale(oldu) < 1e-7:
            return False
        if self.dynamic_query_set:
            newu, u = self.strategy.apply(user=self.user, outcome=outcome)
        else:
            u = self.user.ask(query)
            newu = u.answer.constraint.marginal(outcome)
            if self.queries_of_outcome is not None:
                if _scale(newu) > 1e-7:
                    newu = newu & oldu
                    newmin, newmax = newu.loc, newu.scale + newu.loc
                    good_queries = []
                    for i, qind in enumerate(self.queries_of_outcome.get(outcome, [])):
                        _o, _q, _c = self.queries[qind]
                        if _q is None:
                            continue
                        answers = _q.answers
                        tokeep = []
                        for j, ans in enumerate(answers):
                            rng = ans.constraint.range
                            if newmin == rng[0] and newmax == rng[1]:
                                continue
                            if newmin <= rng[0] <= newmax or rng[0] <= newmin <= rng[1]:
                                tokeep.append(j)
                        if len(tokeep) < 2:
                            self.queries[i] = None, None, None
                            continue
                        good_queries.append(qind)
                        if len(tokeep) < len(answers):
                            ans = _q.answers
                            self.queries[i].answers = [ans[j] for j in tokeep]
                    self.queries_of_outcome[outcome] = good_queries
                else:
                    for i, _ in enumerate(self.queries_of_outcome.get(outcome, [])):
                        self.queries[i] = None, None, None
                        self.queries_of_outcome[outcome] = []
        self.total_voi += -eeu - self.current_eeu
        outcome_index = self.indices[outcome]
        if _scale(newu) < 1e-7:
            self.utility_function.distributions[outcome] = newu
        else:
            self.utility_function.distributions[outcome] = newu & oldu
        eu = float(newu)
        self.eus[outcome_index] = eu
        self.update_optimal_policy(
            index=outcome_index, outcome=outcome, oldu=float(oldu), newu=eu
        )
        if self.dynamic_query_set:
            o, q, c = outcome, self.strategy.next_query(outcome), 0.0
            if not (o is None or q is None):
                self.queries.append((o, q, c))
                qeeu = self._query_eeu(
                    query,
                    len(self.queries) - 1,
                    outcome,
                    cost,
                    outcome_index,
                    self.eu_policy,
                    self.current_eeu,
                )
                self.add_query((qeeu, len(self.queries) - 1))
        self.init_query_eeus()
        self.elicitation_history.append((query, newu, state.step, self.current_eeu))
        return True

    def init_query_eeus(self) -> None:
        """Updates the heap eeu_query which has records of (-EEU, quesion)"""
        queries = self.queries
        eu_policy, eeu = self.eu_policy, self.current_eeu
        eeu_query = []
        for qindex, current in enumerate(queries):
            outcome, query, cost = current
            if query is None or outcome is None:
                continue
            outcome_index = self.indices[outcome]
            qeeu = self._query_eeu(
                query, qindex, outcome, cost, outcome_index, eu_policy, eeu
            )
            eeu_query.append((qeeu, qindex))
        heapify(eeu_query)
        self.eeu_query = eeu_query

    def utility_on_rejection(
        self, outcome: "Outcome", state: MechanismState
    ) -> UtilityValue:
        raise ValueError("utility_on_rejection should never be called on VOI Elicitors")

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        """Adds a query to the heap of queries

            Args:
                qeeu: A Tuple giving (-EEU, query_index)

            Remarks:
                - Note that the first member of the tuple is **minus** the EEU
                - The sedond member of the tuple is an index of the query in
                  the queries list (not the query itself).
        """
        heappush(self.eeu_query, qeeu)

    @abstractmethod
    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors.

        The optimal plicy should be sorted ascendingly
        on -EU or -EU * Acceptance"""

    @abstractmethod
    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        """
        Find the eeu value associated with this query and return it with
        the query index.

        Args:
            query: The query object
            qindex: The index of the query in the queries list
            outcome: The outcome about which is this query
            cost: The cost of asking the query
            outcome_index: The index of the outcome in the outcomes list
            eu_policy: The expected utility policy
            eeu: The current EEU

        Remarks:
            - Should return - EEU

        """


class VOIElicitor(BaseVOIElicitor):
    """
    The Optimal Querying Agent (OQA) proposed by [Baarslag and Kaisers]_


    .. [Baarslag and Kaisers] Tim Baarslag and Michael Kaisers. 2017. The Value
       of Information in Automated Negotiation: A Decision Model for Eliciting
       User Preferences. In Proceedings of the 16th Conference on Autonomous
       Agents and MultiAgent Systems (AAMAS ’17). International Foundation for
       Autonomous Agents and Multiagent Systems, Richland, SC, 391–400.
       (https://dl.acm.org/doi/10.5555/3091125.3091185)

    """

    def eeu(self, policy: np.ndarray, eus: np.ndarray) -> float:
        """Expected Expected Negotiator for following the policy"""
        p = np.ones((len(policy) + 1))
        m = self.opponent_model.acceptance_probabilities()[policy]
        r = 1 - m
        eup = -eus * m
        p[1:-1] = np.cumprod(r[:-1])
        try:
            result = np.sum(eup * p[:-1])
        except FloatingPointError:
            result = 0.0
            try:
                result = eup[0] * p[0]
                for i in range(1, len(eup)):
                    try:
                        result += eup[0] * p[i]
                    except:
                        break
            except FloatingPointError:
                result = 0.0
        return round(float(result), 6)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        # remaining_steps = ami.remaining_steps if ami.remaining_steps is not None else ami.n_outcomes
        D = n_outcomes
        indices = set(list(range(n_outcomes)))
        p = self.opponent_model.acceptance_probabilities()
        eus = self.eus
        eeus1outcome = eus * p
        best_indx = argmax(eeus1outcome)
        eu_policy = [(-eus[best_indx], best_indx)]
        indices.remove(best_indx)
        D -= 1
        best_eeu = eus[best_indx]
        for _ in range(D):
            if len(indices) < 1:
                break
            candidate_policies = [copy.copy(eu_policy) for _ in indices]
            best_index, best_eeu, eu_policy = None, -10.0, None
            for i, candidate_policy in zip(indices, candidate_policies):
                heappush(candidate_policy, (-eus[i], i))
                # now we have the sorted list of outcomes as a candidate policy
                _policy = np.array([_[1] for _ in candidate_policy])
                _eus = np.array([_[0] for _ in candidate_policy])
                current_eeu = self.eeu(policy=_policy, eus=_eus)
                if (
                    current_eeu > best_eeu
                ):  # all numbers are negative so really that means current_eeu > best_eeu
                    best_eeu, best_index, eu_policy = current_eeu, i, candidate_policy
            if best_index is not None:
                indices.remove(best_index)
        self.outcome_in_policy = {}
        for i, (_, outcome) in enumerate(eu_policy):
            self.outcome_in_policy[outcome] = i
        heapify(eu_policy)
        self.eu_policy, self.current_eeu = eu_policy, best_eeu

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        current_util = self.utility_function(outcome)
        answers = query.answers
        answer_probabilities = query.probs
        answer_eeus = []
        for answer in answers:
            self.init_optimal_policy()
            policy_record_index = self.outcome_in_policy[outcome_index]
            eu_policy = copy.deepcopy(self.eu_policy)
            new_util = (
                -float(answer.constraint.marginal(outcome) & current_util),
                outcome_index,
            )
            eu_policy[policy_record_index] = new_util
            heapify(eu_policy)
            _policy = np.array([_[1] for _ in eu_policy])
            _eus = np.array([_[0] for _ in eu_policy])
            answer_eeus.append(self.eeu(policy=_policy, eus=_eus))
        return cost - sum([a * b for a, b in zip(answer_probabilities, answer_eeus)])


class VOIFastElicitor(BaseVOIElicitor):
    """
    FastVOI algorithm proposed by Mohammad and Nakadai [MN2018]_


    .. [MN2018] Mohammad, Y., & Nakadai, S. (2018, October).
       FastVOI: Efficient utility elicitation during negotiations. In
       International Conference on Principles and Practice of Multi-Agent
       Systems (pp. 560-567). Springer.
       (https://link.springer.com/chapter/10.1007/978-3-030-03098-8_42)
    """

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        eus = -self.eus
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        policy = np.array([_[1] for _ in eu_policy])
        eu = np.array([_[0] for _ in eu_policy])
        p = np.ones((len(policy) + 1))
        ac = self.opponent_model.acceptance_probabilities()[policy]
        eup = -eu * ac
        r = 1 - ac
        p[1:] = np.cumprod(r)
        try:
            s = np.cumsum(eup * p[:-1])
        except FloatingPointError:
            s = np.zeros(len(eup))
            try:
                s[0] = eup[0] * p[0]
            except FloatingPointError:
                s[0] = 0
            for i in range(1, len(eup)):
                try:
                    s[i] = s[i - 1] + eup[0] * p[i]
                except:
                    s[i:] = s[i - 1]
                    break
        self.current_eeu = round(s[-1], 6)
        self.p, self.s = p, s
        self.eu_policy = sortedlist(eu_policy)
        self.outcome_in_policy = {}
        for j, pp in enumerate(self.eu_policy):
            self.outcome_in_policy[pp[1]] = pp

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        answers = query.answers
        answer_probabilities = query.probs
        answer_eeus = []
        current_util = self.utility_function(outcome)
        old_util = self.outcome_in_policy[outcome_index]
        old_indx = eu_policy.index(old_util)
        eu_policy.remove(old_util)
        for answer in answers:
            reeu = self.current_eeu
            a = self.opponent_model.probability_of_acceptance(outcome)
            eu = float(answer.constraint.marginal(outcome) & current_util)
            if old_util[0] != -eu:
                new_util = (-eu, outcome_index)
                p, s = self.p, self.s
                eu_policy.add(new_util)
                new_indx = eu_policy.index(new_util)
                moved_back = new_indx > old_indx or new_indx == old_indx
                u_old, u_new = -old_util[0], eu
                try:
                    if new_indx == old_indx:
                        reeu = eeu - a * u_old * p[new_indx] + a * u_new * p[new_indx]
                    else:
                        s_before_src = s[old_indx - 1] if old_indx > 0 else 0.0
                        if moved_back:
                            p_after = p[new_indx + 1]
                            if a < 1.0 - 1e-6:
                                reeu = (
                                    s_before_src
                                    + (s[new_indx] - s[old_indx]) / (1 - a)
                                    + a * u_new * p_after / (1 - a)
                                    + eeu
                                    - s[new_indx]
                                )
                            else:
                                reeu = s_before_src + eeu - s[new_indx]
                        else:
                            s_before_dst = s[new_indx - 1] if new_indx > 0 else 0.0
                            if a < 1.0 - 1e-6:
                                reeu = (
                                    s_before_dst
                                    + a * u_new * p[new_indx]
                                    + (s_before_src - s_before_dst) * (1 - a)
                                    + eeu
                                    - s[old_indx]
                                )
                            else:
                                reeu = (
                                    s_before_dst
                                    + a * u_new * p[new_indx]
                                    + eeu
                                    - s[old_indx]
                                )
                except FloatingPointError:
                    pass

                self.eu_policy.remove(new_util)
            answer_eeus.append(reeu)
        self.eu_policy.add(old_util)
        qeeu = cost - sum([a * b for a, b in zip(answer_probabilities, answer_eeus)])
        return qeeu


class VOINoUncertaintyElicitor(BaseVOIElicitor):
    """A dummy VOI Elicitation Agent. It simply assumes no uncertainty in
    own utility function"""

    def eeu(self, policy: np.ndarray, eup: np.ndarray) -> float:
        """Expected Expected Negotiator for following the policy"""
        p = np.ones((len(policy) + 1))
        r = 1 - self.opponent_model.acceptance_probabilities()[policy]
        p[1:] = np.cumprod(r)
        try:
            result = np.sum(eup * p[:-1])
        except FloatingPointError:
            result = 0.0
            try:
                result = eup[0] * p[0]
                for i in range(1, len(eup)):
                    try:
                        result += eup[0] * p[i]
                    except:
                        break
            except FloatingPointError:
                result[0] = 0.0
        return float(result)  # it was - for a reason I do not undestand (2018.11.16)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        p = self.opponent_model.acceptance_probabilities()
        eus = -self.eus * p
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        self.current_eeu = self.eeu(
            policy=np.array([_[1] for _ in eu_policy]),
            eup=np.array([_[0] for _ in eu_policy]),
        )
        self.eu_policy = eu_policy
        self.outcome_in_policy = {}
        # for j, (_, indx) in enumerate(eu_policy):
        for _, indx in eu_policy:
            self.outcome_in_policy[indx] = (_, indx)

    def init_query_eeus(self) -> None:
        pass

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        pass

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        return -1.0

    def elicit_single(self, state: MechanismState):
        return False


class VOIOptimalElicitor(BaseElicitor):
    """
    Optimal VOI elicitor proposed by [Mohammad and Nakadai]_

    This algorithm restricts the type of queries that can be asked but does
    not require the user to set the set of queries apriori and can use
    unconuntable sets of queries of the form: "Is u(o) > x?"


    .. [Mohammad and Nakadai] Yasser Mohammad and Shinji Nakadai. 2019. Optimal
       Value of Information Based Elicitation During Negotiation. In Proceedings
       of the 18th International Conference on Autonomous Agents and MultiAgent
       Systems (AAMAS ’19). International Foundation for Autonomous Agents and
       Multiagent Systems, Richland, SC, 242–250.
       (https://dl.acm.org/doi/10.5555/3306127.3331699)

    """

    def __init__(
        self,
        user: "User",
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        adaptive_answer_probabilities=True,
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        resolution=0.025,
        true_utility_on_zero_cost=False,
        each_outcome_once=False,
        update_related_queries=True,
        prune=True,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
    ) -> None:
        super().__init__(
            strategy=None,
            user=user,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        # todo confirm that I need this. aspiration mixin. I think I do not.
        # self.aspiration_init(max_aspiration=1.0, aspiration_type="boulware")
        self.eu_policy = None
        self.eeu_query = None
        self.query_index_of_outcome = None
        self.adaptive_answer_probabilities = adaptive_answer_probabilities
        self.current_eeu = None
        self.eus = None
        self.outcome_in_policy = None
        self.each_outcome_once = each_outcome_once
        self.queries_of_outcome = None
        self.queries = None
        self.update_related_queries = update_related_queries
        self.total_voi = 0.0
        self.resolution = resolution
        self.prune = prune

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        queries: Optional[List[Query]] = None,
    ) -> None:
        super().init_elicitation(ufun=ufun)
        if queries is not None:
            raise ValueError(
                f"self.__class__.__name__ does not allow the user to specify queries"
            )
        strt_time = time.perf_counter()
        ami = self._ami
        self.eus = np.array([_.mean() for _ in self.utility_distributions()])
        self.offerable_outcomes = ami.outcomes
        self.init_optimal_policy()
        self.init_query_eeus()
        self._elicitation_time += time.perf_counter() - strt_time

    def best_offer(self, state: MechanismState) -> Tuple[Optional["Outcome"], float]:
        """Maximum Expected Utility at a given aspiration level (alpha)

        Args:
            state:
        """
        if self.each_outcome_once:
            # todo this needs correction. When I opp from the eu_policy, all eeu_query become wrong
            if len(self.eu_policy) < 1:
                self.init_optimal_policy()
            _, outcome_index = self.eu_policy.pop()
        else:
            outcome_index = self.eu_policy[0][1]
        if self.eus[outcome_index] < self.reserved_value:
            return None, self.reserved_value
        return (
            self._ami.outcomes[outcome_index],
            self.expect(
                self.utility_function(self._ami.outcomes[outcome_index]), state=state
            ),
        )

    def can_elicit(self) -> bool:
        return True

    def before_eliciting(self):
        pass

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        if any(o != n for o, n in zip(old, new)):
            self.init_optimal_policy()
            self.init_query_eeus()

    def update_optimal_policy(
        self, index: int, outcome: "Outcome", oldu: float, newu: float
    ):
        """Updates the optimal policy after a change happens to some utility"""
        if oldu != newu:
            self.init_optimal_policy()

    def elicit_single(self, state: MechanismState):
        if self.eeu_query is not None and len(self.eeu_query) < 1:
            return False
        if not self.can_elicit():
            return False
        eeu, q = heappop(self.eeu_query)
        if q is None or -eeu <= self.current_eeu:
            return False
        if (not self.continue_eliciting_past_reserved_val) and (
            -eeu - (self.user.cost_of_asking() + self.elicitation_cost)
            < self.reserved_value
        ):
            return False
        outcome, query, _ = self.queries[q]
        if query is None:
            return False
        self.queries[q] = (None, None, None)
        oldu = self.utility_function.distributions[outcome]
        if _scale(oldu) < 1e-7:
            return False
        u = self.user.ask(query)
        newu = u.answer.constraint.marginal(outcome)
        if self.queries_of_outcome is not None:
            if _scale(newu) > 1e-7:
                newu = newu & oldu
                newmin, newmax = newu.loc, newu.scale + newu.loc
                good_queries = []
                for i, qind in enumerate(self.queries_of_outcome.get(outcome, [])):
                    _o, _q, _c = self.queries[qind]
                    if _q is None:
                        continue
                    answers = _q.answers
                    tokeep = []
                    for j, ans in enumerate(answers):
                        rng = ans.constraint.range
                        if newmin == rng[0] and newmax == rng[1]:
                            continue
                        if newmin <= rng[0] <= newmax or rng[0] <= newmin <= rng[1]:
                            tokeep.append(j)
                    if len(tokeep) < 2:
                        self.queries[i] = None, None, None
                        continue
                    good_queries.append(qind)
                    if len(tokeep) < len(answers):
                        ans = _q.answers
                        self.queries[i].answers = [ans[j] for j in tokeep]
                self.queries_of_outcome[outcome] = good_queries
            else:
                for i, _ in enumerate(self.queries_of_outcome.get(outcome, [])):
                    self.queries[i] = None, None, None
                    self.queries_of_outcome[outcome] = []
        self.total_voi += -eeu - self.current_eeu
        outcome_index = self.indices[outcome]
        if _scale(newu) < 1e-7:
            self.utility_function.distributions[outcome] = newu
        else:
            self.utility_function.distributions[outcome] = newu & oldu
        eu = float(newu)
        self.eus[outcome_index] = eu
        self.update_optimal_policy(
            index=outcome_index, outcome=outcome, oldu=float(oldu), newu=eu
        )
        self._update_query_eeus(
            k=outcome_index,
            outcome=outcome,
            s=self.s,
            p=self.p,
            n=self._ami.n_outcomes,
            eeu=self.current_eeu,
            eus=[-_[0] for _ in self.eu_policy],
        )
        self.elicitation_history.append((query, newu, state.step, self.current_eeu))
        return True

    def _update_query_eeus(self, k: int, outcome: "Outcome", s, p, n, eeu, eus):
        """Updates the best query for a single outcome"""
        this_outcome_solutions = []
        m = self.opponent_model.probability_of_acceptance(outcome)
        m1 = 1.0 - m
        m2 = m / m1 if m1 > 1e-6 else 0.0
        uk = self.utility_function.distributions[outcome]
        beta, alpha = uk.scale + uk.loc, uk.loc
        delta = beta - alpha
        if abs(delta) < max(self.resolution, 1e-6):
            return
        sk1, sk, pk = s[k - 1] if k > 0 else 0.0, s[k], p[k]
        for jp in range(k + 1):
            sjp1, sjp = s[jp - 1] if jp > 0 else 0.0, s[jp]
            if (
                beta < eus[jp]
            ):  # ignore cases where it is impossible to go to this low j
                continue
            for jm in range(k, n):
                if jp == k and jm == k:
                    continue
                if (
                    alpha > eus[jp]
                ):  # ignore cases where it is impossible to go to this large j
                    continue
                try:
                    _, sjm = s[jm - 1] if jm > 0 else 0.0, s[jm]
                    if m1 > 1e-6:
                        y = ((sk1 - sk) + m * (sjm - sk1)) / m1
                    else:
                        y = 0.0
                    z = sk1 - sk + m * (sjp1 - sk1)
                    pjm1, pjp, pjm = p[jm + 1], p[jp], p[jm]
                    if jp < k < jm:  # Problem 1
                        a = (m2 * pjm1 - m * pjp) / (2 * delta)
                        b = (y - z) / delta
                        c = (
                            2 * z * beta
                            + m * pjp * beta * beta
                            - 2 * y * alpha
                            - m2 * pjm1 * alpha * alpha
                        ) / (2 * delta)
                    elif jp < k == jm:  # Problem 2
                        a = m * (pk - pjp) / (2 * delta)
                        b = -(2 * z + m * pk * (beta + alpha)) / (2 * delta)
                        c = (
                            beta
                            * (2 * z + m * pjp * beta + m * pk * alpha)
                            / (2 * delta)
                        )
                    else:  # Problem 3
                        a = (m2 * pjm1 - m * pk) / (2 * delta)
                        b = (2 * y + m * pk * (beta + alpha)) / (2 * delta)
                        c = (
                            -alpha
                            * (2 * y + m * pk * beta + m2 * pjm1 * alpha)
                            / (2 * delta)
                        )
                    if abs(a) < 1e-6:
                        continue
                    x = -b / (2 * a)
                    voi = c - a * x * x
                except FloatingPointError:
                    continue
                if x < alpha or x > beta or voi < self.user.cost_of_asking():
                    if self.prune:
                        break
                    continue  # ignore cases when the optimum is at the limit
                q = Query(
                    answers=[
                        Answer(
                            outcomes=[outcome],
                            constraint=RangeConstraint((x, beta)),
                            name="yes",
                        ),
                        Answer(
                            outcomes=[outcome],
                            constraint=RangeConstraint((alpha, x)),
                            name="no",
                        ),
                    ],
                    probs=[(beta - x) / delta, (x - alpha) / delta],
                    name=f"{outcome}>{x}",
                )
                this_outcome_solutions.append((voi, q))
            if self.prune and len(this_outcome_solutions) > 0:
                break
        if len(this_outcome_solutions) > 0:
            voi, q = max(this_outcome_solutions, key=lambda x: x[0])
            self.queries.append((outcome, q, self.user.cost_of_asking()))
            qindx = len(self.queries) - 1
            heappush(self.eeu_query, (-voi - eeu, qindx))
            self.queries_of_outcome[outcome] = [qindx]

    def init_query_eeus(self) -> None:
        """Updates the heap eeu_query which has records of (-EEU, quesion)"""
        # todo code for creating the optimal queries
        outcomes = self._ami.outcomes
        policy = [_[1] for _ in self.eu_policy]
        eus = [-_[0] for _ in self.eu_policy]
        n = len(outcomes)
        p, s = self.p, self.s
        eeu = self.current_eeu
        self.queries_of_outcome = dict()
        self.queries = []
        self.eeu_query = []
        heapify(self.eeu_query)
        for k, outcome_indx in enumerate(policy):
            self._update_query_eeus(
                k=k, outcome=outcomes[outcome_indx], s=s, p=p, n=n, eeu=eeu, eus=eus
            )

    def utility_on_rejection(
        self, outcome: "Outcome", state: MechanismState
    ) -> UtilityValue:
        raise ValueError("utility_on_rejection should never be called on VOI Elicitors")

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        heappush(self.eeu_query, qeeu)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        eus = -self.eus
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        policy = np.array([_[1] for _ in eu_policy])
        eu = np.array([_[0] for _ in eu_policy])
        p = np.ones((len(policy) + 1))
        ac = self.opponent_model.acceptance_probabilities()[policy]
        eup = -eu * ac
        r = 1 - ac
        p[1:] = np.cumprod(r)
        try:
            s = np.cumsum(eup * p[:-1])
        except FloatingPointError:
            s = np.zeros(len(eup))
            try:
                s[0] = eup[0] * p[0]
            except FloatingPointError:
                s[0] = 0
            for i in range(1, len(eup)):
                try:
                    s[i] = s[i - 1] + eup[0] * p[i]
                except:
                    s[i:] = s[i - 1]
                    break
        self.current_eeu = round(s[-1], 6)
        self.p, self.s = p, s
        self.eu_policy = sortedlist(eu_policy)
        self.outcome_in_policy = {}
        for j, pp in enumerate(self.eu_policy):
            self.outcome_in_policy[pp[1]] = pp


OQA = VOIElicitor
"""An Alias for `VOIElicitor`"""
