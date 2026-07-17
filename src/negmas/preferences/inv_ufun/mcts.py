"""Utility function inversion via MCTS (Monte Carlo Tree Search).

MCTS builds a search tree over the issue assignment space and uses UCB1
selection, random rollouts, and backpropagation to find outcomes with
utilities in a requested range.  The algorithm works with any ufun whose
outcome space has enumerable issues and values.

Reference
---------
Buron, C. L., Guessoum, Z., & Ductor, S. (2014).
*MCTS-based automated negotiation agent.*
In PRIMA 2014 (pp. 436-443).

Description from:
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.  Table 2.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any


from negmas.outcomes import Outcome

from ..protocols import InverseUFun
from ._common import EPS, _norm_to_raw, _raw_to_norm, _resolve_rng

if TYPE_CHECKING:
    from ..base_ufun import BaseUtilityFunction

__all__ = ["MCTSInverseUtilityFunction"]


class _MCTSNode:
    """A node in the MCTS tree.

    Each node corresponds to a partial outcome where issues 0..depth-1 have been
    assigned specific values.  Depth == n_issues means the node is a leaf (full
    outcome).
    """

    __slots__ = ("partial", "depth", "visits", "total_reward", "children", "untried")

    def __init__(self, partial: tuple, depth: int, untried: list[Any]) -> None:
        self.partial: tuple = partial
        self.depth: int = depth
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.children: dict[Any, _MCTSNode] = {}
        self.untried: list[Any] = untried  # values not yet expanded at this depth


class MCTSInverseUtilityFunction(InverseUFun):
    """Utility function inverter using Monte Carlo Tree Search (MCTS).

    MCTS builds a search tree over the issue-value assignment space using
    UCB1 selection, random rollouts, and backpropagation.  The algorithm
    works with **any** ufun whose outcome space has enumerable issues and
    values (unlike BIDS/Attribute-Planning which require additive ufuns).

    This is a **clamping** inverter (see module docs). ``worst_in``/``best_in``
    expand the range upward and fall back to the nearest boundary outcome
    (best if the range is in the upper half of the utility range, worst if in
    the lower half) rather than returning ``None``.

    Args:
        ufun: The utility function to invert.
        n_simulations: Number of MCTS iterations per query.  Default: 200.
        c_ucb: UCB1 exploration constant.  Default: √2 ≈ 1.41421356.
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        n_simulations: int = 200,
        c_ucb: float = 1.41421356,
        max_ranking_outcomes: int = 20_000,
    ) -> None:
        self._ufun = ufun
        self._n_simulations = n_simulations
        self._c_ucb = c_ucb
        self._max_ranking_outcomes = max_ranking_outcomes
        self._initialized = False

        # filled by init()
        self._issues: list[Any] = []
        self._val_list: list[list[Any]] = []
        self._u_min: float = 0.0
        self._u_max: float = 1.0
        self._ordered_outcomes: list[tuple[float, Outcome]] = []

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def ufun(self) -> BaseUtilityFunction:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Enumerate issues and values from the ufun's outcome space.

        Raises:
            ValueError: if the ufun has no outcome space or no issues.
        """
        os = self._ufun.outcome_space
        issues = getattr(os, "issues", None) if os is not None else None
        if not issues:
            raise ValueError(
                "MCTSInverseUtilityFunction requires a ufun with an outcome space "
                "that has enumerable issues."
            )
        assert os is not None

        self._issues = list(issues)
        self._val_list = [list(issue.all) for issue in self._issues]

        u_min, u_max = self._ufun.minmax()
        self._u_min = float(u_min)
        self._u_max = float(u_max)
        self._ordered_outcomes = []

        # Build a best-first ranking when the space is small enough.
        cardinality = getattr(os, "cardinality", None)
        if (
            isinstance(cardinality, (int, float))
            and cardinality > 0
            and cardinality <= self._max_ranking_outcomes
        ):
            try:
                outs = list(
                    os.enumerate_or_sample(max_cardinality=self._max_ranking_outcomes)
                )
                self._ordered_outcomes = sorted(
                    ((float(self._ufun(o)), o) for o in outs), key=lambda x: x[0]
                )
            except Exception:
                self._ordered_outcomes = []
        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MCTSInverseUtilityFunction.init() has not been called.")

    def _norm_to_raw(self, t: float) -> float:
        return _norm_to_raw(t, self._u_min, self._u_max)

    def _raw_to_norm(self, u: float) -> float:
        return _raw_to_norm(u, self._u_min, self._u_max)

    def _resolve_rng(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> tuple[float, float] | None:
        return _resolve_rng(rng, normalized, self._u_min, self._u_max)

    def _new_root(self) -> _MCTSNode:
        return _MCTSNode((), 0, list(range(len(self._val_list[0]))))

    def _rollout(self, node: _MCTSNode) -> tuple:
        """Complete a random outcome from the node's partial assignment."""
        partial = list(node.partial)
        for i in range(node.depth, len(self._val_list)):
            partial.append(random.choice(self._val_list[i]))
        return tuple(partial)

    def _select_child(self, node: _MCTSNode) -> _MCTSNode:
        """UCB1 selection among fully-expanded children."""
        log_parent = math.log(node.visits) if node.visits > 0 else 0.0
        best_child: _MCTSNode | None = None
        best_score = float("-inf")
        for child in node.children.values():
            if child.visits == 0:
                return child  # unvisited child: prioritise immediately
            score = (child.total_reward / child.visits) + self._c_ucb * math.sqrt(
                log_parent / child.visits
            )
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        """Pick an untried value and expand a new child node."""
        val_idx = node.untried.pop(random.randrange(len(node.untried)))
        depth = node.depth + 1
        new_partial = node.partial + (self._val_list[node.depth][val_idx],)
        if depth < len(self._val_list):
            untried = list(range(len(self._val_list[depth])))
        else:
            untried = []
        child = _MCTSNode(new_partial, depth, untried)
        node.children[val_idx] = child
        return child

    def _run_mcts(
        self, reward_fn: Any, n_simulations: int
    ) -> tuple[Outcome | None, float]:
        """Run *n_simulations* MCTS iterations using *reward_fn*.

        Returns:
            (best_outcome, best_reward) — the terminal outcome with the
            highest reward found, or (None, -inf) if no terminal was reached.
        """
        root = self._new_root()
        best_outcome: Outcome | None = None
        best_reward = float("-inf")
        n_issues = len(self._val_list)

        for _ in range(n_simulations):
            # --- Selection ---
            node = root
            path: list[_MCTSNode] = [node]
            while node.depth < n_issues and not node.untried and node.children:
                node = self._select_child(node)
                path.append(node)

            # --- Expansion ---
            if node.depth < n_issues and node.untried:
                node = self._expand(node)
                path.append(node)

            # --- Simulation (rollout) ---
            if node.depth == n_issues:
                outcome: Outcome = node.partial
            else:
                outcome = self._rollout(node)

            reward = reward_fn(outcome)

            # Track best outcome seen
            if reward > best_reward:
                best_reward = reward
                best_outcome = outcome

            # --- Backpropagation ---
            for n in path:
                n.visits += 1
                n.total_reward += reward

        return best_outcome, best_reward

    # ------------------------------------------------------------------
    # InverseUFun protocol
    # ------------------------------------------------------------------

    def some(
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
    ) -> list[Outcome]:
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return []
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        k = n if n is not None else 10

        def reward_fn(o: Outcome) -> float:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            return 1.0 if mn_raw - EPS <= u <= mx_raw + EPS else -1.0

        seen: set[Outcome] = set()
        result: list[Outcome] = []
        for _ in range(k):
            o, r = self._run_mcts(reward_fn, self._n_simulations)
            if o is not None and r > 0 and o not in seen:
                seen.add(o)
                result.append(o)
        return result

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw

        def reward_fn(o: Outcome) -> float:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            return 1.0 if mn_raw - EPS <= u <= mx_raw + EPS else -1.0

        o, r = self._run_mcts(reward_fn, self._n_simulations)
        if o is not None and r > 0:
            return o

        if fallback_to_higher:
            # Try a random outcome and check if it's above mn_raw
            for _ in range(20):
                candidate = tuple(random.choice(vals) for vals in self._val_list)
                u = float(self._ufun(candidate))  # type: ignore[arg-type]
                if u >= mn_raw - EPS:
                    return candidate

        if fallback_to_best:
            return self.best()

        return None

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """Return the outcome with the **highest** utility in *rng* via MCTS.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], max]`` and the search is
          retried.
        * If still nothing is found and ``fallback_to_best`` is ``True``, the
          best outcome overall is returned.
        * Otherwise ``None`` is returned.
        """
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw

        def reward_fn(o: Outcome) -> float:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            return u if mn_raw - EPS <= u <= mx_raw + EPS else (self._u_min - 1.0)

        o, r = self._run_mcts(reward_fn, self._n_simulations)
        if o is not None and r > self._u_min - 1.0 + EPS:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u <= mx_raw + EPS:
                return o
        if fallback_to_higher and (not normalized or mx_raw < 1 - EPS):
            new_rng = (mn_raw, float(self.max()))
            return self.best_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_best=fallback_to_best,
            )
        if fallback_to_best:
            return self.best()
        return None

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_worst: bool = True,
    ) -> Outcome | None:
        """Return the outcome with the **lowest** utility in *rng* via MCTS.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], max]`` and the search is
          retried.
        * If still nothing is found and ``fallback_to_worst`` is ``True``, the
          **nearest boundary** outcome is returned: the best outcome if the
          range lies above the maximum utility, the worst outcome if it lies
          below the minimum.
        * Otherwise ``None`` is returned.
        """
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw

        def reward_fn(o: Outcome) -> float:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            return -u if mn_raw - EPS <= u <= mx_raw + EPS else (self._u_min - 1.0)

        o, r = self._run_mcts(reward_fn, self._n_simulations)
        if o is not None and r > self._u_min - 1.0 + EPS:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u <= mx_raw + EPS:
                return o
        if fallback_to_higher and (not normalized or mx_raw < 1 - EPS):
            new_rng = (mn_raw, float(self.max()))
            return self.worst_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_worst=fallback_to_worst,
            )
        if fallback_to_worst:
            # Return the nearest boundary outcome. If the requested range is
            # in the upper half of the utility range, the best outcome is
            # closer; otherwise the worst is closer. This keeps the fallback
            # near the requested range.
            u_min, u_max = self.minmax()
            mid = (u_min + u_max) / 2.0
            if mn_raw >= mid:
                return self.best()
            return self.worst()
        return None

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        self._check_initialized()
        mn_raw = self._norm_to_raw(rng[0])
        mx_raw = self._norm_to_raw(rng[1])
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw

        def reward_fn(o: Outcome) -> float:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            return 1.0 if mn_raw - EPS <= u <= mx_raw + EPS else -1.0

        seen: set[Outcome] = set()
        result: list[Outcome] = []
        for _ in range(10):
            o, r = self._run_mcts(reward_fn, self._n_simulations)
            if o is not None and r > 0 and o not in seen:
                seen.add(o)
                result.append(o)
        return result

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        self._check_initialized()
        os = self._ufun.outcome_space
        total = os.cardinality if os is not None else None
        if total is None or total == 0:
            return []
        lo_frac = 1.0 - rng[1] / total
        hi_frac = 1.0 - rng[0] / total
        lo_frac = max(0.0, min(1.0, lo_frac))
        hi_frac = max(0.0, min(1.0, hi_frac))
        return self.within_fractions((lo_frac, hi_frac))

    def min(self) -> float:
        return float(self._ufun.minmax()[0])

    def max(self) -> float:
        return float(self._ufun.minmax()[1])

    def worst(self) -> Outcome:
        return self._ufun.extreme_outcomes()[0]

    def best(self) -> Outcome:
        return self._ufun.extreme_outcomes()[1]

    def minmax(self) -> tuple[float, float]:
        return self._ufun.minmax()  # type: ignore[return-value]

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
        return self._ufun.extreme_outcomes()  # type: ignore[return-value]

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        return self.one_in(rng, normalized)

    @property
    def outcomes(self) -> list[Outcome]:
        """Returns outcomes sorted worst→best when ranking is available."""
        return [o for _, o in self._ordered_outcomes]

    def outcome_at(self, indx: int) -> Outcome | None:
        """Returns outcome at rank index (0 = best) when available."""
        if indx < 0 or not self._ordered_outcomes:
            return None
        j = len(self._ordered_outcomes) - 1 - indx
        if j < 0 or j >= len(self._ordered_outcomes):
            return None
        return self._ordered_outcomes[j][1]

    def utility_at(self, indx: int) -> float | None:
        """Returns utility at rank index (0 = best) when available."""
        if indx < 0 or not self._ordered_outcomes:
            return None
        j = len(self._ordered_outcomes) - 1 - indx
        if j < 0 or j >= len(self._ordered_outcomes):
            return None
        return self._ordered_outcomes[j][0]
