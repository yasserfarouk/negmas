from __future__ import annotations

"""Implements GA-based negotiation mechanisms"""

import copy
import random

from attrs import define, field

from negmas.common import Action

from .mechanisms import Mechanism, MechanismState, MechanismStepResult
from .outcomes import Outcome


@define
class GAState(MechanismState):
    """Defines extra values to keep in the mechanism state. This is accessible to all negotiators"""

    dominant_outcomes: list[Outcome] = field(default=list)


class GAMechanism(Mechanism):
    """Naive GA-based mechanism that assume multi-issue discrete domains.

    Args:
        *args: positional arguments to be passed to the base Mechanism
        **kwargs: keyword arguments to be passed to the base Mechanism
        n_population: The number of outcomes for each generation
        mutate_rate: The rate of mutation
    """

    def generate(self, n: int) -> list[Outcome]:
        return self.random_outcomes(n)

    def __init__(
        self, *args, n_population: int = 100, mutate_rate: float = 0.1, **kwargs
    ):
        kwargs["initial_state"] = GAState()
        super().__init__(*args, **kwargs)
        self._current_state: GAState

        self.n_population = n_population

        self.mutate_rate = mutate_rate

        self.population = self.generate(self.n_population)

        self.dominant_outcomes = self.population[:]
        self._current_state.dominant_outcomes = self.dominant_outcomes  # type: ignore

        self.ranks = {}

    def crossover(self, outcome1: Outcome, outcome2: Outcome) -> Outcome:
        """Uniform crossover"""
        outcome = list(copy.deepcopy(outcome1))
        for i in range(len(self.issues)):
            if bool(random.getrandbits(1)):
                outcome[i] = outcome2[i]

        return tuple(outcome)

    def mutate(self, outcome: Outcome) -> Outcome:
        """Uniform crossover with random outcome"""
        return self.crossover(outcome, self.generate(1)[0])

    def select(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Select Pareto optimal outcomes"""
        return self.dominant_outcomes

    def next_generation(self, parents: list[Outcome]) -> list[Outcome]:
        """Generate the next generation from parents"""
        self.population = parents[:]
        for _ in range(self.n_population - len(self.dominant_outcomes)):
            if random.random() > self.mutate_rate and len(self.dominant_outcomes) >= 2:
                self.population.append(
                    self.crossover(*random.sample(self.dominant_outcomes, 2))
                )
            else:
                self.population.append(
                    self.mutate(random.choice(self.dominant_outcomes))
                )
        return self.population

    def update_ranks(self):
        self.ranks.clear()
        outcomes = {}
        for outcome in self.population:
            outcomes[str(outcome)] = outcome  # merge duplicates
            self.ranks[str(outcome)] = {}

        # get ranking from agents
        for neg in self.negotiators:
            sorted_outcomes = list(outcomes.values())
            neg.sort(sorted_outcomes, descending=True)
            for i, outcome in enumerate(sorted_outcomes):
                self.ranks[str(outcome)][neg] = i

    def update_dominant_outcomes(self):
        """Return dominant outcomes of population"""
        self.dominant_outcomes.clear()
        outcomes = {}
        for outcome in self.population:
            outcomes[str(outcome)] = outcome  # merge duplicates

        # get dominant outcomes
        # naive approach (should be improved)
        for target in outcomes.keys():
            for outcome in outcomes.keys():
                if target == outcome:
                    continue
                for neg in self.negotiators:
                    if self.ranks[target][neg] < self.ranks[outcome][neg]:
                        break
                else:
                    break
            else:
                self.dominant_outcomes.append(outcomes[target])
        self._current_state.dominant_outcomes = self.dominant_outcomes  # type: ignore

    def __call__(
        self, state: GAState, action: Action | None = None
    ) -> MechanismStepResult:
        self.update_ranks()
        self.update_dominant_outcomes()
        self.next_generation(self.select(self.population))
        return MechanismStepResult(state=state)
