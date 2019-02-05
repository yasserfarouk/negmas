import itertools
import math
import multiprocessing
import random
from collections import defaultdict
from typing import Optional, List, Any, Tuple, Union, Iterable, Iterator
from typing import TYPE_CHECKING, Dict

from negmas.helpers import instantiate, get_class
from negmas.situated import World, Agent

if TYPE_CHECKING:
    pass

__all__ = [
    'WorldGenerator',
    'Tournament',
]


class WorldGenerator:
    """Generates worlds for a tournament"""

    def __init__(self, world_class: Union[str, World], base_config: Union[str, Dict[str, Any]]
                 , variations: Optional[Dict[str, List[Any]]] = None):
        """

        Args:
            world_class: Type of the world (class name)
            base_config:
            variations:
        """
        if isinstance(world_class, str):
            world_class_t: World = get_class(class_name=world_class, scope=globals())
        else:
            world_class_t = world_class
        self.world_class = world_class_t
        self.config = world_class_t.read_config(config=base_config)

        if variations is None:
            variations = dict()
        self.variations = variations
        n = 1
        for vals in self.variations.values():
            n *= len(vals)
        self.__len = n
        self.config_variations = zip(variations.keys(), variations.values())

    def __len__(self):
        """The number of worlds that can be generated"""
        return self.__len

    def __iter__(self):
        config = self.config.copy()
        for pairs in itertools.product(self.config_variations):
            for n, v in pairs:
                names = n.split('/')
                if len(names) == 1:
                    config[n] = v
                else:
                    vparam = config
                    for name in names[:-1]:
                        vparam = vparam[name]
                    vparam[names[-1]] = v
            yield self.world_class.from_config(config=config, ignore_children=False, try_parsing_children=True
                                               , scope=globals())


def iter_sample_fast(iterator: Iterator, n: int) -> Iterable:
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(n):
            results.append(next(iterator))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, n):
        r = random.randint(0, i)
        if r < n:
            results[r] = v  # at a decreasing rate, replace random items
    return results


class Tournament:
    """Controls a full tournament"""

    def __init__(self, world_class: Union[str, World], base_config: Union[str, Dict[str, Any]]
                 , variations: Optional[Dict[str, List[Any]]]
                 , competitors: Tuple[str]
                 , builtins: Tuple[str]
                 , tournament_type: str = 'one_vs_all'  # other options: one_vs_one, one_vs_builtin
                 , n_runs_per_variation: int = 1
                 , n_max_runs: Optional[int] = None
                 , n_concurrent_runs: Optional[int] = None
                 ):
        """

        Args:
            world_class:
            base_config:
            variations:
            competitors:
            builtins:
            tournament_type:
            n_runs_per_variation:
            n_max_runs:
            n_concurrent_runs:
        """
        self.generator = WorldGenerator(world_class=world_class, base_config=base_config, variations=variations)
        self.competitors = competitors
        self.builtins = builtins
        self.type = tournament_type
        self.n_per_variation = n_runs_per_variation
        self.n_max = n_max_runs
        self.n_concurrent = n_concurrent_runs if n_concurrent_runs is not None else multiprocessing.cpu_count()
        n = 1
        m = len(self.competitors)
        if self.type in ('one_vs_all', 'ova'):
            n *= math.factorial(m)
        elif self.type in ('one_vs_one', 'ovo'):
            n *= m * (m - 1)
        elif self.type in ('one_vs_builtins', 'ovb'):
            n *= m
        n_per_assignment = len(self.generator) * self.n_per_variation
        n_total = n * n_per_assignment
        # can run all possibilities
        self.__len = n_total
        if self.type in ('one_vs_all', 'ova'):
            self.assignments = itertools.permutations(self.competitors)
        elif self.type in ('one_vs_one', 'ovo'):
            self.assignments = ((c1, c2) for c1 in competitors for c2 in competitors if c1 != c2)
        elif self.type in ('one_vs_builtins', 'ovb'):
            self.assignments = ((c,) for c in competitors)
        if self.n_max is not None and n_total > self.n_max:
            self.assignments = (_ for _ in iter_sample_fast(self.assignments, self.n_max // n_per_assignment))
        self.runs = ((a, w, i) for i in range(self.n_per_variation) for w in self.generator for a in self.assignments)
        self.scores: Dict[str, List[float]] = defaultdict(list)
        for c in self.competitors:
            self.scores[c] = []

    def __len__(self) -> int:
        return self.__len

    def assign(self, world: World, builtins: Tuple[Agent], competitors: Tuple[Agent]) -> World:
        return world

    def evaluate(self, world: World, agent_id: str) -> float:
        return 0.0

    def __iter__(self):
        for i, (competitors, world, _) in enumerate(self.runs):
            agents = tuple(instantiate(c) for c in competitors)
            agent_ids = dict(zip(competitors, (a.id for a in agents)))
            world = self.assign(world=world, builtins=tuple(instantiate(b) for b in self.builtins)
                                , competitors=agents)
            world.run()
            for c, aid in agent_ids:
                self.scores[c].append(self.evaluate(world=world, agent_id=aid))
            yield

    def run(self) -> None:
        for _ in self:
            pass
