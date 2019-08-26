from pprint import pprint

from negmas.apps.scml import anac2019_std, GreedyFactoryManager, DoNothingFactoryManager, anac2019_collusion
from negmas.apps.scml.utils import anac2019_sabotage


def test_std():
    results = anac2019_std(competitors=[DoNothingFactoryManager, GreedyFactoryManager], n_steps=5, n_configs=1
                           , n_runs_per_world=1, max_worlds_per_config=2)
    assert len(results.total_scores) >= 2
    assert results.total_scores.loc[results.total_scores.agent_type == 'do_nothing_factory_manager', 'score'].values[
               0] == 0.0


def test_collusion():
    results = anac2019_collusion(competitors=[DoNothingFactoryManager, GreedyFactoryManager], n_steps=5, n_configs=1
                                 , n_runs_per_world=1, max_worlds_per_config=2)
    assert len(results.total_scores) >= 2
    assert results.total_scores.loc[results.total_scores.agent_type == 'do_nothing_factory_manager', 'score'].values[
               0] == 0.0


class Greedy1(GreedyFactoryManager):
    pass


def test_sabotage():
    results = anac2019_sabotage(competitors=[DoNothingFactoryManager, Greedy1], n_steps=5, n_configs=1
                                , n_runs_per_world=1, min_factories_per_level=1, n_default_managers=1
                                , n_agents_per_competitor=2, max_worlds_per_config=2)
    assert len(results.total_scores) >= 2
