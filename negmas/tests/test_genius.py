import pathlib

import numpy as np
import pkg_resources
import pytest

from negmas import GeniusNegotiator, load_genius_domain, load_genius_domain_from_folder, genius_bridge_is_running

dom_folder = pathlib.Path(pkg_resources.resource_filename('negmas'
                                                          , resource_name='tests/data/scenarios/anac/y2010/Travel'))
dom = dom_folder / 'travel_domain.xml'
util1 = dom_folder / 'travel_chox.xml'
util2 = dom_folder / 'travel_fanny.xml'


@pytest.fixture(scope='module')
def init_genius():
    pass


@pytest.mark.skipif(condition=not genius_bridge_is_running(), reason='No Genius Bridge, skipping genius-agent tests')
def test_genius_agent(init_genius):
    p, _, issues = load_genius_domain_from_folder(dom_folder
                                                  , agent_factories=[
            lambda: GeniusNegotiator(java_class_name='agents.anac.y2015.Atlas3.Atlas3'
                                     , domain_file_name=dom, utility_file_name=util1),
            lambda: GeniusNegotiator(java_class_name='agents.anac.y2015.AgentX.AgentX'
                                     , domain_file_name=dom, utility_file_name=util2)]
                                                  , keep_issue_names=True, keep_value_names=True, time_limit=20)
    assert len(p.negotiators) > 1
    issue_list = [f'{k}:{v}' for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)'"
        ", 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo', 'Festivals'"
        ", 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.', 'Monuments'"
        ", 'Special streets', 'Palace', 'Landscape and nature']"]
    atlas3, agentx = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(atlas3.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(agentx.ufun(s.current_offer)) for s in p.history])
    welfare = u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 0


@pytest.mark.skipif(condition=not genius_bridge_is_running(), reason='No Genius Bridge, skipping genius-agent tests')
def test_genius_agent_step_limit(init_genius):
    p, _, issues = load_genius_domain_from_folder(dom_folder
                                                  , agent_factories=[
            lambda: GeniusNegotiator(java_class_name='agents.anac.y2015.Atlas3.Atlas3'
                                     , domain_file_name=dom, utility_file_name=util1),
            lambda: GeniusNegotiator(java_class_name='agents.anac.y2015.AgentX.AgentX'
                                     , domain_file_name=dom, utility_file_name=util2)]
                                                  , keep_issue_names=True, keep_value_names=True, n_steps=20,
                                                  time_limit=None)
    issue_list = [f'{k}:{v}' for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)', 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo', 'Festivals', 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.', 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']"]
    atlas3, agentx = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(atlas3.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(agentx.ufun(s.current_offer)) for s in p.history])
    welfare = u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)


@pytest.mark.skipif(condition=not genius_bridge_is_running(), reason='No Genius Bridge, skipping genius-agent tests')
def test_genius_agent_step_long_session(init_genius):
    a1 = GeniusNegotiator(java_class_name="agents.anac.y2015.Atlas3.Atlas3"
                          , domain_file_name=dom, utility_file_name=util1)
    a2 = GeniusNegotiator(java_class_name="agents.anac.y2015.Atlas3.Atlas3"
                          , domain_file_name=dom, utility_file_name=util2)
    p, _, issues = load_genius_domain(dom, keep_issue_names=True
                                      , keep_value_names=True
                                      , n_steps=20, time_limit=None)
    issue_list = [f'{k}:{v}' for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)', 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo', 'Festivals', 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.', 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']"]
    p.add(a1)
    p.add(a2)
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(a1.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(a2.ufun(s.current_offer)) for s in p.history])
    welfare = u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 4


@pytest.mark.skipif(condition=not genius_bridge_is_running(), reason='No Genius Bridge, skipping genius-agent tests')
def test_genius_agent_same_utility(init_genius):
    from negmas import GeniusNegotiator, load_genius_domain
    dom = dom_folder / 'travel_domain.xml'
    util1 = dom_folder / 'travel_chox.xml'
    util2 = util1
    a1 = GeniusNegotiator(java_class_name="agents.anac.y2015.Atlas3.Atlas3"
                          , domain_file_name=dom, utility_file_name=util1)
    a2 = GeniusNegotiator(java_class_name="agents.anac.y2015.AgentX.AgentX"
                          , domain_file_name=dom, utility_file_name=util2)
    p, _, issues = load_genius_domain(dom, keep_issue_names=True
                                      , keep_value_names=True
                                      , time_limit=30)
    issue_list = [f'{k}:{v}' for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)'"
        ", 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo'"
        ", 'Festivals', 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.'"
        ", 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']"]
    p.add(a1)
    p.add(a2)
    p.run()
    u1 = np.array([float(a1.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(a2.ufun(s.current_offer)) for s in p.history])
    welfare = u1 + u2
    assert len(u1) == 2
    assert welfare[0] == welfare[1] == 2.0
    assert p.state.agreement is not None
    assert p.state.broken is False


class TestGeniusAgentSessions:
    def prepare(self, utils=(1, 1), single_issue=True, keep_issue_names=True,
                keep_value_names=True):
        from negmas import convert_genius_domain_from_folder
        src = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        dst = pkg_resources.resource_filename('negmas', resource_name='tests/data/LaptopConv1D')
        if single_issue:
            assert convert_genius_domain_from_folder(src_folder_name=src
                                                     , dst_folder_name=dst
                                                     , force_single_issue=True
                                                     , cache_and_discretize_outcomes=True
                                                     , n_discretization=10
                                                     )
            base_folder = dst
        else:
            base_folder = src
        neg, agent_info, issues = load_genius_domain_from_folder(base_folder
                                                                 , keep_issue_names=keep_issue_names
                                                                 , keep_value_names=keep_value_names)
        # atlas = GeniusNegotiator.random_negotiator(
        atlas = GeniusNegotiator(java_class_name='agents.anac.y2015.Atlas3.Atlas3',
                                 domain_file_name=base_folder + '/Laptop-C-domain.xml'
                                 , utility_file_name=base_folder + f'/Laptop-C-prof{utils[0]}.xml'
                                 , keep_issue_names=keep_issue_names
                                 , keep_value_names=keep_value_names)
        # agentx = GeniusNegotiator.random_negotiator(
        agentx = GeniusNegotiator(java_class_name='agents.anac.y2015.AgentX.AgentX',
                                  domain_file_name=base_folder + '/Laptop-C-domain.xml'
                                  , utility_file_name=base_folder + f'/Laptop-C-prof{utils[1]}.xml'
                                  , keep_issue_names=keep_issue_names
                                  , keep_value_names=keep_value_names
                                  )
        neg.add(atlas)
        neg.add(agentx)
        return neg

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agents_can_run_on_converted_single_issue_ufun1(self, init_genius):
        neg = self.prepare(utils=(1, 1), single_issue=True)
        assert neg.pareto_frontier(sort_by_welfare=True)[0] == [(1.0, 1.0)]
        state = neg.run()
        # pprint(neg.history)
        assert neg.agreement is not None
        #        assert len(neg.history) <= 3
        assert neg.agreement == {'Laptop-Harddisk-External Monitor': "HP+60 Gb+19'' LCD"}

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agents_can_run_on_converted_single_issue_ufun2(self, init_genius):
        neg = self.prepare(utils=(2, 2), single_issue=True)
        assert neg.pareto_frontier(sort_by_welfare=True)[0] == [(1.0, 1.0)]
        state = neg.run()
        assert neg.agreement is not None
        assert len(neg.history) <= 3
        assert neg.agreement == {'Laptop-Harddisk-External Monitor': "Macintosh+80 Gb+19'' LCD"}

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agents_can_run_on_converted_single_issue(self, init_genius):
        neg = self.prepare(utils=(1, 2), single_issue=True)
        assert neg.pareto_frontier(sort_by_welfare=True)[0] == [(0.7715533992081258, 0.8450562871935449),
                                                                (0.5775524426410947, 1.0),
                                                                (1.0, 0.5136317604069089),
                                                                (0.8059990434329689, 0.6685754732133642)
                                                                ]

        state = neg.run()
        # assert len(neg.history) >= 2
        assert neg.agreement is not None

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agents_can_run_on_converted_multiple_issues(self, init_genius):
        neg = self.prepare(utils=(1, 1), single_issue=False)
        frontier = neg.pareto_frontier(sort_by_welfare=True)[0]
        true_frontier = [(1.0, 1.0)]
        assert len(frontier) == len(true_frontier)
        for a, b in zip(frontier, true_frontier):
            assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5

        state = neg.run()
        assert len(neg.history) < 3
        assert neg.agreement is not None
        assert neg.agreement == {'Laptop': 'HP', 'Harddisk': '60 Gb', 'External Monitor': "19'' LCD"}

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agents_can_run_on_converted_multiple_issues_no_names(self, init_genius):
        neg = self.prepare(utils=(1, 1), single_issue=False, keep_issue_names=False)
        frontier = neg.pareto_frontier(sort_by_welfare=True)[0]
        true_frontier = [(1.0, 1.0)]
        assert len(frontier) == len(true_frontier)
        for a, b in zip(frontier, true_frontier):
            assert abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5

        state = neg.run()
        assert len(neg.history) < 3
        assert neg.agreement is not None
        assert neg.agreement == ('HP', '60 Gb', "19'' LCD")

    @pytest.mark.skipif(condition=not genius_bridge_is_running(),
                        reason='No Genius Bridge, skipping genius-agent tests')
    def test_genius_agent_example(self, init_genius):
        agent_name1 = 'agents.anac.y2015.Atlas3.Atlas3'
        agent_name2 = 'agents.anac.y2015.Atlas3.Atlas3'
        single_issue = False
        keep_issue_names, keep_value_names = False, False
        utils = (1, 1)

        from negmas import convert_genius_domain_from_folder
        src = pkg_resources.resource_filename('negmas', resource_name='tests/data/Laptop')
        dst = pkg_resources.resource_filename('negmas', resource_name='tests/data/LaptopConv1D')
        if single_issue:
            assert convert_genius_domain_from_folder(src_folder_name=src
                                                     , dst_folder_name=dst
                                                     , force_single_issue=True
                                                     , cache_and_discretize_outcomes=True
                                                     , n_discretization=10
                                                     )
            base_folder = dst
        else:
            base_folder = src
        neg, agent_info, issues = load_genius_domain_from_folder(base_folder
                                                                 , keep_issue_names=keep_issue_names
                                                                 , keep_value_names=keep_value_names)
        # atlas = GeniusNegotiator.random_negotiator(
        atlas = GeniusNegotiator(java_class_name=agent_name1,
                                 domain_file_name=base_folder + '/Laptop-C-domain.xml'
                                 , utility_file_name=base_folder + f'/Laptop-C-prof{utils[0]}.xml'
                                 , keep_issue_names=keep_issue_names
                                 , keep_value_names=keep_value_names)
        agentx = GeniusNegotiator(java_class_name=agent_name2,
                                  domain_file_name=base_folder + '/Laptop-C-domain.xml'
                                  , utility_file_name=base_folder + f'/Laptop-C-prof{utils[1]}.xml'
                                  , keep_issue_names=keep_issue_names
                                  , keep_value_names=keep_value_names
                                  )
        neg.add(atlas)
        neg.add(agentx)
        # print(agent_name1, agent_name2, neg.run(timeout=1))


if __name__ == '__main__':
    pytest.main(args=[__file__])
