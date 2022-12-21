from __future__ import annotations

import pathlib
from math import isclose

import numpy as np
import pkg_resources
import pytest
from numpy.testing import assert_almost_equal

from negmas import (
    GeniusNegotiator,
    genius_bridge_is_running,
    load_genius_domain_from_folder,
)
from negmas.genius import AgentX, Atlas3, Caduceus, GeniusBridge, YXAgent
from negmas.genius.gnegotiators import AgentK

DOMAIN_FOLDER = pathlib.Path(
    pkg_resources.resource_filename(
        "negmas", resource_name="tests/data/scenarios/anac/y2010/Travel"
    )
)
DOMAIN_FILE = DOMAIN_FOLDER / "travel_domain.xml"
UTIL1 = DOMAIN_FOLDER / "travel_chox.xml"
UTIL2 = DOMAIN_FOLDER / "travel_fanny.xml"


SKIP_IF_NO_BRIDGE = True


@pytest.fixture(scope="module")
def init_genius():
    GeniusBridge.start(0)


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent(init_genius):
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    p = domain.make_session(
        [
            GeniusNegotiator(
                java_class_name="agents.anac.y2015.Atlas3.Atlas3",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL1,
            ),
            GeniusNegotiator(
                java_class_name="agents.anac.y2015.AgentX.AgentX",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL2,
            ),
        ],
        time_limit=120,
        n_steps=None,
    )
    issues = domain.issues
    assert len(p.negotiators) > 1
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    first, second = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(first.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(second.ufun(s.current_offer)) for s in p.history])
    u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 0


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_top2016_caduceus_first(init_genius):
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    issues = domain.issues
    p = domain.make_session(
        [
            Caduceus(
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL1,
            ),
            AgentX(
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL2,
            ),
        ],
        time_limit=20,
        n_steps=None,
    )
    assert len(p.negotiators) > 1
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    first, second = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(first.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(second.ufun(s.current_offer)) for s in p.history])
    u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 0


def do_run(first_type, second_type):
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    issues = domain.issues
    p = domain.make_session(
        [
            first_type(domain_file_name=DOMAIN_FILE, utility_file_name=UTIL1),
            second_type(domain_file_name=DOMAIN_FILE, utility_file_name=UTIL2),
        ],
        time_limit=20,
        n_steps=None,
    )
    assert len(p.negotiators) > 1
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    first, second = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(first.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(second.ufun(s.current_offer)) for s in p.history])
    # welfare = u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 0 or len(u2) > 0


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
@pytest.mark.parametrize("first_type, second_type", [(AgentX, YXAgent)])
def test_genius_agent_top2016_yx_second_classes(init_genius, first_type, second_type):
    do_run(first_type, second_type)


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_top2016_yx_second(init_genius):
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    issues = domain.issues
    p = domain.make_session(
        [
            GeniusNegotiator(
                java_class_name="agents.anac.y2016.yxagent.YXAgent",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL1,
            ),
            GeniusNegotiator(
                java_class_name="agents.anac.y2015.AgentX.AgentX",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL2,
            ),
        ],
        time_limit=20,
        n_steps=None,
    )
    assert len(p.negotiators) > 1
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    first, second = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(first.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(second.ufun(s.current_offer)) for s in p.history])
    u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 0


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_step_limit(init_genius):
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    issues = domain.issues
    p = domain.make_session(
        [
            GeniusNegotiator(
                java_class_name="agents.anac.y2015.Atlas3.Atlas3",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL1,
            ),
            GeniusNegotiator(
                java_class_name="agents.anac.y2015.AgentX.AgentX",
                domain_file_name=DOMAIN_FILE,
                utility_file_name=UTIL2,
            ),
        ],
        time_limit=None,
        n_steps=20,
    )
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)', 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo', 'Festivals', 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.', 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    atlas3, agentx = p.negotiators[0], p.negotiators[1]
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(atlas3.ufun(s.current_offer)) for s in p.history])
    u2 = np.array([float(agentx.ufun(s.current_offer)) for s in p.history])
    u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_step_long_session(init_genius):
    a1 = GeniusNegotiator(
        java_class_name="agents.anac.y2015.Atlas3.Atlas3",
        domain_file_name=DOMAIN_FILE,
        utility_file_name=UTIL1,
    )
    a2 = GeniusNegotiator(
        java_class_name="agents.anac.y2015.Atlas3.Atlas3",
        domain_file_name=DOMAIN_FILE,
        utility_file_name=UTIL2,
    )
    domain = load_genius_domain_from_folder(DOMAIN_FOLDER).normalize()
    issues = domain.issues
    p = domain.make_session([a1, a2], time_limit=None, n_steps=20)
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
    assert issue_list == [
        "0:Atmosphere: ['Cultural heritage', 'Local traditions', 'Political stability', 'Security (personal)', 'Liveliness', 'Turistic activities', 'Hospitality']",
        "1:Amusement: ['Nightlife and entertainment', 'Nightclubs', 'Excursion', 'Casinos', 'Zoo', 'Festivals', 'Amusement park']",
        "2:Culinary: ['Local cuisine', 'Lunch facilities', 'International cuisine', 'Cooking workshops']",
        "3:Shopping: ['Shopping malls', 'Markets', 'Streets', 'Small boutiques']",
        "4:Culture: ['Museum', 'Music hall', 'Theater', 'Art gallery', 'Cinema', 'Congress center']",
        "5:Sport: ['Bike tours', 'Hiking', 'Indoor activities', 'Outdoor activities', 'Adventure']",
        "6:Environment: ['Parks and Gardens', 'Square', 'Historical places', 'See, river, etc.', 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    p.add(a1)
    p.add(a2)
    p.run()
    # print(f'{len(p.history)} bids exchanged')
    u1 = np.array([float(a1._preferences(s.current_offer)) for s in p.history])
    u2 = np.array([float(a2._preferences(s.current_offer)) for s in p.history])
    u1 + u2
    # print(
    #     f'Negotiator 1: {u1.mean()}({u1.std()})[{u1.min()}, {u1.max()}]\nNegotiator 2: {u2.mean()}({u2.std()})[{u1.min()}, {u1.max()}]'
    #     f'\nWelfare: {welfare.mean()}({welfare.std()})[{welfare.min()}, {welfare.max()}]')
    # print(p.state)
    assert len(u1) > 4


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_same_utility_with_normalization():
    from negmas import load_genius_domain_from_folder

    domain = load_genius_domain_from_folder(DOMAIN_FOLDER).normalize()
    issues = domain.issues
    p = domain.make_session(time_limit=300, n_steps=None)
    assert p is not None, "Could not create a mechanism"
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    a1 = AgentK(preferences=domain.ufuns[0])
    a2 = Atlas3(preferences=domain.ufuns[0])
    p.add(a1)
    p.add(a2)
    final = p.run()
    u1 = [float(a1.preferences(o)) for o in p.offers]
    u2 = [float(a2.preferences(o)) for o in p.offers]
    assert len(u1) >= 1 or len(u2) >= 1
    u1, u2 = a1.ufun(final.agreement), a2.ufun(final.agreement)
    welfare = u1 + u2
    assert p.state.agreement is not None
    assert p.state.broken is False
    assert welfare > 1


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_agent_same_utility():
    from negmas import load_genius_domain_from_folder

    domain = load_genius_domain_from_folder(DOMAIN_FOLDER)
    issues = domain.issues
    p = domain.make_session(time_limit=300, n_steps=None)
    assert p is not None, "Could not create a mechanism"
    issue_list = [f"{k}:{v}" for k, v in enumerate(issues)]
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
        ", 'Monuments', 'Special streets', 'Palace', 'Landscape and nature']",
    ]
    a1 = AgentK(preferences=domain.ufuns[0])
    a2 = Atlas3(preferences=domain.ufuns[0])
    p.add(a1)
    p.add(a2)
    final = p.run()
    u1 = [float(a1.preferences(o)) for o in p.offers]
    u2 = [float(a2.preferences(o)) for o in p.offers]
    assert len(u1) >= 1 or len(u2) >= 1
    u1, u2 = a1.ufun(final.agreement), a2.ufun(final.agreement)
    welfare = u1 + u2
    assert p.state.agreement is not None
    assert p.state.broken is False
    assert welfare > 1


class TestGeniusAgentSessions:
    def prepare(
        self,
        utils=(0, 0),
        single_issue=True,
    ):

        base_folder = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/Laptop"
        )

        domain = load_genius_domain_from_folder(base_folder).normalize()
        if single_issue:
            domain = domain.to_single_issue(stringify=True)
        atlas = GeniusNegotiator(
            java_class_name="agents.anac.y2015.Atlas3.Atlas3",
            preferences=domain.ufuns[utils[0]],
        )
        # agentx = GeniusNegotiator.random_negotiator(
        agentx = GeniusNegotiator(
            java_class_name="agents.anac.y2015.AgentX.AgentX",
            preferences=domain.ufuns[utils[1]],
        )
        neg = domain.make_session(n_steps=100)
        neg.add(atlas)
        neg.add(agentx)
        return neg

    @pytest.mark.skipif(
        condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
        reason="No Genius Bridge, skipping genius-agent tests",
    )
    def test_genius_agents_can_run_on_converted_single_issue_preferences1(
        self, init_genius
    ):
        neg = self.prepare(utils=(0, 0), single_issue=True)
        best = neg.pareto_frontier(sort_by_welfare=True)[0][0]
        assert isclose(best[0], 1.0, abs_tol=0.001) and isclose(
            best[1], 1.0, abs_tol=0.001
        )
        neg.run()
        assert neg.agreement is not None
        assert len(neg.history) <= 3
        assert neg.agreement == ("v18",)

    @pytest.mark.skipif(
        condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
        reason="No Genius Bridge, skipping genius-agent tests",
    )
    def test_genius_agents_can_run_on_converted_single_issue_preferences2(
        self, init_genius
    ):
        neg = self.prepare(utils=(1, 1), single_issue=True)
        best = neg.pareto_frontier(sort_by_welfare=True)[0][0]
        assert isclose(best[0], 1.0, abs_tol=0.001) and isclose(
            best[1], 1.0, abs_tol=0.001
        )
        neg.run()
        assert neg.agreement is not None
        assert len(neg.history) <= 3
        assert neg.agreement == ("v12",)

    @pytest.mark.skipif(
        condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
        reason="No Genius Bridge, skipping genius-agent tests",
    )
    def test_genius_agents_can_run_on_converted_single_issue(self, init_genius):
        neg = self.prepare(utils=(0, 1), single_issue=True)
        front, _ = neg.pareto_frontier(sort_by_welfare=True)
        assert_almost_equal(
            front,
            [
                (0.7715, 0.845),
                (0.577, 1.0),
                (1.0, 0.5136),
                (0.805, 0.668),
            ],
            decimal=2,
        )

        neg.run()
        assert len(neg.history) >= 1
        assert neg.agreement is not None

    @pytest.mark.skipif(
        condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
        reason="No Genius Bridge, skipping genius-agent tests",
    )
    def test_genius_agents_can_run_on_converted_multiple_issues(self, init_genius):
        neg = self.prepare(utils=(0, 0), single_issue=False)
        frontier = neg.pareto_frontier(sort_by_welfare=True)[0]
        true_frontier = [(1.0, 1.0)]
        assert len(frontier) == len(true_frontier)
        for a, b in zip(frontier, true_frontier):
            assert abs(a[0] - b[0]) < 1 and abs(a[1] - b[1]) < 1
        # neg.set_sync_call(True)
        neg.run()
        assert len(neg.history) < 3
        assert neg.agreement is not None
        assert neg.agreement == ("HP", "60 Gb", "19'' LCD")

    # @pytest.mark.skipif(
    #     condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    #     reason="No Genius Bridge, skipping genius-agent tests",
    # )
    # def test_genius_agents_can_run_on_converted_multiple_issues_no_names(
    #     self, init_genius
    # ):
    #     neg = self.prepare(
    #         utils=(0, 0),
    #         single_issue=False,
    #         keep_issue_names=True,
    #         keep_value_names=True,
    #     )
    #     frontier = neg.pareto_frontier(sort_by_welfare=True)[0]
    #     true_frontier = [(1.0, 1.0)]
    #     assert len(frontier) == len(true_frontier)
    #     for a, b in zip(frontier, true_frontier):
    #         assert abs(a[0] - b[0]) < 1 and abs(a[1] - b[1]) < 1
    #
    #     state = neg.run()
    #     assert len(neg.history) < 4, len(neg.history)
    #     assert neg.agreement is not None
    #     # assert neg.agreement == {
    #     #     "External Monitor": "19'' LCD",
    #     #     "Harddisk": "60 Gb",
    #     #     "Laptop": "HP",
    #     # }
    #     assert (neg.agreement.values()) == ("HP", "60 Gb", "19'' LCD"), neg.agreement

    @pytest.mark.skipif(
        condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
        reason="No Genius Bridge, skipping genius-agent tests",
    )
    def test_genius_agent_example(self, init_genius):
        agent_name1 = "agents.anac.y2015.Atlas3.Atlas3"
        agent_name2 = "agents.anac.y2015.Atlas3.Atlas3"
        single_issue = False

        base_folder = pkg_resources.resource_filename(
            "negmas", resource_name="tests/data/Laptop"
        )
        domain = load_genius_domain_from_folder(base_folder)
        if single_issue:
            domain = domain.to_single_issue()
        # atlas = GeniusNegotiator.random_negotiator(
        atlas = GeniusNegotiator(
            java_class_name=agent_name1,
            domain_file_name=domain.agenda.name,
            utility_file_name=domain.ufuns[0].name,
        )
        agentx = GeniusNegotiator(
            java_class_name=agent_name2,
            domain_file_name=domain.agenda.name,
            utility_file_name=domain.ufuns[1].name,
        )
        neg = domain.make_session(n_steps=100)
        neg.add(atlas)
        neg.add(agentx)


if __name__ == "__main__":
    pytest.main(args=[__file__])
