import pytest

from negmas.apps.scml import *


@pytest.mark.parametrize('n_steps,consumption_horizon'
    , [(10, 10), (60, 10)], ids=['short', 'default'])
def test_anac2019(n_steps, consumption_horizon):
    world = anac2019_world(n_steps=n_steps, consumption_horizon=consumption_horizon)
    world.run()
    assert world.business_size > 0.0
    assert world.breach_rate < 0.9
    assert world.agreement_rate > 0.1


if __name__ == '__main__':
    pytest.main(args=[__file__])
