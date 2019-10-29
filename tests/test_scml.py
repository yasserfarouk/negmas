import pytest

from negmas.apps.scml.utils import anac2019_world


@pytest.mark.parametrize(
    "n_steps,consumption_horizon", [(5, 5), (10, 5)], ids=["tiny", "short"]
)
def test_anac2019(n_steps, consumption_horizon):
    world = anac2019_world(n_steps=n_steps, consumption_horizon=consumption_horizon)
    world.run()


if __name__ == "__main__":
    pytest.main(args=[__file__])
