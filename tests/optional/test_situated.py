import random
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from negmas.helpers import unique_name
from negmas.situated.world import World
from negmas.tests.test_situated import DummyWorld, NegAgent, NegPerStepWorld

results = []  # will keep results not to use printing
N_NEG_STEPS = 20


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    single_checkpoint=st.booleans(),
    checkpoint_every=st.integers(0, 6),
    exist_ok=st.booleans(),
)
def test_world_auto_checkpoint(tmp_path, single_checkpoint, checkpoint_every, exist_ok):
    import shutil

    new_folder: Path = tmp_path / unique_name("empty", sep="")
    new_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)
    filename = "mechanism"
    n_steps = 20

    world = DummyWorld(
        n_steps=n_steps,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=new_folder,
        checkpoint_filename=filename,
        extra_checkpoint_info=None,
        exist_ok=exist_ok,
        single_checkpoint=single_checkpoint,
    )

    world.run()

    if 0 < checkpoint_every <= n_steps:
        if single_checkpoint:
            assert (
                len(list(new_folder.glob("*"))) == 2
            ), f"World ran for: {world.current_step}"
        else:
            assert len(list(new_folder.glob("*"))) >= 2 * (
                max(1, world.current_step // checkpoint_every)
            )
    elif checkpoint_every > n_steps:
        assert len(list(new_folder.glob("*"))) == 2
    else:
        assert len(list(new_folder.glob("*"))) == 0


@given(p_request=st.floats(0.0, 1.0))
@settings(deadline=10_000, max_examples=20)
def test_neg_world_steps_normally(p_request):
    n_steps = N_NEG_STEPS
    world = NegPerStepWorld(n_steps)
    for _ in range(5):
        world.join(NegAgent(p_request=p_request, name=f"a{_}"))
    assert world.current_step == 0
    for i in range(n_steps):
        assert world.step()
        assert world.current_step == i + 1
    for i in range(n_steps):
        assert not world.step()
        assert world.current_step == n_steps


@pytest.mark.skip("Flaky")
@given(
    pertype=st.booleans(),
    n_step=st.sampled_from((-1, None, 10, 100)),
    method=st.sampled_from((np.mean, np.std, np.median)),
    n=st.integers(1, 10),
)
def test_combine_stats(pertype, n_step, method, n):
    worlds = []
    n_steps = [random.randint(10, 65) for _ in range(n)]
    for ns in n_steps:
        world = DummyWorld(n_steps=ns, no_logs=True)
        world.run()
        worlds.append(world)
    assert len(worlds) == n

    for stat in world.stat_names:  # type: ignore
        x = World.combine_stats(
            tuple(worlds), stat, pertype=pertype, n_steps=n_step, method=method
        )
        for k, v in x.items():
            if n_step is None:
                assert len(v) == max(
                    n_steps
                ), f"{k=}: {len(v)=} but {max(n_steps)=} and None"
            elif n_step < 0:
                assert len(v) == max(
                    n_steps
                ), f"{k=}: {len(v)=} but {max(n_steps)=} and < 0"
            else:
                assert len(v) == n_step, f"{k=}: {len(v)=} but {n_step=}"


# @given(
#     pertype=st.booleans(),
#     nstep=st.sampled_from((-1, None, 10, 100)),
#     n=st.integers(1, 10),
# )
# @example(pertype=False, nstep=-1, n=1)
# def test_plot_combined(pertype, nstep, n):
#     worlds = []
#     n_steps = [random.randint(10, 65) for _ in range(n)]
#     for ns in n_steps:
#         world = DummyWorld(n_steps=ns, no_logs=True)
#         world.run()
#         worlds.append(world)
#     assert len(worlds) == n
#     DummyWorld.plot_combined_stats(
#         tuple(worlds),
#         stats="activity_level",
#         n_steps=nstep,
#         pertype=pertype,
#         makefig=False,
#     )
#     # import matplotlib.pyplot as plt
#     #
#     # plt.show()
#     # assert False
