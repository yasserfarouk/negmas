from pathlib import Path

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from negmas.helpers import unique_name
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
