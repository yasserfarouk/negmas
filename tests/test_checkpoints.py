from pathlib import Path, PosixPath

from hypothesis import given, example
import hypothesis.strategies as st
from negmas import SAOMechanism, MappingUtilityFunction, AspirationNegotiator
from negmas.checkpoints import CheckpointRunner
from negmas.helpers import unique_name


def checkpoint_every(args):
    pass


@given(
    single_checkpoint=st.booleans(),
    checkpoint_every=st.integers(1, 4),
    exist_ok=st.booleans(),
)
def test_can_run_from_checkpoint(tmp_path, single_checkpoint, checkpoint_every, exist_ok):
    import shutil
    new_folder: Path = tmp_path / unique_name("empty", sep="")
    second_folder: Path = tmp_path / unique_name("second", sep="")
    new_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)
    filename = "mechanism"
    second_folder.mkdir(parents=True, exist_ok=True)

    n_outcomes, n_negotiators = 5, 3
    n_steps = 50
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        offering_is_accepting=True,
        avoid_ultimatum=False,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=new_folder,
        checkpoint_filename=filename,
        extra_checkpoint_info=None,
        exist_ok=exist_ok,
        single_checkpoint=single_checkpoint,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])

    mechanism.run()
    files = list(new_folder.glob("*"))
    if 0 < checkpoint_every <= n_steps:
        if single_checkpoint:
            assert len(list(new_folder.glob("*"))) == 2
        else:
            assert len(list(new_folder.glob("*"))) >= 2 * (
                max(1, mechanism.state.step // checkpoint_every)
            )
    elif checkpoint_every > n_steps:
        assert len(list(new_folder.glob("*"))) == 2
    else:
        assert len(list(new_folder.glob("*"))) == 0

    runner = CheckpointRunner(folder=new_folder)

    assert len(runner.steps) * 2 == len(files)
    assert runner.current_step == -1
    assert runner.loaded_object is None

    runner.step()

    assert runner.current_step == (0 if not single_checkpoint else runner.steps[-1])
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.reset()
    assert len(runner.steps) * 2 == len(files)
    assert runner.current_step == -1
    assert runner.loaded_object is None

    runner.goto(runner.last_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.next_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.previous_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.first_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.reset()

    runner.run()


@given(
    checkpoint_every=st.integers(1, 4),
    exist_ok=st.booleans(),
    copy=st.booleans(),
    fork_after_reset=st.booleans(),
)
@example(tmp_path=PosixPath('/private/var/folders/5t/9gn9cd716f34b13y2rk40p4w0000gn/T/pytest-of-yasser/pytest-104/test_can_run_from_checkpoint0'), checkpoint_every=1, exist_ok=False, copy=True, fork_after_reset=False)
def test_can_run_from_checkpoint(tmp_path, checkpoint_every, exist_ok, copy, fork_after_reset):
    import shutil
    new_folder: Path = tmp_path / unique_name("empty", sep="")
    second_folder: Path = tmp_path / unique_name("second", sep="")
    new_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new_folder)
    new_folder.mkdir(parents=True, exist_ok=True)
    second_folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(second_folder)
    second_folder.mkdir(parents=True, exist_ok=True)

    filename = "mechanism"

    n_outcomes, n_negotiators = 5, 3
    n_steps = 50
    mechanism = SAOMechanism(
        outcomes=n_outcomes,
        n_steps=n_steps,
        offering_is_accepting=True,
        avoid_ultimatum=False,
        checkpoint_every=checkpoint_every,
        checkpoint_folder=new_folder,
        checkpoint_filename=filename,
        extra_checkpoint_info=None,
        exist_ok=exist_ok,
        single_checkpoint=False,
    )
    ufuns = MappingUtilityFunction.generate_random(n_negotiators, outcomes=n_outcomes)
    for i in range(n_negotiators):
        mechanism.add(AspirationNegotiator(name=f"agent{i}"), ufun=ufuns[i])

    mechanism.run()
    files = list(new_folder.glob("*"))
    if 0 < checkpoint_every <= n_steps:
        assert len(list(new_folder.glob("*"))) >= 2 * (
            max(1, mechanism.state.step // checkpoint_every)
        )
    elif checkpoint_every > n_steps:
        assert len(list(new_folder.glob("*"))) == 2
    else:
        assert len(list(new_folder.glob("*"))) == 0

    runner = CheckpointRunner(folder=new_folder)

    assert len(runner.steps) * 2 == len(files)
    assert runner.current_step == -1
    assert runner.loaded_object is None

    runner.step()

    assert runner.current_step == 0
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.reset()
    assert len(runner.steps) * 2 == len(files)
    assert runner.current_step == -1
    assert runner.loaded_object is None

    runner.goto(runner.last_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.next_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.previous_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.first_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.reset()

    if fork_after_reset:
        m = runner.fork(copy_past_checkpoints=copy
                        , folder=second_folder)
        assert m is None
        return
    runner.step()
    m = runner.fork(copy_past_checkpoints=copy, folder=second_folder)

    if copy:
        step = runner.current_step
        assert len(list(second_folder.glob("*"))) >= 2
        assert len(list(second_folder.glob(f"*{step}.mechanism"))) > 0
    else:
        assert len(list(second_folder.glob("*"))) == 0

    assert isinstance(m, SAOMechanism)
    step = m.current_step
    m.step()
    assert m.current_step == step + 1

    state = m.run()
    assert state.agreement is not None

    runner.reset()
    assert len(runner.steps) * 2 == len(files)
    assert runner.current_step == -1
    assert runner.loaded_object is None

    runner.goto(runner.last_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.next_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.previous_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.goto(runner.first_step, exact=True)
    assert isinstance(runner.loaded_object, SAOMechanism)
    assert runner.loaded_object.state.step == runner.current_step

    runner.reset()

    runner.run()

