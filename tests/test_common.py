from __future__ import annotations

import random
import time

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from negmas import NamedObject
from negmas.helpers import unique_name

random.seed(time.perf_counter())

good_attribs = [
    "current_step",
    "_current_step",
    "_Entity__current_step",
    "_step",
]

bad_attribs = ["sdfds", "ewre"]


class WithStep(NamedObject):
    _step = 3


class MyEntity(NamedObject):
    pass


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    exist_ok=st.booleans(),
    single_checkpoint=st.booleans(),
    with_name=st.booleans(),
    with_info=st.booleans(),
    step_attribs=st.tuples(st.sampled_from(good_attribs + bad_attribs)),
)
def test_checkpoint(
    tmp_path, exist_ok, with_name, with_info, single_checkpoint, step_attribs
):
    x = WithStep()

    fname = unique_name("abc", rand_digits=10, add_time=True, sep=".")
    try:
        file_name = x.checkpoint(
            path=tmp_path,
            file_name=fname if with_name else None,
            info={"r": 3} if with_info else None,
            exist_ok=exist_ok,
            single_checkpoint=single_checkpoint,
            step_attribs=step_attribs,
        )
        assert (
            file_name.name.split(".")[0].isnumeric()
            or single_checkpoint
            or all(_ in bad_attribs for _ in set(step_attribs))
            or not any(hasattr(x, _) for _ in step_attribs)
        )
    except ValueError as e:
        if "exist_ok" in str(e):
            assert not exist_ok
        else:
            raise e

    x = MyEntity()

    fname = unique_name("abc", rand_digits=10, add_time=True, sep=".")
    try:
        file_name = x.checkpoint(
            path=tmp_path,
            file_name=fname if with_name else None,
            info={"r": 3} if with_info else None,
            exist_ok=exist_ok,
            single_checkpoint=single_checkpoint,
            step_attribs=step_attribs,
        )
        assert (
            file_name.name.split(".")[0].isnumeric()
            or single_checkpoint
            or all(_ in bad_attribs for _ in set(step_attribs))
            or not any(hasattr(x, _) for _ in step_attribs)
        )
    except ValueError as e:
        if "exist_ok" in str(e):
            assert not exist_ok
        else:
            raise e
