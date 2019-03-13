import os
import numpy as np
import pkg_resources
import pytest
import pathlib
from hypothesis import given, settings
import hypothesis.strategies as st


def donot_test_can_init_jnegmas_on_default_port():
    from negmas.java import init_jnegmas_bridge
    init_jnegmas_bridge(None, 0)


