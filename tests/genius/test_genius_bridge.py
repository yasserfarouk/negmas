from __future__ import annotations

import os

import pytest

from negmas import genius_bridge_is_running
from negmas.genius import GeniusBridge

DO_BRIDGE_OPS = os.environ.get("NEGMAS_TEST_BRIDBE_OPS", False)


@pytest.fixture(scope="module")
def init_genius():
    GeniusBridge.start(0)


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_cleaning(init_genius):
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_shuttingdown(init_genius):
    GeniusBridge.shutdown()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_threads(init_genius):
    GeniusBridge.kill_threads()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_polietly(init_genius):
    GeniusBridge.kill()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_forcibly(init_genius):
    GeniusBridge.kill_forced()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_starting():
    port = GeniusBridge.start(-1)
    assert port
    assert genius_bridge_is_running(port)


@pytest.mark.skipif(
    condition=not DO_BRIDGE_OPS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_restarting():
    port = GeniusBridge.restart()
    assert port
    assert genius_bridge_is_running(port)
