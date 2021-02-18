import time

import pytest

from negmas import genius_bridge_is_running
from negmas.genius import GeniusBridge

SKIP_IF_NO_BRIDGE = True


@pytest.fixture(scope="module")
def init_genius():
    GeniusBridge.start(0)


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_cleaning(init_genius):
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_shuttingdown(init_genius):
    GeniusBridge.shutdown()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_threads(init_genius):
    GeniusBridge.kill_threads()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_polietly(init_genius):
    GeniusBridge.kill()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE and not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_forcibly(init_genius):
    GeniusBridge.kill_forced()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE, reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_starting():
    port = GeniusBridge.start(0)
    assert port > 0
    assert genius_bridge_is_running(port)


@pytest.mark.skipif(
    condition=SKIP_IF_NO_BRIDGE, reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_restarting():
    port = GeniusBridge.restart()
    assert port > 0
    assert genius_bridge_is_running(port)
