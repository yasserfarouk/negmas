import time

import pytest

from negmas import genius_bridge_is_running
from negmas.genius import GeniusBridge


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_cleaning():
    GeniusBridge.clean()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_shuttingdown():
    GeniusBridge.shutdown()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_threads():
    GeniusBridge.kill_threads()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_polietly():
    GeniusBridge.kill()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_killing_forcibly():
    GeniusBridge.kill_forced()
    assert not genius_bridge_is_running()


@pytest.mark.skipif(
    condition=genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_starting():
    GeniusBridge.start()
    assert genius_bridge_is_running()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_genius_bridge_restarting():
    GeniusBridge.restart()
    # time.sleep(1)
    assert genius_bridge_is_running()
