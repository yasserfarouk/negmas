from __future__ import annotations

import pytest

# Track whether we started the bridge ourselves
_bridge_started_by_tests = False
_bridge_port = None


@pytest.fixture(scope="session", autouse=True)
def manage_genius_bridge():
    """
    Manage Genius bridge for the test session.

    - If bridge is installed and not running, start it
    - Only stop the bridge after tests if we started it ourselves
    """
    global _bridge_started_by_tests, _bridge_port

    try:
        from negmas.genius.bridge import (
            GeniusBridge,
            genius_bridge_is_installed,
            genius_bridge_is_running,
        )
        from negmas.genius.common import DEFAULT_JAVA_PORT

        _bridge_port = DEFAULT_JAVA_PORT

        # Check if bridge is installed
        if not genius_bridge_is_installed():
            # Bridge not installed, nothing to do
            yield
            return

        # Check if bridge is already running
        if genius_bridge_is_running(_bridge_port):
            # Bridge already running, don't touch it
            _bridge_started_by_tests = False
            yield
            return

        # Bridge is installed but not running - start it
        try:
            GeniusBridge.start(port=_bridge_port)
            _bridge_started_by_tests = True
        except Exception:
            # Failed to start, continue anyway (tests will be skipped)
            _bridge_started_by_tests = False

        yield

        # Cleanup: only stop if we started it
        if _bridge_started_by_tests:
            try:
                GeniusBridge.stop(port=_bridge_port)
            except Exception:
                pass
    except ImportError:
        # py4j or other dependencies not available
        yield


@pytest.fixture(autouse=True)
def cleanup_after_each_test():
    """Clean up agents after each test to prevent resource leaks."""
    yield
    try:
        from negmas.genius.bridge import GeniusBridge

        # Just clean agents, don't stop the bridge
        GeniusBridge.clean_all()
    except Exception:
        pass
