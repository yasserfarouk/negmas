from __future__ import annotations
import os
from pathlib import Path

import papermill as pm
import pytest

from negmas.genius import genius_bridge_is_running, GeniusBridge

NEGMAS_IGNORE_TEST_NOTEBOOKS = os.environ.get("NEGMAS_IGNORE_TEST_NOTEBOOKS", False)
# NEGMAS_IGNORE_TEST_NOTEBOOKS = True

# Timeout for notebook execution in seconds (default: 10 minutes per cell)
NOTEBOOK_CELL_TIMEOUT = int(os.environ.get("NEGMAS_NOTEBOOK_CELL_TIMEOUT", 600))


def notebooks():
    base = Path(__file__).parent.parent.parent / "notebooks"
    return list(_ for _ in base.glob("**/*.ipynb") if "checkpoints" not in str(_))


@pytest.fixture(scope="module", autouse=True)
def ensure_genius_bridge():
    """Ensure the Genius bridge is running before notebook tests."""
    bridge_started = False
    if not genius_bridge_is_running():
        if GeniusBridge.is_installed():
            GeniusBridge.start()
            bridge_started = True
    yield
    # Optionally shut down the bridge if we started it
    if bridge_started:
        GeniusBridge.shutdown()


@pytest.mark.skipif(
    condition=NEGMAS_IGNORE_TEST_NOTEBOOKS,
    reason="No Genius Bridge, skipping genius-agent tests",
)
@pytest.mark.parametrize("notebook", notebooks())
def test_notebook(notebook):
    base = Path(__file__).parent.parent.parent / "notebooks"
    dst = notebook.relative_to(base)
    dst = Path(__file__).parent / "tmp_notebooks" / str(dst)
    dst.parent.mkdir(exist_ok=True, parents=True)
    pm.execute_notebook(notebook, dst, execution_timeout=NOTEBOOK_CELL_TIMEOUT)


if __name__ == "__main__":
    print(notebooks())
