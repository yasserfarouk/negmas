from __future__ import annotations

import os
from pathlib import Path

import papermill as pm
import pytest

NEGMAS_IGNORE_TEST_NOTEBOOKS = os.environ.get("NEGMAS_IGNORE_TEST_NOTEBOOKS", False)
# NEGMAS_IGNORE_TEST_NOTEBOOKS = True


def notebooks():
    base = Path(__file__).parent.parent.parent / "notebooks"
    return list(_ for _ in base.glob("**/*.ipynb") if "checkpoints" not in str(_))


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
    pm.execute_notebook(notebook, dst)


if __name__ == "__main__":
    print(notebooks())
