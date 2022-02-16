import os
from pathlib import Path

import papermill as pm

if __name__ == "__main__":
    base = Path(__file__).parent.parent / "notebooks"
    for notebook in (_ for _ in base.glob("**/*.ipynb") if "checkpoints" not in str(_)):
        print(f"Working on {str(notebook)}")
        pm.execute_notebook(notebook, notebook)
