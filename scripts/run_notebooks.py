"""Execute the numbered forecasting notebooks in order from the repository root."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))


def main() -> None:
    for path in NOTEBOOKS:
        if path.name == "11_uncertainty.ipynb":
            print("Building and scoring final-holdout forecasts", flush=True)
            run_path(str(ROOT / "scripts" / "build_final_holdout_forecasts.py"), run_name="__main__")
            run_path(str(ROOT / "scripts" / "evaluate_holdout.py"), run_name="__main__")
        print(f"Running {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        try:
            NotebookClient(
                notebook,
                timeout=1800,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        except Exception:
            nbformat.write(notebook, path)
            raise
        nbformat.write(notebook, path)
        print(f"Passed {path.name}", flush=True)



if __name__ == "__main__":
    main()
