"""Execute docs notebooks to refresh committed outputs (for myst-nb)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def execute_notebook(path: Path) -> None:
    print(f"Executing {path.name} ...")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent.parent.parent)}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"  wrote {path.name}")


def main() -> None:
    if not NOTEBOOKS:
        raise SystemExit(f"No notebooks found in {NOTEBOOK_DIR}")
    for notebook in NOTEBOOKS:
        execute_notebook(notebook)


if __name__ == "__main__":
    main()
