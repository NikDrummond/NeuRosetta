"""Execute tutorial notebooks to refresh committed outputs (for myst-nb)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOCS_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIRS = (
    DOCS_DIR / "getting_started",
    DOCS_DIR / "tutorials",
    DOCS_DIR / "development",
)
NOTEBOOKS = sorted(
    notebook for notebook_dir in NOTEBOOK_DIRS for notebook in notebook_dir.glob("*.ipynb")
)


def execute_notebook(path: Path) -> None:
    print(f"Executing {path.relative_to(DOCS_DIR.parent)} ...")
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
        dirs = ", ".join(str(d) for d in NOTEBOOK_DIRS)
        raise SystemExit(f"No notebooks found in {dirs}")
    for notebook in NOTEBOOKS:
        execute_notebook(notebook)


if __name__ == "__main__":
    main()
