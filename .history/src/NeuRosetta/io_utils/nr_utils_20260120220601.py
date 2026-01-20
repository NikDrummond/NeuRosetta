from pathlib import Path
from graph_tool.all import load_graph
from typing import TYPE_CHECKING

from ..errors.errors import _check_internal_property
from ..core import _Tree

if TYPE_CHECKING:
    from ..classes import Tree


def _bind_core(tree: _Tree):
    """
    Make sure ID and metadata are bound to the graph
    """

    # check if we have bound ID
    try:
        _check_internal_property(tree.graph, "ID")
    except:
        tree.graph.gp["ID"] = tree.graph.new_gp("int", str(tree.ID))

    # check if we have bound metadata
    try:
        _check_internal_property(tree.graph, "metadata")
    except:
        tree.graph.gp["metadata"] = tree.graph.new_gp("object", tree.metadata)


def save(tree: _Tree, fpath: str | Path | None = None) -> Path:
    """
    Save tree as a .nr file and return the written file path.

    If fpath is None: saves to current working directory.
    If fpath is a directory: saves into that directory as <ID>.nr
    If fpath is a file path: saves exactly there (must end with .nr, or we can enforce/append).
    """
    _bind_core(tree)

    if fpath is None:
        out_path = Path.cwd() / f"{tree.ID}.nr"
    else:
        p = Path(fpath)

        # If user gave a directory, write <dir>/<ID>.nr
        if p.exists() and p.is_dir():
            out_path = p / f"{tree.ID}.nr"
        else:
            # Treat as a file path. If it's a directory-like path that doesn't exist yet,
            # you need a policy. The safest: if it has a suffix, assume it's a file.
            # If no suffix, assume it's a directory path and create it.
            if p.suffix:
                out_path = p
            else:
                p.mkdir(parents=True, exist_ok=True)
                out_path = p / f"{tree.ID}.nr"

    tree.graph.save(str(out_path), fmt="gt")
    return out_path


def load(fpath: str | Path, *, tree_id: int | None = None) -> "Tree":
    """
    Load a Tree from a .nr file.

    If fpath is a directory, we need to know which file to load:
      - if tree_id is given, loads <dir>/<tree_id>.nr
      - if tree_id is None and there is exactly one .nr file, loads it
      - otherwise raises an error
    If fpath is a file, loads it directly.
    """
    from ..classes import Tree

    p = Path(fpath)

    if p.is_dir():
        if tree_id is not None:
            p = p / f"{tree_id}.nr"
        else:
            candidates = sorted(p.glob("*.nr"))
            if len(candidates) == 1:
                p = candidates[0]
            elif len(candidates) == 0:
                raise FileNotFoundError(f"No .nr files found in directory: {p}")
            else:
                raise ValueError(
                    f"Multiple .nr files found in {p}; pass tree_id=... or a file path."
                )

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    g = load_graph(str(p), fmt="gt")
    return Tree(ID=g.gp["ID"], metadata=g.gp["metadata"], graph=g)
