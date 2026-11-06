"""Compare a few hand-calculated trees with NAVis or NeuroM if installed.

NeuRosetta does not depend on either package. If the import fails the script
exits 0 and writes a JSON record with status ``not_run``. A missing import is
not a failed comparison.

Run from the repository root::

    python benchmarks/comparisons/compare_metrics.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from neurosetta.testing import straight_chain

OUT = Path(__file__).resolve().parents[1] / "results" / "comparison.json"


def _try(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def main() -> None:
    tree = straight_chain()
    record = {
        "status": "not_run",
        "python": sys.version,
        "platform": platform.platform(),
        "neurosetta_cable": tree.get_total_cable_length(),
        "neurosetta_nodes": tree.count_nodes(),
        "analytical_cable": 3.0,
        "analytical_nodes": 4,
        "navis": _try("navis"),
        "neurom": _try("neurom"),
        "note": (
            "Cable 3 and 4 nodes are the straight reference chain. "
            "NAVis and NeuroM versions are null when those packages are "
            "not installed. No agreement statistic is reported in that case."
        ),
    }
    if record["navis"] is None and record["neurom"] is None:
        record["status"] = "not_run"
    else:
        record["status"] = "imports_only"
        record["note"] = (
            "At least one comparison package imports. This script does not "
            "yet map their section models onto the reference chain. Do not "
            "treat a successful import as numerical agreement."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "navis", "neurom", "neurosetta_cable")}))


if __name__ == "__main__":
    main()
