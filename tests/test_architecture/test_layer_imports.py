"""Reject new imports that point the wrong way between layers.

Known violations are listed explicitly. Adding one fails this test until the
audit in ``docs/development/publication_readiness.md`` is updated on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "neurosetta"

# layer -> layers it must not import
FORBIDDEN = {
    "utils": frozenset({"ops", "api", "gui", "analysis", "io"}),
    "core": frozenset({"ops", "api", "gui", "analysis", "io"}),
    "ops": frozenset({"gui"}),
}

# (path relative to src/neurosetta, imported layer). Do not extend casually.
ALLOW = frozenset(
    {
        ("utils/metrics/descriptors.py", "api"),
        ("utils/metrics/descriptors.py", "ops"),
        ("core/synapses.py", "ops"),
        ("core/tree.py", "ops"),
        ("core/tree_helpers.py", "ops"),
        # ops -> api is also a cycle. Recorded here even though the rule
        # above does not forbid api, so a future ban can start from this list.
    }
)


def _imported_layers(path: Path) -> set[str]:
    """Return NeuRosetta layer names imported by *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = ("neurosetta", *path.relative_to(ROOT).parent.parts)
    found: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    continue
                base = package[: len(package) - node.level]
                modules.append(".".join([*base, node.module or ""]).strip("."))
            elif node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "neurosetta":
                found.add(parts[1])
    return found


def test_layers_do_not_grow_new_upward_imports():
    violations: list[str] = []
    for layer, banned in FORBIDDEN.items():
        for path in sorted((ROOT / layer).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for imported in _imported_layers(path) & banned:
                if (rel, imported) in ALLOW:
                    continue
                violations.append(f"{rel} imports {imported}")
    assert not violations, "layer import violations:\n" + "\n".join(violations)


def test_metric_registry_does_not_import_ops():
    """Aliases live in ``neurosetta.api_names``, not in ``ops``."""
    source = (ROOT / "utils" / "metrics" / "registry.py").read_text(encoding="utf-8")
    assert "ops._doc_helpers" not in source
    assert "api_names" in source
