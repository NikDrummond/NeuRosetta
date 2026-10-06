"""NeuRosetta Workspace (``.nrw``) archive I/O.

A Workspace archive is a ZIP file containing:

* ``manifest.json`` — versioned index of contents
* ``forest/tNNNNNN.nr`` — Trees via existing ``.nr`` persistence
* ``artifacts/aNNNNNN.*`` — named analysis results

Public entry points: :func:`save_workspace`, :func:`load_workspace`,
:func:`inspect_workspace`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.metadata import set_core_meta
from .nr_utils import load as load_nr
from .nr_utils import save as save_nr
from .workspace_errors import (
    WorkspaceFormatError,
    WorkspaceIntegrityError,
    WorkspaceVersionError,
)
from .workspace_serializers import (
    SUPPORTED_RESULT_TYPES,
    serializer_for,
    serializer_for_type,
    validate_json_compatible,
)

if TYPE_CHECKING:
    from ..api.workspace_class import Workspace

WORKSPACE_FORMAT = "neurosetta-workspace"
WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_EXTENSION = ".nrw"

_MANIFEST_NAME = "manifest.json"
_FOREST_DIR = "forest"
_ARTIFACTS_DIR = "artifacts"


def _package_version() -> str:
    from neurosetta import __version__

    return __version__


def _normalize_workspace_path(path: str | Path) -> Path:
    """Ensure the destination uses the ``.nrw`` extension."""
    p = Path(path)
    if p.suffix.lower() != WORKSPACE_EXTENSION:
        p = p.with_suffix(WORKSPACE_EXTENSION)
    return p


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_ref(index: int) -> str:
    return f"t{index:06d}"


def _artifact_ref(index: int) -> str:
    return f"a{index:06d}"


def _validate_archive_member_path(member: str) -> str:
    """Reject absolute paths and path traversal in ZIP member names."""
    if not member or member.endswith("/"):
        raise WorkspaceFormatError(f"invalid archive member path: {member!r}")
    # ZIP uses forward slashes; reject OS separators and traversal.
    normalized = member.replace("\\", "/")
    if normalized != member:
        raise WorkspaceFormatError(f"archive member path must use '/': {member!r}")
    if normalized.startswith("/") or normalized.startswith("../") or "/../" in f"/{normalized}/":
        raise WorkspaceFormatError(f"unsafe archive member path: {member!r}")
    parts = Path(normalized).parts
    if any(part in ("", "..") for part in parts):
        raise WorkspaceFormatError(f"unsafe archive member path: {member!r}")
    if Path(normalized).is_absolute():
        raise WorkspaceFormatError(f"absolute archive member path is not allowed: {member!r}")
    return normalized


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkspaceFormatError(f"{label} must be a JSON object, got {type(value).__name__}")
    return value


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema identity / version and required top-level keys."""
    data = dict(manifest)
    fmt = data.get("format")
    if fmt != WORKSPACE_FORMAT:
        raise WorkspaceFormatError(
            f"not a NeuRosetta Workspace archive (format={fmt!r}, expected {WORKSPACE_FORMAT!r})"
        )

    if "schema_version" not in data:
        raise WorkspaceFormatError("manifest missing schema_version")
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise WorkspaceFormatError(f"schema_version must be an int, got {type(version).__name__}")
    if version > WORKSPACE_SCHEMA_VERSION:
        raise WorkspaceVersionError(
            f"Workspace schema_version {version} is newer than supported "
            f"version {WORKSPACE_SCHEMA_VERSION}; upgrade NeuRosetta to load this file"
        )
    if version < 1:
        raise WorkspaceVersionError(f"invalid Workspace schema_version: {version}")
    if version != WORKSPACE_SCHEMA_VERSION:
        # Boundary for future migrations (none yet for v1-only support).
        raise WorkspaceVersionError(
            f"unsupported Workspace schema_version {version} "
            f"(this NeuRosetta build supports {WORKSPACE_SCHEMA_VERSION})"
        )

    _require_mapping(data.get("workspace"), label="workspace")
    forest = _require_mapping(data.get("forest"), label="forest")
    if "members" not in forest or not isinstance(forest["members"], list):
        raise WorkspaceFormatError("forest.members must be a list")
    if "selections" not in data or not isinstance(data["selections"], dict):
        raise WorkspaceFormatError("selections must be a JSON object")
    if "results" not in data or not isinstance(data["results"], dict):
        raise WorkspaceFormatError("results must be a JSON object")
    return data


def _migrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Apply schema migrations. Currently a no-op for schema version 1."""
    return manifest


def _read_manifest_from_zip(zf: zipfile.ZipFile) -> dict[str, Any]:
    try:
        raw = zf.read(_MANIFEST_NAME)
    except KeyError as exc:
        raise WorkspaceFormatError("archive is missing manifest.json") from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceFormatError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise WorkspaceFormatError("manifest.json must contain a JSON object")
    return _migrate_manifest(_validate_manifest_structure(manifest))


def _extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    """Extract a single validated member to *dest* (parent dirs created)."""
    safe = _validate_archive_member_path(member)
    try:
        info = zf.getinfo(safe)
    except KeyError as exc:
        raise WorkspaceIntegrityError(f"archive member missing: {safe}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info, "r") as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _verify_member_hash(zf: zipfile.ZipFile, member: str, expected: str | None) -> None:
    if expected is None:
        return
    safe = _validate_archive_member_path(member)
    try:
        data = zf.read(safe)
    except KeyError as exc:
        raise WorkspaceIntegrityError(f"archive member missing: {safe}") from exc
    actual = _sha256_bytes(data)
    if actual != expected:
        raise WorkspaceIntegrityError(
            f"SHA-256 mismatch for {safe}: expected {expected}, got {actual}"
        )


def _build_manifest(
    *,
    workspace: Workspace,
    forest_members: list[dict[str, Any]],
    selections: dict[str, list[str]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    validate_json_compatible(workspace.metadata, path="metadata")
    ws_block: dict[str, Any] = {
        "name": workspace.name,
        "description": workspace.description,
        "metadata": dict(workspace.metadata),
    }
    validate_json_compatible(ws_block, path="workspace")
    return {
        "format": WORKSPACE_FORMAT,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "neurosetta_version": _package_version(),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workspace": ws_block,
        "forest": {"members": forest_members},
        "selections": selections,
        "results": results,
    }


def save_workspace(workspace: Workspace, path: str | Path) -> Path:
    """Save a :class:`~neurosetta.api.Workspace` to a ``.nrw`` archive.

    The destination is written atomically: contents are staged to a temporary
    file in the same directory and then replaced with :func:`os.replace`.

    Parameters
    ----------
    workspace : Workspace
        Workspace to persist.
    path : str or pathlib.Path
        Output path. If the suffix is not ``.nrw``, ``.nrw`` is appended
        (e.g. ``\"analysis\"`` → ``analysis.nrw``).

    Returns
    -------
    pathlib.Path
        Final archive path.

    Raises
    ------
    TypeError
        If workspace metadata is not JSON-compatible, or a registered result
        has an unsupported type.
    WorkspaceError
        On archive construction failures.

    Notes
    -----
    Individual Trees are stored with the existing ``.nr`` serializer. Saving
    reuses mesh/synapse freeze/rehydrate so the live Forest remains usable.

    Examples
    --------
    >>> path = nr.save_workspace(ws, "analysis.nrw")  # doctest: +SKIP
    """
    from ..api.workspace_class import Workspace as WorkspaceCls

    if not isinstance(workspace, WorkspaceCls):
        raise TypeError(f"expected Workspace, got {type(workspace).__name__}")

    # Fail early on metadata / result types before writing anything.
    validate_json_compatible(workspace.metadata, path="metadata")
    result_serializers: dict[str, Any] = {}
    for result_name in workspace.list_results():
        value = workspace.get_result(result_name)
        try:
            result_serializers[result_name] = serializer_for(value)
        except TypeError as exc:
            raise TypeError(
                f"Workspace result {result_name!r} has unsupported type "
                f"{type(value).__name__}; supported artifact types: "
                f"{', '.join(SUPPORTED_RESULT_TYPES)}"
            ) from exc

    dest = _normalize_workspace_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=f"{WORKSPACE_EXTENSION}.tmp",
        prefix=f".{dest.stem}.",
        dir=dest.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with tempfile.TemporaryDirectory(prefix="nrw-stage-") as stage_dir:
            stage = Path(stage_dir)
            forest_dir = stage / _FOREST_DIR
            artifacts_dir = stage / _ARTIFACTS_DIR
            forest_dir.mkdir()
            artifacts_dir.mkdir()

            id_to_ref: dict[Any, str] = {}
            forest_members: list[dict[str, Any]] = []
            staged_files: list[tuple[Path, str]] = []

            for index, tree in enumerate(workspace.forest):
                ref = _tree_ref(index)
                rel = f"{_FOREST_DIR}/{ref}.nr"
                nr_path = forest_dir / f"{ref}.nr"
                save_nr(tree, nr_path)
                digest = _sha256_file(nr_path)
                forest_members.append({"ref": ref, "path": rel, "sha256": digest})
                id_to_ref[tree.ID] = ref
                staged_files.append((nr_path, rel))

            selections: dict[str, list[str]] = {}
            for sel_name in workspace.list_selections():
                refs: list[str] = []
                for tree_id in workspace.selection_ids(sel_name):
                    try:
                        refs.append(id_to_ref[tree_id])
                    except KeyError as exc:
                        raise KeyError(
                            f"selection {sel_name!r} refers to Tree ID {tree_id!r} "
                            "missing from the Workspace Forest"
                        ) from exc
                selections[sel_name] = refs

            results_manifest: dict[str, dict[str, Any]] = {}
            for art_index, result_name in enumerate(workspace.list_results()):
                ser = result_serializers[result_name]
                ref = _artifact_ref(art_index)
                rel = f"{_ARTIFACTS_DIR}/{ref}{ser.extension}"
                art_path = artifacts_dir / f"{ref}{ser.extension}"
                ser.dump(workspace.get_result(result_name), art_path)
                digest = _sha256_file(art_path)
                results_manifest[result_name] = {
                    "ref": ref,
                    "type": ser.type_id,
                    "path": rel,
                    "sha256": digest,
                }
                staged_files.append((art_path, rel))

            manifest = _build_manifest(
                workspace=workspace,
                forest_members=forest_members,
                selections=selections,
                results=results_manifest,
            )
            manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            )

            with zipfile.ZipFile(
                tmp_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                zf.writestr(_MANIFEST_NAME, manifest_bytes)
                for file_path, arcname in staged_files:
                    zf.write(file_path, arcname=arcname)

            # Sanity-check the archive before replacing the destination.
            with zipfile.ZipFile(tmp_path, mode="r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise WorkspaceIntegrityError(f"created archive failed CRC check: {bad}")
                _read_manifest_from_zip(zf)

        os.replace(tmp_path, dest)
        return dest
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_workspace(path: str | Path, *, verify: bool = True) -> Workspace:
    """Load a :class:`~neurosetta.api.Workspace` from a ``.nrw`` archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.nrw`` file.
    verify : bool, optional
        When True (default), verify SHA-256 digests recorded in the manifest.

    Returns
    -------
    Workspace
        Reconstructed workspace with Forest, selections, and results.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    WorkspaceFormatError
        If the file is not a valid Workspace archive.
    WorkspaceVersionError
        If the schema version is unsupported.
    WorkspaceIntegrityError
        If a referenced member is missing or fails hash verification.
    """
    from ..api import Forest
    from ..api.workspace_class import Workspace

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workspace archive not found: {p}")

    try:
        zf_cm = zipfile.ZipFile(p, mode="r")
    except zipfile.BadZipFile as exc:
        raise WorkspaceFormatError(f"not a valid ZIP / .nrw archive: {p}") from exc

    with zf_cm as zf:
        manifest = _read_manifest_from_zip(zf)
        names = set(zf.namelist())

        forest_members = manifest["forest"]["members"]
        trees = []
        ref_to_tree: dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="nrw-load-") as extract_dir:
            extract_root = Path(extract_dir)
            for entry in forest_members:
                if not isinstance(entry, dict):
                    raise WorkspaceFormatError("forest.members entries must be objects")
                ref = entry.get("ref")
                member_path = entry.get("path")
                if not isinstance(ref, str) or not isinstance(member_path, str):
                    raise WorkspaceFormatError("forest member requires string ref and path")
                safe = _validate_archive_member_path(member_path)
                if safe not in names:
                    raise WorkspaceIntegrityError(f"referenced Tree member missing: {safe}")
                if verify:
                    _verify_member_hash(zf, safe, entry.get("sha256"))

                dest = extract_root / safe
                _extract_member(zf, safe, dest)
                tree = load_nr(dest)
                # Do not leave file_path pointing at a deleted temp extract path.
                set_core_meta(tree.metadata, "file_path", str(p.resolve()))
                trees.append(tree)
                ref_to_tree[ref] = tree

            results: dict[str, Any] = {}
            for result_name, meta in manifest["results"].items():
                if not isinstance(meta, dict):
                    raise WorkspaceFormatError(f"result {result_name!r} metadata must be an object")
                member_path = meta.get("path")
                type_id = meta.get("type")
                if not isinstance(member_path, str) or not isinstance(type_id, str):
                    raise WorkspaceFormatError(
                        f"result {result_name!r} requires string path and type"
                    )
                safe = _validate_archive_member_path(member_path)
                if safe not in names:
                    raise WorkspaceIntegrityError(
                        f"referenced artifact missing for {result_name!r}: {safe}"
                    )
                if verify:
                    _verify_member_hash(zf, safe, meta.get("sha256"))
                dest = extract_root / safe
                _extract_member(zf, safe, dest)
                ser = serializer_for_type(type_id)
                results[result_name] = ser.load(dest)

        forest = Forest(trees)  # type: ignore[no-untyped-call]
        ws_meta = manifest["workspace"]
        workspace = Workspace(
            forest,
            name=ws_meta.get("name"),
            description=ws_meta.get("description"),
            metadata=ws_meta.get("metadata") or {},
        )

        for sel_name, refs in manifest["selections"].items():
            if not isinstance(refs, list):
                raise WorkspaceFormatError(f"selection {sel_name!r} must be a list of refs")
            ids = []
            for ref in refs:
                if not isinstance(ref, str):
                    raise WorkspaceFormatError(
                        f"selection {sel_name!r} contains non-string ref {ref!r}"
                    )
                try:
                    ids.append(ref_to_tree[ref].ID)
                except KeyError as exc:
                    raise WorkspaceIntegrityError(
                        f"selection {sel_name!r} references unknown Tree ref {ref!r}"
                    ) from exc
            workspace.add_selection(sel_name, ids)

        for result_name, value in results.items():
            workspace.add_result(result_name, value)

        return workspace


def inspect_workspace(path: str | Path, *, verify: bool = False) -> dict[str, Any]:
    """Read Workspace archive metadata without loading Tree graphs.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.nrw`` file.
    verify : bool, optional
        When True, verify SHA-256 digests for all referenced members.
        By default False (manifest-only inspection).

    Returns
    -------
    dict
        Summary with keys ``name``, ``description``, ``schema_version``,
        ``neurosetta_version``, ``tree_count``, ``selections``, ``results``,
        ``metadata``, and ``created_at`` (when present).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    WorkspaceFormatError
        If the file is not a valid Workspace archive.
    WorkspaceVersionError
        If the schema version is unsupported.
    WorkspaceIntegrityError
        If ``verify=True`` and a member is missing or corrupt.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workspace archive not found: {p}")

    try:
        zf_cm = zipfile.ZipFile(p, mode="r")
    except zipfile.BadZipFile as exc:
        raise WorkspaceFormatError(f"not a valid ZIP / .nrw archive: {p}") from exc

    with zf_cm as zf:
        manifest = _read_manifest_from_zip(zf)
        names = set(zf.namelist())

        if verify:
            for entry in manifest["forest"]["members"]:
                if not isinstance(entry, dict):
                    raise WorkspaceFormatError("forest.members entries must be objects")
                member_path = entry.get("path")
                if not isinstance(member_path, str):
                    raise WorkspaceFormatError("forest member path must be a string")
                if member_path not in names:
                    raise WorkspaceIntegrityError(f"referenced Tree member missing: {member_path}")
                _verify_member_hash(zf, member_path, entry.get("sha256"))
            for result_name, meta in manifest["results"].items():
                if not isinstance(meta, dict):
                    raise WorkspaceFormatError(f"result {result_name!r} metadata must be an object")
                member_path = meta.get("path")
                if not isinstance(member_path, str):
                    raise WorkspaceFormatError(f"result {result_name!r} path must be a string")
                if member_path not in names:
                    raise WorkspaceIntegrityError(
                        f"referenced artifact missing for {result_name!r}: {member_path}"
                    )
                _verify_member_hash(zf, member_path, meta.get("sha256"))

        selections = {
            name: len(refs) if isinstance(refs, list) else refs
            for name, refs in manifest["selections"].items()
        }
        results = {
            name: (meta.get("type") if isinstance(meta, dict) else meta)
            for name, meta in manifest["results"].items()
        }
        ws = manifest["workspace"]
        return {
            "name": ws.get("name"),
            "description": ws.get("description"),
            "metadata": dict(ws.get("metadata") or {}),
            "schema_version": manifest["schema_version"],
            "neurosetta_version": manifest.get("neurosetta_version"),
            "created_at": manifest.get("created_at"),
            "tree_count": len(manifest["forest"]["members"]),
            "selections": selections,
            "results": results,
        }


__all__ = [
    "WORKSPACE_EXTENSION",
    "WORKSPACE_FORMAT",
    "WORKSPACE_SCHEMA_VERSION",
    "inspect_workspace",
    "load_workspace",
    "save_workspace",
]
