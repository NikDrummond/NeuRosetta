# GUI walkthrough

NeuRosetta ships a PySide6 + vedo/VTK desktop viewer for interactive inspection
and light editing of morphologies. It is a front-end over the same ops as the
library — reroot and subtree extraction call
{meth}`~neurosetta.api.Tree.get_rerooted_tree` and
{meth}`~neurosetta.api.Tree.subtree_mask_from_root` /
{meth}`~neurosetta.api.Tree.get_subtree` under the hood.

Use the GUI when you want to **see** a root choice or subtree boundary before
committing it. Use the Python API ({doc}`tree_surgery`, {doc}`forests`) for
anything batch or reproducible.

## Requirements

The GUI needs `pyside6` from **conda-forge** (not pip) plus a working display.
See {doc}`../getting_started/installation` if `run_neuro_GUI` fails to open a
window (WSL / SSH forwarding notes are there too).

## Launch

From an activated environment with neurosetta installed:

```bash
run_neuro_GUI
```

or from Python:

```python
import neurosetta as nr
nr.start_GUI()
```

Both enter the same Qt main window.

## Layout

| Region | What it is |
|--------|------------|
| **Centre** | vedo/VTK 3D view — rotate, zoom, pick points |
| **Right side panel** | Navigation, mesh overlay, edit tools, save / flag |
| **Menu bar** | File, Tools, Viewer (units, scale bar, neuron colour) |
| **Log dock** | Optional Tools → Show Log |

### Side panel controls

| Control | Purpose |
|---------|---------|
| Jump to File… | Skip to a neuron by 1-based index or stem name (folder loads) |
| Set Mesh Path… / Show Mesh | Overlay a mesh (e.g. neuropil) from a directory |
| Show Subtree | Toggle visualisation of the current subtree mask |
| Flag Neuron | Sets `metadata["Flag"]` on the active tree |
| Reroot Neuron | Enter pick mode → choose new root |
| Subtree from Point | Enter pick mode → mask a subtree from a node |
| AutoSave | Write on navigation / edit when enabled |
| Save / Save As… | Persist the current tree |

### Menus (same actions)

- **File** — Load File…, Load Folder…, Save / Save As…, Exit
- **Tools** — Show Log, Show Current Subtree, Reroot Neuron, Subtree from Point
- **Viewer** — Units (nm / µm), Toggle Scale Bar, Set Scale Bar Size…, Set Neuron Colour…

## Typical session

### 1. Load morphologies

**File → Load File…** for a single `.swc` / `.nr`, or **Load Folder…** for a
directory. Folder mode keeps a counter (`3/10` style) so you can step through a
dataset.

Prefer `.nr` once you have edited something — native files keep bound
properties and metadata (including `Flag` / `isReduced`). SWC is fine for first
import; convert and reopen as `.nr` if edges look wrong (known SWC tracking
quirk in the viewer).

Example data for a first look:

```python
import neurosetta as nr
print(nr.example_data_dir())       # .nr examples
print(nr.example_data_dir("swc"))  # matching .swc
```

Point **Load Folder…** at either directory.

### 2. Navigate

- Browse with the folder counter / next-previous behaviour of the loader
- **Jump to File…** — index (`1…N`) or filename stem without extension
- Rotate / zoom in the VTK view as usual for a vedo plotter

### 3. Inspect

- **Viewer → Units** — switch display between nanometres and micrometres
- **Toggle Scale Bar** / **Set Scale Bar Size…** — overlay for screenshots
- **Set Neuron Colour…** — pick a solid colour for the active neuron
- **Show Mesh** — after **Set Mesh Path…**, overlay a surface for spatial context

### 4. Edit — reroot

Same conceptual op as {doc}`tree_surgery`:

1. Click **Reroot Neuron** (side panel or Tools menu)
2. Pick **exactly one** point on the neuron in the 3D view
3. Confirm with **Set as Root** (button appears on the view)
4. The tree is rerooted in place; the view refreshes

Vertex indices are rebuilt after reroot (root is typically index `0`) — same as
the library API.

### 5. Edit — subtree from a point

1. Click **Subtree from Point**
2. Pick one point — that node becomes the subtree root
3. Confirm with **Define Subtree**
4. Optionally enable **Show Subtree** to highlight the mask
5. Extraction uses the same mask → `get_subtree` path as scripting

```{warning}
Subtree extraction is **in place**. Save a copy first (or work on a duplicate
file) if you still need the full arbor.
```

If the 3D view looks broken after extraction, save as `.nr` and reopen, or
rely on the library's `tree.plot3d.rebuild()` in a script.

### 6. Flag and save

- **Flag Neuron** — toggles the protected `Flag` metadata entry (handy for
  manual QC pass/fail while stepping a folder)
- **Save…** / **Save As…** — write `.nr` (preferred) or SWC
- **AutoSave** — persist automatically when you navigate / finish an edit

## Relation to the library API

| GUI action | Library counterpart |
|------------|---------------------|
| Load SWC / NR | `import_swc`, `load` / `load_example_data` |
| Save NR | `save` / `Tree.save_tree` |
| Reroot | `Tree.get_rerooted_tree(..., inplace=True)` |
| Subtree from point | `subtree_mask_from_root` → `get_subtree` |
| Show subtree | mask + `build_3d_subtree` / plot helpers |
| Flag | `Tree.set_flag` / `metadata["Flag"]` |
| Mesh overlay | `import_mesh` + vedo actors |
| Units / scale bar | display-only; units on disk via `set_units` / `convert_units` |

Scripting remains the better path for batch analysis; the GUI is for
exploration and single-neuron curation.

## Known caveats

- SWC edge tracking in the GUI can be unreliable; if edges look wrong, convert
  to `.nr` after import and reopen.
- Subtree extraction has the same plotting caveat as the library API —
  save/reload or rebuild 3D actors if visualisation breaks.
- Needs a real display: headless CI / pure SSH without X/Wayland forwarding
  will not show the window (see installation notes).

## Next steps

- Structural edits in code: {doc}`tree_surgery`
- Plotting without the Qt app: {doc}`plotting`
- Full parameter docs: {doc}`../api/gui` and {doc}`../api/index`
