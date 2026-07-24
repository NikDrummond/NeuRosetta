# GUI walkthrough

NeuRosetta includes a PySide6 + vedo/VTK desktop viewer for interactive
inspection and light editing of morphologies.

## Launch

From an activated environment with NeuRosetta installed:

```bash
run_neuro_GUI
```

or:

```python
import NeuRosetta as nr
nr.start_GUI()
```

## Typical session

1. **Open files or a folder** — load `.swc` and/or `.nr` morphologies (and
   meshes where supported).
2. **Navigate** — browse neurons in the loaded set; jump to a file when
   working from a directory.
3. **Inspect** — rotate/zoom the 3D view; toggle units (nm / µm) and scale
   bar; adjust neuron colour.
4. **Edit** — reroot from a picked point; extract a subtree from a picked
   point (same conceptual ops as {doc}`04_tree_surgery`).
5. **Overlay** — show a mesh (for example a neuropil) with the neuron.
6. **Save** — write edited trees back to `.nr` (preferred) or SWC.

## Relation to the library API

| GUI action | Library counterpart |
|------------|---------------------|
| Load SWC / NR | `import_swc`, `load` |
| Save NR | `save` / `Tree.save_tree` |
| Reroot | `Tree.get_rerooted_tree` |
| Subtree from point | mask + `get_subtree` |
| Mesh overlay | `import_mesh` + vedo actors |

Scripting remains the better path for batch analysis; the GUI is for
exploration and single-neuron curation.

## Known caveats

- SWC edge tracking in the GUI can be unreliable; if edges look wrong,
  convert to `.nr` after import and reopen.
- Subtree extraction has the same plotting caveat as the library API
  (save/reload if visualisation breaks).

## Next steps

- Browse the {doc}`../api/index` for full parameter documentation.
- For conda-forge packaging status and contribution notes, see the project
  [README](https://github.com/NikDrummond/NeuRosetta).
