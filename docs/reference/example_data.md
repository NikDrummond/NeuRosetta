# Bundled example data

Example morphologies from the [FAFB-FlyWire](https://codex.flywire.ai/?dataset=fafb)
dataset ship with the repository under `docs/data/`.

| File stem | FlyWire segment ID | Notes |
|-----------|-------------------|-------|
| `720575940596125868` | `720575940596125868` | T5c optic-lobe neuron |
| `720575940599459782` | `720575940599459782` | T5b optic-lobe neuron |
| `720575940599704006` | `720575940599704006` | T5a optic-lobe neuron |
| `720575940599729862` | `720575940599729862` | T5a optic-lobe neuron |

## Layout

| Subdirectory | Format | Load with |
|--------------|--------|-----------|
| `nr/` | Native NeuRosetta `.nr` files | {func}`~neurosetta.load_example_data` or {func}`~neurosetta.load` |
| `swc/` | SWC morphologies | {func}`~neurosetta.import_swc` on {func}`~neurosetta.example_data_dir` |

File stems are FlyWire segment IDs. {func}`~neurosetta.load_example_data` loads from
`nr/` by default.

```python
import neurosetta as nr

forest = nr.load_example_data()
tree = nr.load_example_data(720575940596125868)

swc_dir = nr.example_data_dir(format="swc")
tree = nr.import_swc(swc_dir / "720575940596125868.swc")
```

See {doc}`../getting_started/example_data` for a runnable walkthrough.

## Citations

The example morphologies come from the FlyWire ecosystem:

1. Dorkenwald et al., 2024. Neuronal wiring diagram of an adult brain. [doi](https://doi.org/10.1038/s41586-024-07558-y)
2. Schlegel et al., 2024. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. [doi](https://doi.org/10.1038/s41586-024-07686-5)
3. Zheng et al., 2018. A Complete Electron Microscopy Volume of the Brain of Adult *Drosophila melanogaster*. [doi](https://doi.org/10.1016/j.cell.2018.06.019)
