# Sample data

Example morphologies from the [FAFB-FlyWire](https://codex.flywire.ai/?dataset=fafb)
dataset, used by the NeuRosetta tutorials.

| File stem | FlyWire segment ID | Notes |
|-----------|-------------------|-------|
| `720575940596125868` | `720575940596125868` | T5c optic-lobe neuron |
| `720575940599459782` | `720575940599459782` | T5b optic-lobe neuron |
| `720575940599704006` | `720575940599704006` | T5a optic-lobe neuron |
| `720575940599729862` | `720575940599729862` | T5a optic-lobe neuron |

## Layout

| Subdirectory | Format | Load with |
|--------------|--------|-----------|
| `nr/` | Native NeuRosetta `.nr` files | `load_example_data()` or `load()` |
| `swc/` | SWC morphologies | `import_swc(example_data_dir(format="swc"))` |

File stems are FlyWire segment IDs. `load_example_data()` loads from `nr/` by default.

```python
import neurosetta as nr

forest = nr.load_example_data()
tree = nr.load_example_data(720575940596125868)

swc_dir = nr.example_data_dir(format="swc")
tree = nr.import_swc(swc_dir / "720575940596125868.swc")
```
