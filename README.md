# NeuRosetta

Tools for morphological analysis of EM neuron reconstructions.

NeuRosetta provides `Tree` and `Forest` APIs for neuron morphologies, SWC/NR/mesh
I/O, 2D/3D plotting, neuropil surface reconstruction, and a PySide6 GUI.

**Documentation:** [https://nikdrummond.github.io/NeuRosetta/](https://nikdrummond.github.io/NeuRosetta/)

**License:** LGPL-3.0-only

## Requirements

- Python 3.11 or 3.12
- [conda-forge](https://conda-forge.org/) packages, especially **graph-tool**
  (not available on PyPI)

NeuRosetta is **not yet published on PyPI or conda-forge**. Install from source
for now (see below).

## Install (recommended: conda / mamba)

Clone the repository and create the environment from `environment.yml` (installs
the package in editable mode):

```bash
git clone https://github.com/NikDrummond/NeuRosetta.git
cd NeuRosetta
mamba env create -f environment.yml
mamba activate nr
```

Or, in an existing env with dependencies already available:

```bash
python -m pip install -e ".[dev,docs]"
```

Once published:

```bash
# PyPI (planned)
python -m pip install neurosetta

# conda-forge (planned)
mamba create -n nr -c conda-forge neurosetta
mamba activate nr
```

## Quick start

```python
import neurosetta as nr

tree = nr.import_swc("path/to/1.swc")
print(tree.count_nodes(), tree.get_total_cable_length())
tree.show_2d()
```

Launch the GUI:

```bash
run_neuro_GUI
```

or:

```python
import neurosetta as nr
nr.start_GUI()
```

## Build documentation locally

```bash
mamba activate nr
python -m pip install -e ".[docs]"   # if docs extras are not already installed
cd docs
make html
# open _build/html/index.html
```

## Run tests

From the repository root:

```bash
python -m pytest
```

## Citation / links

- Source: [https://github.com/NikDrummond/NeuRosetta](https://github.com/NikDrummond/NeuRosetta)
- Docs: [https://nikdrummond.github.io/NeuRosetta/](https://nikdrummond.github.io/NeuRosetta/)
