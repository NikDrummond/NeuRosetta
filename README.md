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

## Install (recommended: conda / mamba)

```bash
# optional: faster solver
conda install -n base -c conda-forge mamba -y

mamba create -n nr -c conda-forge python=3.11 \
  numpy pandas scipy matplotlib tqdm trimesh vedo \
  scikit-learn scikit-image pyside6 graph-tool \
  pip hatchling
mamba activate nr
python -m pip install NeuRosetta
```

Once published on conda-forge:

```bash
mamba create -n nr -c conda-forge neurosetta
mamba activate nr
```

### Editable / development install

From a clone of this repository:

```bash
mamba env create -f environment.yml
mamba activate nr
# environment.yml already installs the package in editable mode
```

Or, in an existing env with dependencies available:

```bash
python -m pip install -e ".[dev,docs]"
```

## Quick start

```python
import NeuRosetta as nr

tree = nr.import_swc("path/to/1.swc")
print(tree.count_nodes(), tree.get_cable_length())
tree.show_2d()
```

Launch the GUI:

```bash
run_neuro_GUI
```

or:

```python
import NeuRosetta as nr
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
