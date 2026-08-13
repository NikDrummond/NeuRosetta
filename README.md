# NeuRosetta

[![Tests](https://github.com/NikDrummond/NeuRosetta/actions/workflows/tests.yml/badge.svg)](https://github.com/NikDrummond/NeuRosetta/actions/workflows/tests.yml)
[![Lint](https://github.com/NikDrummond/NeuRosetta/actions/workflows/lint.yml/badge.svg)](https://github.com/NikDrummond/NeuRosetta/actions/workflows/lint.yml)
[![Docs](https://github.com/NikDrummond/NeuRosetta/actions/workflows/docs.yml/badge.svg)](https://nikdrummond.github.io/NeuRosetta/)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](https://www.python.org/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Tools for morphological analysis of EM neuron reconstructions.

NeuRosetta provides `Tree` and `Forest` APIs for neuron morphologies, SWC/NR/mesh
I/O, 2D/3D plotting, neuropil surface reconstruction, and a PySide6 GUI.

**Documentation:** [https://nikdrummond.github.io/NeuRosetta/](https://nikdrummond.github.io/NeuRosetta/)

## Requirements

- Python 3.11 or 3.12
- [conda-forge](https://conda-forge.org/) packages, especially **graph-tool**
  (not available on PyPI)

NeuRosetta is **not yet published on PyPI or conda-forge**. Install from source
for now (see below).

## Quick Start

If you haven't, install [Miniforge](https://github.com/conda-forge/miniforge) (recommended —
includes `mamba`) or [Miniconda](https://docs.anaconda.com/miniconda/). If you
use Miniconda, add `mamba` for faster dependency solves:

```bash
conda install -n base -c conda-forge mamba -y
```

To get up and running quickly, you first need to create a conda/mamba environment with `graph-tool`. Additionally, if you wish to use the GUI, you should grab pyside6 from conda forge, not pip.  

```bash
mamba create -n nr -c conda-forge python=3.11 graph-tool pyside6
mamba activate nr
```

## Example Data

NeuRosetta comes with some example data from the [FlyWire](flywire.ai) dataset. We povide the following neuron skeletons in both `.swc`, [a standad neuron morphology file type](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html), and `.nr`, which is NeuRosettas own file type, which you can read about in the [NeuRosetta documentation](https://nikdrummond.github.io/NeuRosetta/).
The example dataset used come from the following papers: [1](#ref1), [2](#ref2), [3](#ref3).

You can start up NeuRosetta and read in the example files to start playing around like so:

```python
import neurosetta as nr

forest = nr.load_example_data()
forest
```

## The GUI

You can get the path to where the example data is like so:

```bash
nr.example_data_dir()
```
Knowing the path, you can load them into the GUI and have a look. You can stat the GUI in the terminal:

```bash
run_neuro_GUI
```

or from within Python:

```python
import neurosetta as nr
nr.start_GUI()
```

## Features

- **Tree & Forest APIs**: directed tree graphs for single neurons and batched
  collections (`filter`, `apply`, parallel I/O) with flexible and fast subsetting and parallelisation.
- **I/O**: SWC and native `.nr` (graph-tool) import/export; mesh I/O via vedo/trimesh
- **Graph operations**: counts, traversals, subtree extraction, rerooting,
  reduction, cable length, partition asymmetry, degree statistics
- **Tree Geometry**: Geometric analysis of neuron morphology using a custom (fast!) 3-dimensional geometry numba backend.
- **Plotting**: 2D/3D matplotlib and vedo views, dendrograms, interactive `Viewer`
- **Units**: Pint-backed spatial metadata (nm, µm, voxels) with in-place conversion
- **Neuropils & meshes**: surface containers, point-to-surface distances,
  alpha-shape neuropil reconstruction from forest coordinates
- **GUI**: PySide6 desktop app for inspection, rerooting, and subtree picking

## Contributing

Contributions are welcome — bug reports, docs fixes, tests, and new features.

1. **Fork & clone** the repo, then create a conda environment:

   ```bash
   git clone https://github.com/NikDrummond/NeuRosetta.git
   cd NeuRosetta
   mamba env create -f environment.yml
   mamba activate nr
   ```

   Or, if dependencies are already installed:

   ```bash
   python -m pip install -e ".[dev,docs]"
   ```

2. **Make changes** on a feature branch. Keep new tree/forest logic in `ops/`
   (not `utils/` or `core/` directly); see the
   [architecture guide](https://nikdrummond.github.io/NeuRosetta/concepts/architecture.html)
   for layer rules.

3. **Test & lint** before opening a PR:

   ```bash
   python -m pytest
   ruff check src tests
   ruff format src tests
   ```

4. **Open a pull request** against `main` with a short description of the
   change. Link any related
   [issue](https://github.com/NikDrummond/NeuRosetta/issues) if one exists.

Questions or ideas with no code yet? Open an issue first.

## Build documentation locally

```bash
mamba activate nr
pip install "neurosetta[dev,docs] @ git+https://github.com/NikDrummond/NeuRosetta.git"

cd docs
make html
# open _build/html/index.html
```

## Run tests

From the repository root:

```bash
python -m pytest
```
Lint with Ruff:

```bash
ruff check src tests
ruff format src tests
```

## Citation / links

- Source: [https://github.com/NikDrummond/NeuRosetta](https://github.com/NikDrummond/NeuRosetta)
- Docs: [https://nikdrummond.github.io/NeuRosetta/](https://nikdrummond.github.io/NeuRosetta/)

If you use NeuRosetta for any analysis, please cite the paper that first intoduced it:

Population Morphology Implies a Common Developmental Blueprint for Drosophila Motion Detectors
Nikolas Drummond, Arthur Zhao, Alexander Borst
bioRxiv 2025.11.15.688637; doi: https://doi.org/10.1101/2025.11.15.688637 

## References

<a id="ref1"></a>
1. Dorkenwald et al., 2024. Neuronal wiring diagram of an adult brain. [doi](https://doi.org/10.1038/s41586-024-07558-y)

<a id="ref2"></a>
2. Schlegel et al., 2024. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. [doi](https://doi.org/10.1038/s41586-024-07686-5)

<a id="ref3"></a>
3. Zheng et al., 2018. A Complete Electron Microscopy Volume of the Brain of Adult *Drosophila melanogaster*. [doi](https://doi.org/10.1016/j.cell.2018.06.019)