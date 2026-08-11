# Installation

This tutorial covers installing NeuRosetta and verifying that the package
imports correctly.

## Why conda?

NeuRosetta depends on [graph-tool](https://graph-tool.skewed.de/), which is
distributed on **conda-forge** and is not available from PyPI. The recommended
install path is therefore conda/mamba plus an editable install from source.

NeuRosetta is **not yet published on PyPI or conda-forge**.

Supported Python versions: **3.11** and **3.12**.

## Install from source

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
(or Mambaforge). Optionally install `mamba` for faster solves:

```bash
conda install -n base -c conda-forge mamba -y
```

Clone the repository and create the environment (installs NeuRosetta in editable
mode):

```bash
git clone https://github.com/NikDrummond/NeuRosetta.git
cd NeuRosetta
mamba env create -f environment.yml
mamba activate nr
```

Or, with dependencies already present in an activated environment:

```bash
python -m pip install -e ".[dev,docs]"
```

Once the package is published, you will be able to install with:

```bash
# PyPI (planned)
python -m pip install NeuRosetta

# conda-forge (planned)
mamba create -n nr -c conda-forge neurosetta
mamba activate nr
```

## Verify the install

```python
import NeuRosetta as nr

print(nr.__version__)
print(nr.Tree, nr.Forest, nr.import_swc)
```

You should see the package version (currently `0.1.0`) and the public API
symbols.

## Launch the GUI

From the activated environment:

```bash
run_neuro_GUI
```

or:

```python
import NeuRosetta as nr
nr.start_GUI()
```

## Build the documentation locally

```bash
cd docs
make html
```

Open `_build/html/index.html` in a browser.

## Next steps

Continue with {doc}`02_io` to load SWC morphologies from the bundled sample
data under `docs/data/`.
