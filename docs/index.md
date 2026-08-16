# NeuRosetta documentation

**NeuRosetta** is a Python toolbox for morphological analysis of EM neuron
reconstructions. It centres on `Tree` and `Forest` objects built on
[graph-tool](https://graph-tool.skewed.de/), with SWC/NR/mesh I/O, plotting,
neuropil surface reconstruction, and an optional GUI.

**New here?** Start with {doc}`getting_started/overview` for the mental model,
then {doc}`getting_started/installation` and {doc}`getting_started/example_data`.
Contributors should read {doc}`development/architecture` and
{doc}`development/extending_nr`.

```{tip}
As both ADHD and Dyslexia are problems for a not insignificant proportion of the population (including myself), we have tried to help this out a little. Going though pages and pages of documentation can be a hassle.

As such, if you have a look in the top right corner next to the buttons to switch between system/light/dark modes, there are an additional two buttons. 

The first lets you switch between your system font, [Atkinson Hyperlegeble](https://en.wikipedia.org/wiki/Atkinson_Hyperlegible) and [OpenDyslexic](https://en.wikipedia.org/wiki/OpenDyslexic). These fonts aer supposed to make it easie for those with Dyslexia.

The second button switches to a [Bionic Reading](https://bionic-reading.com/br-about/) mode which is supposed to help with attention (the jury is out with this one though, to be fair, but why not?).
```

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
PySide6, although available from PyPI needs to come from conda-forge. Also, it is only needed for the GUI. In bash, with you virtual environment created and activated, you can install NeuRosetta:

```bash
pip install "git+https://github.com/NikDrummond/NeuRosetta.git"
```

## Example Data

NeuRosetta ships bundled FlyWire morphologies under `docs/data/`. See
{doc}`reference/example_data` for file layout and citations.

```python
import neurosetta as nr

forest = nr.load_example_data()
forest
```
You can get the path to the directorry the example data is stored in using:

```python
nr.example_data_dir()
```

## Starting the GUI

You can start the GUI from python using:

```python
import neurosetta as nr
nr.start_GUI()
```
or from the command line using:
```bash
run_neuo_GUI
```
You can load the diectory with the example data in, or any other neuron morphology files you have. You can read moer about the GUI in {doc}`tutorials/gui`.

## Documentation Overview

```{toctree}
:maxdepth: 1
:caption: Getting started

getting_started/overview
getting_started/roadmap
getting_started/installation
getting_started/example_data
getting_started/io
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/tree_basics
tutorials/forests
tutorials/plotting
tutorials/tree_surgery
tutorials/gui
```

```{toctree}
:maxdepth: 1
:caption: Development guide

development/architecture
development/extending_nr
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/example_data
reference/units
reference/configuration
reference/metrics
```

```{toctree}
:maxdepth: 1
:caption: API

api/index
```

## License

NeuRosetta is released under the **LGPL-3.0-only** license.

## Citing NeuRosetta

If you use NeuRosetta for any analysis, please cite the paper that first intoduced it:

Population Morphology Implies a Common Developmental Blueprint for Drosophila Motion Detectors
Nikolas Drummond, Arthur Zhao, Alexander Borst
bioRxiv 2025.11.15.688637; [doi] (https://doi.org/10.1101/2025.11.15.688637 )

## References

<a id="ref1"></a>
1. Dorkenwald et al., 2024. Neuronal wiring diagram of an adult brain. [doi](https://doi.org/10.1038/s41586-024-07558-y)

<a id="ref2"></a>
2. Schlegel et al., 2024. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. [doi](https://doi.org/10.1038/s41586-024-07686-5)

<a id="ref3"></a>
3. Zheng et al., 2018. A Complete Electron Microscopy Volume of the Brain of Adult *Drosophila melanogaster*. [doi](https://doi.org/10.1016/j.cell.2018.06.019)
