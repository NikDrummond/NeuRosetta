# Installation

This tutorial covers installing NeuRosetta and verifying that the package
imports correctly.

NeuRosetta depends on [graph-tool](https://graph-tool.skewed.de/), which is
distributed on **conda-forge** and is **not available from PyPI**. The
recommended install path is therefore **conda/mamba** for the heavy dependencies,
followed by either an editable install from a local clone or a `pip install`
from GitHub.

NeuRosetta is **not yet published on PyPI or conda-forge**. Supported Python
versions: **3.11** and **3.12** (the bundled `environment.yml` pins 3.11).

## Create a conda/mamba environment

Install [Miniforge](https://github.com/conda-forge/miniforge) (recommended —
includes `mamba`) or [Miniconda](https://docs.anaconda.com/miniconda/). If you
use Miniconda, add `mamba` for faster dependency solves:

```bash
conda install -n base -c conda-forge mamba -y
```

To create a minimal envionment manually you can do:

```bash
mamba create -n nr -c conda-forge python=3.11 graph-tool pyside6
mamba activate nr
```

(pyside6-note)=
```{note}
### PySide6 Installation

`pyside6` is needed in order to run the Neuosetta GUI, but is not technically needed for NeuRosetta in general, so you only need this if you wish to use the GUI. 

`pyside6` should come from conda-forge, not pip, ass PySide6 (Qt 6.5+) needs the system lib `libxcb-cursor0`, but PySide6 on PyPI still dependss on OS X11 libs, so it won't install this.
```

You can switch `nr` to whatever you want the environment to be named. For a working NeuRosetta install you will also need the remaining runtime
dependencies (`numpy`, `scipy`, `pandas`, `matplotlib`, `vedo`, `trimesh`,
`pyside6`, and others). The full list is in `environment.yml` and
`pyproject.toml`.

Altenatively, if you have either cloned the [NeuRosetta](https://github.com/NikDrummond/NeuRosetta.git) repo fom github, downloaded the `environment.yml` file from the repo, or simply copied its contents into your own `.yml` file, you can creat the needed environment with all dependencies as follows (this includes **graph_tool** amoungst other requirements):

```bash
mamba env remove -n nr
mamba env create -f environment.yml
mamba activate nr
```
Similarly, you can clone the repo from github as follows, and use the included `environment.yml` file, which will (unless you modify `environment.yml` create an environment called `nr`)

```bash
git clone https://github.com/NikDrummond/NeuRosetta.git
cd NeuRosetta
mamba env create -f environment.yml
mamba activate nr
```

## Install from GitHub using pip

Once a conda/mamba environment with **graph-tool** is active, you can install
NeuRosetta directly from GitHub without keeping a local clone:

```bash
pip install "git+https://github.com/NikDrummond/NeuRosetta.git"
```

Include development and documentation extras:

```bash
pip install "neurosetta[dev,docs] @ git+https://github.com/NikDrummond/NeuRosetta.git"
```

```{important}
`pip install git+https://...` installs the **Python package only**. It does
**not** install graph-tool. You must create and activate a conda/mamba
environment with graph-tool first, or the import will fail at runtime.
```

To upgrade to the latest commit on a branch:

```bash
pip install --upgrade --force-reinstall "git+https://github.com/NikDrummond/NeuRosetta.git"
```

## Install from source

For day-to-day development, clone the repository and install in **editable**
mode so local changes are picked up immediately.

```bash
git clone https://github.com/NikDrummond/NeuRosetta.git
cd NeuRosetta
mamba env create -f environment.yml   # installs deps + editable package via pip: -e .
mamba activate nr
```

If the conda dependencies are already present in an activated environment, an
editable install alone is enough:

```bash
python -m pip install -e ".[dev,docs]"
```

Once the package is published, you will be able to install with:

```bash
# PyPI (planned)
python -m pip install neurosetta

# conda-forge (planned)
mamba create -n nr -c conda-forge neurosetta
mamba activate nr
```

## Verify

From an activated `nr` environment:

```python
import neurosetta as nr

print(nr.__version__)
print(nr.Tree, nr.Forest, nr.import_swc)
```

You should see the package version (currently `0.1.0`) and the public API
symbols, with no import errors.

Optionally confirm that graph-tool is available:

```python
import graph_tool  # noqa: F401
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

## Optional: documentation and tests

Build the documentation locally:

```bash
mamba activate nr
cd docs
make html
```

Open `_build/html/index.html` in a browser.

Run the test suite from the repository root:

```bash
python -m pytest
```

CI currently runs a smoke subset until the full suite is green:
`test_core.py`, `test_errors.py`, `test_tree_graphs/test_counting.py`.

Lint with Ruff:

```bash
ruff check src tests
ruff format src tests
```

## Windows — use WSL2

**Do not install NeuRosetta in native Windows conda.** graph-tool has no
Windows build on conda-forge. Use **WSL2** with a Linux conda/mamba
installation instead. **Importantl!** NeuRosetta has **not** been tested on windows, so this is not really advisable. 

### Set up WSL2

In **PowerShell (Admin)** on Windows:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then open **Ubuntu** from the Start menu.

### Install inside WSL

Install Miniforge for Linux inside WSL, then follow the Linux install steps
above:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# restart the shell or: source ~/.bashrc

git clone https://github.com/NikDrummond/NeuRosetta.git
cd NeuRosetta
mamba env create -f environment.yml
mamba activate nr
```

```{tip}
Keep the repository on the **Linux filesystem** (for example `~/Code/NeuRosetta`),
not under `/mnt/c/...`. File I/O from WSL to the Windows drive is much slower
and can cause subtle issues with conda and Python tooling.
```

Edit code from Windows using VS Code with the **Remote - WSL** extension,
or open the WSL path directly (`\\wsl$\Ubuntu\home\<user>\Code\NeuRosetta`).

Always run Python from a WSL terminal with `mamba activate nr` — do not mix
a Windows Python install with the WSL conda environment.

### Display and 3D viewers (vedo)

NeuRosetta uses [vedo](https://vedo.embl.es/) (VTK/OpenGL) for interactive 3D
plotting (`tree.show_3d()`, {class}`~neurosetta.ops.plotting.Viewer`, forest
3D views). Opening these **external viewer windows from WSL has been
problematic** in practice:

- On **Windows 11**, WSLg usually forwards GUI windows automatically, but vedo
  pop-up windows can still fail to open, appear blank, or hang on close.
- On **Windows 10** (or without WSLg), you need an X server (for example
  VcXsrv) and must set `DISPLAY` manually:

  ```bash
  export DISPLAY=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}'):0
  ```

- The PySide6 GUI (`run_neuro_GUI`) can hit similar display forwarding issues
  under WSL.

**Workarounds when interactive windows fail:**

- Use **matplotlib** 2D plots (`tree.show_2d()`) — these are generally reliable.
- Render 3D scenes **off-screen** and save to file:

  ```python
  from neurosetta import Viewer

  with Viewer(interactive=False) as viewer:
      viewer.add_neuron(tree)
      viewer.screenshot("neuron.png")
  ```

- Pass `interactive=False` to `show_3d()` where supported, or set
  `offscreen=True` on {class}`~neurosetta.ops.plotting.Viewer`.
- For heavy 3D work, run NeuRosetta on a native **Linux** or **macOS** machine
  if possible.

See {doc}`05_plotting` for more on 2D/3D plotting options.

## Troubleshooting

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: graph_tool` | graph-tool not installed, or wrong env | Activate `nr`; install via conda-forge |
| `pip install neurosetta` fails | Not published on PyPI yet | Install from GitHub or source (above) |
| Import works but 3D viewer hangs | WSL display / OpenGL | Use off-screen screenshots; see WSL section |
| Slow file I/O in WSL | Repo on `/mnt/c/` | Move clone to `~/...` on Linux filesystem |
| Stale install after `git pull` | Non-editable pip install | Re-run `pip install -e ".[dev,docs]"` or reinstall from git |
| `Could not load the Qt platform plugin "xcb"` / `xcb-cursor0` | Missing system X11 lib; PySide6 from pip on a minimal env | `sudo apt install libxcb-cursor0` (and other `libxcb-*` deps if needed); prefer `pyside6` from conda-forge or via `environment.yml` (see {ref}`PySide6 note <pyside6-note>`) |

Remember to activate the environment in every new shell session:

```bash
mamba activate nr
```

## Next steps

Continue with {doc}`02_io` to load SWC morphologies from the bundled sample
data under `docs/data/`.
