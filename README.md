# NeuRosetta

### Create a usable environment (quick version while I'm re-building)

If you have a CUDA enable GPU, make sure CUDA is usable!

- make sure you have miniconda available. install instructions can be found [here](https://www.anaconda.com/docs/getting-started/miniconda/install)
- I've used `mamba` for faster install, so run `conda install -n base -c conda-forge mamba -y` if you do NOT have mamba
- Build your environment, i've called this one `nr`:

```bash
mamba create -n nr python=3.11 -y \
  numpy pandas scipy trimesh vedo gudhi ipython \
  -c conda-forge \
  graph-tool
```

This creates an environment with `numpy`, `pandas`, `scipy`, `trimesh`, `vedo`, `gudhi`, and `graph-tool`.

 - Activate the environment:

 ```bash
 conda activate nr
```

- install Jax with GPU using pip:

```bash
python3 -m pip install --upgrade
python3 -m pip install "jax[cuda13]"
```

- ALTERNATIVE Jax without a GPU:

```bash
python3 -m pip install --upgrade
python3 -m pip install jax
```

### Running tests

from root folder:
```bash
python3 -m pytest
``

