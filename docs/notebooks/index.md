# Example notebooks

Runnable walkthroughs that complement the markdown {doc}`../tutorials/02_io` and
{doc}`../tutorials/03_tree_basics` tutorials.

```{note}
**Run locally:** start Jupyter from the repository root (or `docs/`) with the
`nr` environment active.

**Run in the browser:** [Launch on Binder](https://mybinder.org/v2/gh/NikDrummond/NeuRosetta/main?filepath=docs%2Fnotebooks%2F01_io_and_tree_basics.ipynb)
(uses `environment.yml` — first start can take a few minutes).
```

Notebook outputs are pre-run and baked into the docs build (`nb_execution_mode =
"off"`). Re-execute notebooks locally before committing if you change code cells:

```bash
mamba activate nr
python docs/notebooks/_execute_notebooks.py
```

```{toctree}
:maxdepth: 1

01_io_and_tree_basics
```
