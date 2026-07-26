# NeuRosetta documentation

**NeuRosetta** is a Python toolbox for morphological analysis of EM neuron
reconstructions. It centres on `Tree` and `Forest` objects built on
[graph-tool](https://graph-tool.skewed.de/), with SWC/NR/mesh I/O, plotting,
neuropil surface reconstruction, and an optional GUI.

```{toctree}
:maxdepth: 2
:caption: Getting started

tutorials/01_installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/02_io
tutorials/03_tree_basics
tutorials/04_tree_surgery
tutorials/05_plotting
tutorials/06_forests
tutorials/07_meshes_neuropils
tutorials/08_surface_reconstruction
tutorials/09_gui
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/units
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

## Quick example

```python
import NeuRosetta as nr

tree = nr.import_swc("docs/data/1.swc")
print(tree.count_nodes(), tree.count_leaves(), tree.get_total_cable_length())
tree.show_2d()
```

## License

NeuRosetta is released under the **LGPL-3.0-only** license.
