"""API classes for NeuRosetta.

This module provides the public API classes for working with neuron trees,
forests, and neuropil meshes.

Classes
-------
Tree : class
    Main class for single neuron trees.
Forest : class
    Container for multiple Tree objects.
Tree_mesh : class
    Neuron mesh representation.
Neuropil : class
    Single neuropil mesh.
Forest_mesh : class
    Container for multiple neuron meshes.
Neuropils : class
    Container for multiple Neuropil objects.
"""

from .tree_class import Tree
from .forest_class import Forest
from .tree_mesh_class import Tree_mesh
from .neuropil_class import Neuropil
from .forest_mesh_class import Forest_mesh
from .neuropils_class import Neuropils

__all__ = ["Tree", "Forest", "Tree_mesh", "Neuropil", "Forest_mesh", "Neuropils"]