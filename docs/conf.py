# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "NeuRosetta"
author = "Nik Drummond"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "data/README.md"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "NeuRosetta"
html_theme_options = {
    "source_repository": "https://github.com/NikDrummond/NeuRosetta",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        # Main colors
        "color-brand-primary": "#7c3aed",
        "color-brand-content": "#9333ea",

        # Page background
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f5f3ff",

        # Text
        "color-foreground-primary": "#1f2937",
        "color-foreground-secondary": "#4b5563",

        # Admonitions
        "color-admonition-background": "#fef3c7",
    },

    "dark_css_variables": {
        "color-brand-primary": "#c084fc",
        "color-brand-content": "#d8b4fe",

        "color-background-primary": "#111827",
        "color-background-secondary": "#1f2937",

        "color-foreground-primary": "#f9fafb",
        "color-foreground-secondary": "#d1d5db",
    },
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}

# Document class-level function bindings (Tree/Forest method aliases).
autodoc_preserve_defaults = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

nitpicky = False


def _callable_class_attrs(app, what, name, obj, skip, options):
    if what == "class":
        member = getattr(obj, name, None)
        if callable(member) and name in getattr(obj, "__dict__", {}):
            return False
    return None


def setup(app):
    app.connect("autodoc-skip-member", _callable_class_attrs)
