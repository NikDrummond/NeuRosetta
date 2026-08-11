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
    "myst_nb",
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
html_css_files = ["custom.css"]
html_js_files = ["dyslexia-font.js", "bionic-reading.js"]
html_title = "NeuRosetta"
html_theme_options = {
    "source_repository": "https://github.com/NikDrummond/NeuRosetta",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        # Microscopy slate — teal accent on warm stone canvas
        "color-brand-primary": "#0d9488",
        "color-brand-content": "#0f766e",

        "color-background-primary": "#fafaf9",
        "color-background-secondary": "#f0fdfa",
        "color-background-border": "#e7e5e4",
        "color-background-hover": "#ccfbf1",

        "color-foreground-primary": "#1c1917",
        "color-foreground-secondary": "#57534e",
        "color-foreground-muted": "#78716c",
        "color-foreground-border": "#d6d3d1",

        "color-admonition-background": "#fffbeb",
        "color-code-background": "#ecfdf5",
    },

    "dark_css_variables": {
        "color-brand-primary": "#2dd4bf",
        "color-brand-content": "#5eead4",

        "color-background-primary": "#0c1222",
        "color-background-secondary": "#111827",
        "color-background-border": "#1e293b",
        "color-background-hover": "#134e4a",

        "color-foreground-primary": "#f8fafc",
        "color-foreground-secondary": "#cbd5e1",
        "color-foreground-muted": "#94a3b8",
        "color-foreground-border": "#334155",

        "color-admonition-background": "#422006",
        "color-code-background": "#042f2e",
    },
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# Notebooks (myst-nb): render committed outputs; re-execute locally when updating.
nb_execution_mode = "off"
nb_merge_streams = True
nb_show_stderr = "warning"

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
