# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ProbFlow"
copyright = "2025, ProbFlow Contributors"
author = "ProbFlow Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for LaTeX / PDF output ------------------------------------------
latex_documents = [
    (
        "index",
        "ProbFlow.tex",
        "ProbFlow Documentation",
        "ProbFlow Contributors",
        "manual",
    ),
]

# -- nbsphinx configuration -------------------------------------------------
nbsphinx_execute = "never"

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
