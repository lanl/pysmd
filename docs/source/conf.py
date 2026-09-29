"""Configure the Sphinx documentation build."""

import sys
from importlib.metadata import version as distribution_version
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------

project = "PySMD"
copyright = "2026, Triad National Security, LLC"
author = "Ilia M. Mazin, Yu Zhang, Anders M.N. Niklasson"
release = distribution_version("pysmd")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

modindex_common_prefix = ["pysmd."]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# -- Options for HTML output -------------------------------------------------

html_title = "PySMD documentation"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 6,
    "collapse_navigation": False,
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyscf": ("https://pyscf.org/", None),
    "python": ("https://docs.python.org/3/", None),
}

napoleon_include_init_with_doc = True
