# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# sys.path.insert(0, os.path.abspath('../../solid_mechanics'))
# sys.path.insert(0, os.path.abspath('../../examples'))
# sys.path.insert(0, os.path.abspath('../../solver'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.append(os.path.abspath('../../solid_mechanics'))
# sys.path.append(os.path.abspath('../../examples'))
# sys.path.append(os.path.abspath('../../solver'))

# -- Project information -----------------------------------------------------

project = 'OpenSolve.Solid'
copyright = '2021, Benjamin Alheit'
author = 'Benjamin Alheit'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    'sphinx.ext.imgmath',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

imgmath_latex_preamble = """
\\renewcommand{\\d}{\\text{d}}
\\newcommand{\\intomO}[1]{\\int_{\\Omega_{0}} #1 \\, d \\Omega}
\\renewcommand{\\div}{\\text{div}}
\\newcommand{\\grad}{\\text{grad}}
\\newcommand{\\Grad}{\\text{Grad}}
"""
