# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# conda activate chewacla
# ./docs/make.bat html

# -- Path setup --------------------------------------------------------------

import pathlib
import sys
import tomllib
from importlib.metadata import version

root_path = pathlib.Path(__file__).parent.parent.parent
with open(root_path / "pyproject.toml", "rb") as fp:
    toml = tomllib.load(fp)
metadata = toml["project"]

sys.path.insert(0, str(root_path))

# imports here for sphinx to build the documents without many WARNINGS.
import chewacla

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

author = metadata["authors"][0]["name"]
copyright = toml["tool"]["copyright"]["copyright"]
description = metadata["description"]
github_url = metadata["urls"]["source"]
project = metadata["name"]
release = chewacla.__version__
rst_prolog = f".. |author| replace:: {author}"
today_fmt = "%Y-%m-%d %H:%M"

# -- Special handling for version numbers ------------------------------------
# https://github.com/pypa/setuptools_scm#usage-from-sphinx

release = version(project)
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",  # Add sphinx-autoapi extension
]

templates_path = ["_templates"]
exclude_patterns = []

# AutoAPI configuration
autoapi_type = "python"  # Specify the type of API to document
autoapi_dirs = ["../../src"]  # Path to the directory containing the project
autoapi_ignore = ["dev_*", "*tests*"]
autoapi_options = [
    "members",  # Include members (functions, classes, etc.)
    "undoc-members",  # Include undocumented members
    "private-members",  # Include private members
    "special-members",  # Include special members (e.g., __init__)
    "inherited-members",  # Include inherited members
    "show-inheritance",  # Show inheritance diagrams
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_static_path = ["_static"]

autodoc_mock_imports = """
    hklpy2
    numpy
""".split()
