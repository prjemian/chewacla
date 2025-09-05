# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "chewacla"
copyright = "2025, Pete R Jemian"
author = "Pete R Jemian"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",  # Add sphinx-autoapi extension
]

templates_path = ["_templates"]
exclude_patterns = []

# AutoAPI configuration
autoapi_type = "python"  # Specify the type of API to document
autoapi_dirs = [f"../../src/{project}"]  # Path to the directory containing your code
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
# html_title = f"{project} {version}"
html_static_path = ['_static']
