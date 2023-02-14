import sys, os
import sphinx_rtd_theme

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mybar'
copyright = '2022, SleepyAbsol'
author = 'SleepyAbsol'
release = '0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # 'attributetable',
]

# sys.path.insert(0, os.path.abspath('../mybar'))
sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('./mybar'))
# sys.path.insert(0, os.path.abspath('.'))

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    # 'imported-members': True,
    'exclude-members': '__init__',
##    'autodoc_class_content': 'class',
}

# autodoc_member_order = 'groupwise'
autodoc_member_order = 'bysource'
autodoc_class_content = 'class'
autodoc_class_signature = 'separated'
# autodoc_class_signature = 'mixed'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme = 'alabaster'
html_static_path = ['_static']
