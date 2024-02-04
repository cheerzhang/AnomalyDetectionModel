# docs/source/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path based on your project structure

project = 'Your Project Name'
author = 'Your Name'

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ----------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for autodoc extension ----------------------------------------

autodoc_member_order = 'bysource'
