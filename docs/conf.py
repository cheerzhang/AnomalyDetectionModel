import sys, os


# Configuration file for the Sphinx documentation builder.
sys.path.insert(0, os.path.abspath('..'))


# -- Project information

project = 'quick-anomaly-detector'
copyright = '2024, LeZhang'
author = 'LeZhang'

release = '0.2'
version = '0.2.6'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
]
autosummary_generate = True


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for autodoc extension ----------------------------------------
autodoc_member_order = 'bysource'

master_doc = 'index'

