import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'ACN (Agent Communication Networks)'
copyright = '2026, ACN Team'
author = 'ACN Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

source_suffix = {
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'technical_gap_analysis.md', 'Thumbs.db', '.DS_Store']
suppress_warnings = ['toc.not_included']

html_theme = 'sphinx_rtd_theme'
html_static_path = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
}
