# negmas documentation build configuration file, created byconf.py
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#


import os

# import negmas

# NON_RTD_THEME = "python_docs_theme"
# NON_RTD_THEME = "groundwork"
# on_rtd is whether we are on readthedocs.org
import sphinx_rtd_theme

on_rtd = os.environ.get("READTHEDOCS", None)

THEME_NAME = "sphinx_rtd_theme"
THEME_PATH = None
if not on_rtd:
    THEME_PATH = [sphinx_rtd_theme.get_html_theme_path()]

# typing.get_type_hints = lambda obj, *unused: obj


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.smart_resolver",
    # "nb2plots",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_autodoc_annotation",
    # "sphinxcontrib.fulltoc",
    "nbsphinx",
    "sphinx_tabs",
]

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    # 'matplotlib': ('http://matplotlib.sourceforge.net/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = '.rst'
source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

# The strategy toctree document.
master_doc = "index"

# General information about the project.
project = "NegMAS"
copyright = "2018, Yasser Mohammad"
author = "Yasser Mohammad"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = "0.11.0"
# The full version, including alpha/cost/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


if not on_rtd:  # only set the theme if we're building docs locally
    html_context = {
        "css_files": [
            "_static/theme_overrides.css"
        ]  # override wide tables in RTD theme
    }
# theme options for sphinx_rtd_theme
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "titles_only": False,
}
html_theme = THEME_NAME
if THEME_PATH:
    html_theme_path = THEME_PATH

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# theme options for alabaster
# html_theme_options = {
#     'show_relbars': True,
#     'show_related': True,
#
# }

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "negmasdoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "negmas.tex", "NegMAS Documentation", "Yasser Mohammad", "manual")
]

# -- Options for graphviz used in inheritence diagrams -----------------

# graphviz_output_format = "svg"
# if on_rtd:
#     graphviz_output_format = "png"
inheritance_graph_attrs = dict(
    randkir="TB", fontsize=11, size='""'
)  # , size='"16.0, 20.0"')
inheritance_node_attrs = dict(fontsize=11)  # , size='"16.0, 20.0"')

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "negmas", "NegMAS Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "negmas",
        "NegMAS Documentation",
        author,
        "NegMAS",
        "Situated Simultaneous Negotiations Library.",
        "Miscellaneous",
    )
]

default_role = "any"

imgmath_image_format = "png"


# If false, no module index is generated.
html_domain_indices = True

automodsumm_inherited_members = True

autodoc_typehints = "signature"
set_type_checking_flag = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
# autodoc_typehints = "description"

# Don't show class signature with the class' name.
# autodoc_class_signature = "separated"

# mathjax_path =
