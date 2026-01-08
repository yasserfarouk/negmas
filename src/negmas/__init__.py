# -*- coding: utf-8 -*-
"""A framework for conducting multi-strand multilateral asynchronous negotiations on multiple issues."""

__author__ = """Yasser Mohammad"""
__email__ = "yasserfarouk@gmail.com"
__version__ = "0.11.4"


from .config import *
from .types import *
from .common import *
from .inout import *
from .mechanisms import *
from .negotiators import *
from .outcomes import *
from .gb import *
from .sao import *
from .situated import *
from .st import *
from .preferences import *

# Genius module exports - lazily loaded via __getattr__
_GENIUS_EXPORTS = frozenset(
    {
        "DEFAULT_JAVA_PORT",
        "DEFAULT_PYTHON_PORT",
        "DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT",
        "ANY_JAVA_PORT",
        "RANDOM_JAVA_PORT",
        "get_free_tcp_port",
        "GeniusBridge",
        "init_genius_bridge",
        "genius_bridge_is_running",
        "genius_bridge_is_installed",
        "GeniusNegotiator",
    }
)


def __getattr__(name: str):
    """Lazy load genius module exports on first access."""
    if name in _GENIUS_EXPORTS:
        from . import genius

        return getattr(genius, name)
    if name == "genius":
        from . import genius

        return genius
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    config.__all__
    + types.__all__
    + common.__all__
    + outcomes.__all__
    + preferences.__all__
    + negotiators.__all__
    + mechanisms.__all__
    + gb.__all__
    + sao.__all__
    + st.__all__
    + inout.__all__
    + situated.__all__
    + list(_GENIUS_EXPORTS)
    + ["genius"]
    # + modeling.__all__
    # + helpers.prob.__all__
    # + [
    #     "exceptions",
    #     "warnings",
    #     "generics",
    #     "helpers",
    #     "events",
    #     "tournaments",
    #     "elicitation",
    #     "helpers",
    # ]
)
