from __future__ import annotations

import sys

from ..common import DEFAULT_JAVA_PORT
from ..helpers.misc import get_free_tcp_port

__all__ = [
    "DEFAULT_JAVA_PORT",
    "DEFAULT_PYTHON_PORT",
    "DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT",
    "ANY_JAVA_PORT",
    "RANDOM_JAVA_PORT",
    "get_free_tcp_port",
]

DEFAULT_PYTHON_PORT = 25338
RANDOM_JAVA_PORT = 0
ANY_JAVA_PORT = -1
DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT = sys.maxsize
