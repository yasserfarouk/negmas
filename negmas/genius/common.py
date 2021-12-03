from __future__ import annotations

import socket
import sys

__all__ = [
    "DEFAULT_JAVA_PORT",
    "DEFAULT_PYTHON_PORT",
    "DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT",
    "ANY_JAVA_PORT",
    "RANDOM_JAVA_PORT",
    "get_free_tcp_port",
]
DEFAULT_JAVA_PORT = 25337
DEFAULT_PYTHON_PORT = 25338
RANDOM_JAVA_PORT = 0
ANY_JAVA_PORT = -1
DEFAULT_GENIUS_NEGOTIATOR_TIMEOUT = sys.maxsize


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port
