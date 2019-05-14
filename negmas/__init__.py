# -*- coding: utf-8 -*-
"""A framework for conducting multi-strand multilateral asynchronous negotiations on multiple issues."""
__author__ = """Yasser Mohammad"""
__email__ = "yasserfarouk@gmail.com"
__version__ = "0.2.19"

import json
import pathlib


NEGMAS_CONFIG = {}
"""Global configuration parameters for NegMAS"""
CONFIG_KEY_JNEGMAS_JAR = "jnegmas_jar"
"""Key name for the JNegMAS jar in `NEGMAS_CONFIG`"""
CONFIG_KEY_GENIUS_BRIDGE_JAR = "genius_bridge_jar"
"""Key name for the Genius bridge jar in `NEGMAS_CONFIG`"""


# loading config file if any
__conf_path = pathlib.Path("~/negmas/config.json").expanduser().absolute()

if __conf_path.exists():
    with open(__conf_path, "r") as f:
        NEGMAS_CONFIG = json.load(f)

from .common import *
from .outcomes import *
from .utilities import *
from .negotiators import *
from .mechanisms import *
from .sao import *
from .inout import *
from .genius import *
from .situated import *

__all__ = (
    common.__all__
    + outcomes.__all__
    + utilities.__all__
    + negotiators.__all__
    + mechanisms.__all__
    + sao.__all__
    + inout.__all__
    + genius.__all__
    + situated.__all__
    + ["generics", "helpers", "events", "apps", "tournaments"]
)
