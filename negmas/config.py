"""Defines basic config for NEGMAS"""
import json
import pathlib

__all__ = [
    "NEGMAS_CONFIG",
    "CONFIG_KEY_JNEGMAS_JAR",
    "CONFIG_KEY_GENIUS_BRIDGE_JAR",
]

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
