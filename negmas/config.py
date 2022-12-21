from __future__ import annotations

"""Defines basic config for NEGMAS"""
import json
import pathlib

__all__ = [
    "NEGMAS_CONFIG",
    "CONFIG_KEY_JNEGMAS_JAR",
    "CONFIG_KEY_GENIUS_BRIDGE_JAR",
]

NEGMAS_DEFAULT_PATH = "~/negmas/config.json"
"""Default path for NegMAS configurations"""

CONFIG_KEY_JNEGMAS_JAR = "jnegmas_jar"
"""Key name for the JNegMAS jar in `NEGMAS_CONFIG`"""

CONFIG_KEY_GENIUS_BRIDGE_JAR = "genius_bridge_jar"
"""Key name for the Genius bridge jar in `NEGMAS_CONFIG`"""

NEGMAS_CONFIG = {
    CONFIG_KEY_JNEGMAS_JAR: f"{NEGMAS_DEFAULT_PATH}/{CONFIG_KEY_JNEGMAS_JAR}",
    CONFIG_KEY_GENIUS_BRIDGE_JAR: f"{NEGMAS_DEFAULT_PATH}/{CONFIG_KEY_GENIUS_BRIDGE_JAR}",
}
# loading config file if any
__conf_path = pathlib.Path(NEGMAS_DEFAULT_PATH).expanduser().absolute()

if __conf_path.exists():
    with open(__conf_path) as f:
        try:
            NEGMAS_CONFIG = json.load(f)
        except:
            pass
