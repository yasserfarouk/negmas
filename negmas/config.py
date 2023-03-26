from __future__ import annotations

"""Defines basic config for NEGMAS"""
import json
from os import environ
from pathlib import Path

__all__ = [
    "NEGMAS_CONFIG",
    "CONFIG_KEY_JNEGMAS_JAR",
    "CONFIG_KEY_GENIUS_BRIDGE_JAR",
    "negmas_config",
]

LOCAL_NEGMAS_CONFIG_FILENAME = "negmasconf.json"

NEGMAS_DEFAULT_PATH = Path(
    environ.get("NEGMAS_DEFAULT_PATH", Path.home() / "negmas" / "config.json")
)
"""Default path for NegMAS configurations"""

CONFIG_KEY_JNEGMAS_JAR = "jnegmas_jar"
"""Key name for the JNegMAS jar in `NEGMAS_CONFIG`"""

CONFIG_KEY_GENIUS_BRIDGE_JAR = "genius_bridge_jar"
"""Key name for the Genius bridge jar in `NEGMAS_CONFIG`"""

CONFIG_KEY_WARN_SLOW_OPS = "warn_slow_ops"

NEGMAS_CONFIG = {
    CONFIG_KEY_JNEGMAS_JAR: str(NEGMAS_DEFAULT_PATH.parent / "files" / "jnegmas.jar"),
    CONFIG_KEY_GENIUS_BRIDGE_JAR: str(
        NEGMAS_DEFAULT_PATH.parent / "files" / "geniusbridge.jar"
    ),
    CONFIG_KEY_WARN_SLOW_OPS: 100_000_000,
}

# loading config file if any
__conf_path = Path(NEGMAS_DEFAULT_PATH).expanduser().absolute()

if __conf_path.exists():
    try:
        with open(__conf_path) as f:
            NEGMAS_CONFIG.update(json.load(f))
    except:
        pass

local_path = Path.cwd() / LOCAL_NEGMAS_CONFIG_FILENAME
if local_path.exists():
    try:
        with open(local_path) as f:
            NEGMAS_CONFIG.update(json.load(f))
    except:
        pass


def _from_env(key: str, default):
    envkey = "NEGMAS_" + key.upper()
    v = environ.get(envkey, default)
    if key in (CONFIG_KEY_WARN_SLOW_OPS,):
        return int(v) if v else 0
    return v


def negmas_config(key: str, default):
    """
    Returns the config value associated with the given key.


    Remarks:
        - config values are read from the following sources (in descending order of priority):
            - Environment variable with the name NEGMAS_{key} (with the key converted to all uppercase)
            - Local file called negmasconf.json (with the key all lowercase)
            - json file stored at the location indicated by environment variable "NEGMAS_DEFAULT_PATH" (with the key all lowercase)
            - ~/negmas/config.json (with the key all lowercase)
            - A default value hardcoded in the negmas library. For paths, this usually lies under ~/negmas
    """
    return _from_env(key, NEGMAS_CONFIG.get(key.lower(), default))
