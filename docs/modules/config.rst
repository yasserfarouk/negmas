negmas.config
=============

Configuration management for NegMAS. This module handles loading and accessing
configuration values from various sources.

Overview
--------

NegMAS configuration values are read from multiple sources in the following
priority order (highest to lowest):

1. Environment variables with the prefix ``NEGMAS_`` (e.g., ``NEGMAS_GENIUS_BRIDGE_JAR``)
2. Local configuration file ``negmasconf.json`` in the current working directory
3. User configuration file at the path specified by ``NEGMAS_DEFAULT_PATH`` environment variable
4. Default user configuration at ``~/negmas/config.json``
5. Default values hardcoded in the library

Configuration Keys
------------------

The following configuration keys are available:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``genius_bridge_jar``
     - Path to the Genius bridge JAR file for Java integration
   * - ``jnegmas_jar``
     - Path to the JNegMAS JAR file
   * - ``warn_slow_ops``
     - Threshold for warning about slow operations (number of operations)

Usage
-----

To access configuration values programmatically:

.. code-block:: python

    from negmas.config import negmas_config

    # Get the path to the Genius bridge JAR
    jar_path = negmas_config("genius_bridge_jar", default="/path/to/default.jar")

    # Get warning threshold with default
    threshold = negmas_config("warn_slow_ops", default=100_000_000)

Configuration File Format
-------------------------

Configuration files should be JSON format:

.. code-block:: json

    {
        "genius_bridge_jar": "/path/to/geniusbridge.jar",
        "jnegmas_jar": "/path/to/jnegmas.jar",
        "warn_slow_ops": 100000000
    }

API Reference
-------------

.. automodule:: negmas.config
   :members:
   :undoc-members:
   :show-inheritance:
