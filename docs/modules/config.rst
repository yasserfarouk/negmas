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
   * - ``rand_seed``
     - Seed for every random number generator NegMAS uses. Unset (the default)
       means every run draws fresh entropy; an integer makes the run
       reproducible. See :ref:`reproducibility`.

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


.. _reproducibility:

Reproducibility
---------------

Setting ``NEGMAS_RAND_SEED`` before a run seeds every random number generator
NegMAS uses -- the global generators of :mod:`random` and :mod:`numpy.random`
-- so the whole run becomes reproducible:

.. code-block:: bash

    NEGMAS_RAND_SEED=42 python my_experiment.py

The same can be done from within Python, which is also how you re-seed between
runs in the same process:

.. code-block:: python

    from negmas.helpers.rand import seed_all, get_seed

    seed_all(42)
    assert get_seed() == 42

Leaving the variable unset (the default) keeps the historical behaviour: every
run draws fresh entropy. ``NEGMAS_RAND_SEED=random`` asks for that explicitly.

Libraries built on NegMAS can hook their own generators onto the same switch,
so that one setting covers them too:

.. code-block:: python

    from negmas.helpers.rand import register_seeder


    def _seed_my_library(seed: int) -> None:
        my_library.set_seed(seed)


    register_seeder(_seed_my_library)

A seeder registered after a seed is already in effect is called immediately
with it, so import order does not matter. Seeding is best-effort throughout: a
generator that cannot be seeded produces a warning, never an exception.

.. note::

    Hash randomization is not covered, because ``PYTHONHASHSEED`` can only be
    set before the interpreter starts. Export it yourself if iteration order
    over sets of strings affects your results.

API Reference
-------------

.. automodule:: negmas.config
   :members:
   :undoc-members:
   :show-inheritance:
