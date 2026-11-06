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

Parallel runs
~~~~~~~~~~~~~

Each task dispatched to a worker is seeded from its own index, so tasks stay
independent of one another, a run reproduces whatever the scheduling order or
the number of workers, and running with ``njobs=0`` reproduces running with
``njobs=4``.

Threads are the exception: threaded tasks share one process and therefore one
set of global generators, so their interleaving -- and their results -- are not
reproducible. Use the serial or process-based runners when reproducibility
matters.

Code that runs its own process pool, rather than dispatching through
:mod:`negmas.helpers.parallel`, can get the same behaviour with
``task_seed(index)`` and ``apply_seed(seed)``: ``task_seed`` returns ``None``
when nothing is seeded, and ``apply_seed`` seeds a task without making its
seed the base that the remaining tasks derive from.

Seeding one component or one block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The global seed makes a whole run repeat. To make *one* component repeat,
whatever the rest of the program does, give it its own seed. The Pareto
samplers, ``SamplingInverseUtilityFunction`` and the Nice Tit for Tat negotiator
(``sampler_seed``) accept one; the default, ``None``, keeps them on the global
generators exactly as before:

.. code-block:: python

    from negmas.outcomes import make_issue, make_os
    from negmas.preferences import LinearAdditiveUtilityFunction
    from negmas.preferences.pareto_sampler import BruteForceParetoSampler

    os = make_os([make_issue(10, "price"), make_issue(10, "quantity")])
    a = LinearAdditiveUtilityFunction.random(outcome_space=os)
    b = LinearAdditiveUtilityFunction.random(outcome_space=os)
    sampler = BruteForceParetoSampler(seed=42)
    sampler.init(a, b)
    front = sampler.pareto_outcomes()

Any other block of code gets the same with ``seeded``:

.. code-block:: python

    from negmas import seeded

    with seeded(7):
        first = [os.random_outcome() for _ in range(5)]
    with seeded(7):
        again = [os.random_outcome() for _ in range(5)]
    assert first == again

Inside the block, :mod:`random` and the global :mod:`numpy.random` generator
draw from a private stream started from the seed; afterwards both are put back
exactly as they were, so the rest of the program is unaffected. A component's
stream continues from call to call rather than restarting. ``RandomStream`` is
the class behind both, and ``uses_random_stream`` makes the methods of your own
components seedable the same way. Swapping the global generators is not
thread-safe.

With a seed in effect, the ids NegMAS gives to negotiators, mechanisms,
contracts and other named objects are reproducible as well. They come from a
generator of their own, so creating objects never shifts the numbers anything
else draws.

What is not covered: randomness inside the Genius bridge (Java), numpy
``Generator`` objects created by user code, and anything that depends on wall
clock time (time limits, ``pend_per_second``).

Seeding the whole shell
~~~~~~~~~~~~~~~~~~~~~~~

``NEGMAS_RAND_SEED`` does not cover hash randomization, because
``PYTHONHASHSEED`` is read before the interpreter starts and so cannot be set
from inside a running process. The ``negmas seed`` command prints every
environment setting that makes a run reproducible -- NegMAS's own, plus the
seeding knobs of the common libraries used alongside it -- for the shell to
apply:

.. code-block:: bash

    eval "$(negmas seed 44)"

Every run started from that shell then uses seed 44. Pass ``--no-export`` to
get bare ``NAME=VALUE`` lines instead (useful for a ``.env`` file or a
container's environment).

API Reference
-------------

.. automodule:: negmas.config
   :members:
   :undoc-members:
   :show-inheritance:
