negmas CLI
==========

The ``negmas`` CLI provides a unified interface to all NegMAS commands including Genius setup,
tournaments, documentation access, and more.

Installation
------------

The ``negmas`` command is automatically available after installing NegMAS:

.. code-block:: console

    $ pip install negmas

Basic Usage
-----------

Run the main help to see all available subcommands:

.. code-block:: console

    $ negmas --help

Common subcommands include:

- ``negmas genius-setup`` - Download and configure the Genius bridge
- ``negmas genius`` - Start the Genius bridge for Java agent integration
- ``negmas tournament`` - Run negotiation tournaments

Command Reference
-----------------

.. click:: negmas.scripts.app:cli
   :prog: negmas
   :nested: full
