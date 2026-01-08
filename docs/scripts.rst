NegMAS CLI
==========

When installing NegMAS through the pip command, you get two command line tools that can be used to
aid your development and testing:

1. ``negmas`` - A unified interface to all negmas commands (genius setup, tournaments, docs, etc.)
2. ``negotiate`` - A simple way to run negotiations, plot results, and save statistics

negmas Command
--------------

This section describes the ``negmas`` CLI which provides a unified interface to all negmas commands.

The set of supported commands are:

===============       ===================================================================
 Command                                  Meaning
===============       ===================================================================
docs                  Opens negmas docs in the browser (requires docs-setup first)
docs-setup            Downloads and installs docs to ~/negmas/docs
genius                Start the bridge to genius (to use GeniusNegotiator)
genius-setup          Downloads the genius bridge and updates your settings
tournament            Runs a tournament
version               Prints NegMAS version
===============       ===================================================================

You can also run ``negmas --gui`` to enable GUI mode for supported commands.

Documentation Commands
~~~~~~~~~~~~~~~~~~~~~~

**docs-setup**

Downloads and installs the NegMAS documentation to ``~/negmas/docs``. Run this once before using
the ``docs`` command.

.. code-block:: console

    $ negmas docs-setup

**docs**

Opens the NegMAS documentation in your default web browser. Make sure to run ``docs-setup`` first.

.. code-block:: console

    $ negmas docs

Genius Bridge (negmas genius)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command ``genius`` can be used to start a JVM running the Genius_ platform allowing ``GeniusNegotiator`` objects
to interact with existing GENIUS agents (Thanks to Tim Baarslag, Lead Developer of GENIUS, for allowing us
to ship it within NegMAS).

.. _Genius: http://ii.tudelft.nl/genius/

Before using the genius command, you need to set it up:

.. code-block:: console

    $ negmas genius-setup

You can get help on this tool by running:

.. code-block:: console

    $ negmas genius --help


This tool supports the following *optional* arguments:

===================   ==============================================================
 Argument                                  Meaning
===================   ==============================================================
-p/--path TEXT         Path to genius-8.0.4.jar with embedded NegLoader [OPTIONAL]
-r/--port INTEGER      Port to run the NegLoader on. Pass 0 for the default
                       value [OPTIONAL]
--force/--no-force     Force trial even if an earlier instance exists [OPTIONAL]
--help                 Show help message and exit.
===================   ==============================================================


Tournament Command (negmas tournament)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tournament command (``tournament``) allows you to run a tournament between different agents in some world and
compare their relative performance. The tool is general enough to support several world types.


You can get help on this tool by running:

.. code-block:: console

    $ negmas tournament --help

The ``tournament`` command has a set of sub-commands for creating, running, and combining tournament results as follows:

================  ====================================================
Command           Action
================  ====================================================
combine           Combine multiple tournaments at given base path(s)
combine-results   Combine results from multiple tournaments
create            Creates a tournament
eval              Evaluates a tournament and returns the results
run               Runs/continues a tournament
winners           Finds winners of a tournament or a set of tournaments
================  ====================================================


Creating a tournament
^^^^^^^^^^^^^^^^^^^^^

These are the *optional* arguments of this tool:

========================================== ==============================================================
  Argument                                      Meaning
========================================== ==============================================================
  -n, --name TEXT                           The name of the tournament. The special
                                            value "random" will result in a random name
                                            [default: random]
  -s, --steps INTEGER                       Number of steps. If passed then --steps-min
                                            and --steps-max are ignored
  --steps-min INTEGER                       Minimum number of steps (only used if
                                            --steps was not passed  [default: 50]
  --steps-max INTEGER                       Maximum number of steps (only used if
                                            --steps was not passed  [default: 100]
  -t, --timeout INTEGER                     Timeout the whole tournament after the given
                                            number of seconds (0 for infinite)
                                            [default: 0]
  --configs INTEGER                         Number of unique configurations to generate.
                                            [default: 5]
  --runs INTEGER                            Number of runs for each configuration
                                            [default: 2]
  --max-runs INTEGER                        Maximum total number of runs. Zero or
                                            negative numbers mean no limit  [default: -1]
  --agents INTEGER                          Number of agents per competitor (not used
                                            for anac2019std in which this is preset to
                                            1).  [default: 3]
  --factories INTEGER                       Minimum numbers of factories to have per
                                            level.  [default: 5]
  --competitors TEXT                        A semicolon (;) separated list of agent
                                            types to use for the competition.
  --jcompetitors, --java-competitors TEXT   A semicolon (;) separated list of agent
                                            types to use for the competition.
  --non-competitors TEXT                    A semicolon (;) separated list of agent
                                            types to exist in the worlds as non-
                                            competitors (their scores will not be
                                            calculated).
  -l, --log DIRECTORY                       Default location to save logs (A folder will
                                            be created under it)  [default:
                                            ~/negmas/logs/tournaments]
  --world-config FILE                       A file to load extra configuration
                                            parameters for world simulations from.
  --verbosity INTEGER                       verbosity level (from 0 == silent to 1 ==
                                            world progress)  [default: 1]
  --reveal-names / --hidden-names           Reveal agent names (should be used only for
                                            debugging)  [default: True]
  --log-ufuns / --no-ufun-logs              Log ufuns into their own CSV file. Only
                                            effective if --debug is given  [default: False]
  --log-negs / --no-neg-logs                Log all negotiations. Only effective if
                                            --debug is given  [default: False]
  --compact / --debug                       If True, effort is exerted to reduce the
                                            memory footprint whichincludes reducing logs
                                            dramatically.  [default: True]
  --raise-exceptions / --ignore-exceptions  Whether to ignore agent exceptions [default: True]
  --path TEXT                               A path to be added to PYTHONPATH in which
                                            all competitors are stored. You can path a :
                                            separated list of paths on linux/mac and a ;
                                            separated list in windows
  --java-interop-class TEXT                 The full name of a class that is used to
                                            represent Java agents to the python
                                            envirnment. It is only used if jcompetitors
                                            was passed
  --config-generator TEXT                   The full path to a configuration generator
                                            function that is used to generate all
                                            configs for the tournament. MUST be
                                            specified
  --world-generator TEXT                    The full path to a world generator function
                                            that is used to generate all worlds (given
                                            the assigned configs for the tournament.
                                            MUST be specified
  --assigner TEXT                           The full path to an assigner function that
                                            assigns competitors to different
                                            configurations
  --scorer TEXT                             The full path to a scoring function
  --cw INTEGER                              Number of competitors to run at every world
                                            simulation. It must either be left at
                                            default or be a number > 1 and < the number
                                            of competitors passed using --competitors
  --config FILE                             Read configuration from FILE.
========================================== ==============================================================


Running a tournament
^^^^^^^^^^^^^^^^^^^^

After creating a tournament using the ``tournament create`` command, it can be run using the ``tournament run`` command.
The parameters for this command are:

========================================== ==============================================================
 Argument                                   Meaning
========================================== ==============================================================
  -n, --name TEXT                           The name of the tournament. When invoked
                                            after create, there is no need to pass it
  -l, --log DIRECTORY                       Default location to save logs  [default:
                                            ~/negmas/logs/tournaments]
  --verbosity INTEGER                       verbosity level (from 0 == silent to 1 ==
                                            world progress)  [default: 1]
  --parallel / --serial                     Run a parallel/serial tournament on a single
                                            machine  [default: True]
  --distributed /  --single-machine         Run a distributed tournament using dask
                                            [default: False]
  --ip TEXT                                 The IP address for a dask scheduler to run
                                            the distributed tournament. Effective only
                                            if --distributed  [default: 127.0.0.1]
  --port INTEGER                            The IP port number a dask scheduler to run
                                            the distributed tournament. Effective only
                                            if --distributed  [default: 8786]
  --compact / --debug                       If True, effort is exerted to reduce the
                                            memory footprint whichincludes reducing logs
                                            dramatically.  [default: True]
  --path TEXT                               A path to be added to PYTHONPATH in which
                                            all competitors are stored. You can path a :
                                            separated list of paths on linux/mac and a ;
                                            separated list in windows
  --metric TEXT                             The statistical metric used for choosing the
                                            winners. Possibilities are mean, median,
                                            std, var, sum  [default: mean]
  --config FILE                             Read configuration from FILE.
========================================== ==============================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the ``--log`` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing
earlier results, a new folder will be created for each run named by the current date and time. The
folder will contain the following files:


=========================   ========     =================================================================
 File/Folder Name             Format         Content
=========================   ========     =================================================================
configs                     FOLDER       Contains one json file for each world
                                         run tried during the tournament. You can
                                         re-run this world using `run_world` function in the `tournament`
                                         module.
params.json                 JSON         The parameters used to create this tournament
base_configs.json           JSON         The base configurations used in the tournament (without agent/factory
                                         assignments.
assigned_configs.json       JSON         The configurations used after assigning factories to managers
scores.csv                  CSV          Scores of every agent in every world
total_scores.csv            CSV          Scores of every agent **type** averaged over all runs
winners.csv                 CSV          Winner *types* and their average scores
ttest.csv                   CSV          Results of a factorial TTEST comparing the performance of all
                                         agent *types*
=========================   ========     =================================================================

Other than these files, a folder with the same number as the corresponding config file in the configs folder, keeps full
statistics/log of every world *but only if --debug is specified* (see the `World Runner` section for the contents of
this folder.

Evaluating a tournament
^^^^^^^^^^^^^^^^^^^^^^^

To evaluate a tournament and get results, use the ``tournament eval`` command:

.. code-block:: console

    $ negmas tournament eval --help

Combining tournament results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Can be used to combine the results of multiple tournaments runs using ``tournament combine``.
The parameters of this command are:

======================  =======================================================
 Argument                 Meaning
======================  =======================================================
  -d, --dest DIRECTORY  The location to save the results
  --metric TEXT         The statistical metric used for choosing the winners.
                        Possibilities are mean, median, std, var, sum
                        [default: median]
  --config FILE         Read configuration from FILE.
======================  =======================================================

To combine results from multiple tournaments, use ``tournament combine-results``:

.. code-block:: console

    $ negmas tournament combine-results --help


Finding the winners of a tournament
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To report the winners of a tournament, you can use ``tournament winners``. The parameters of this command are:

============================== =======================================================
 Argument                       Meaning
============================== =======================================================
  -n, --name TEXT               The name of the tournament. When invoked after
                                create, there is no need to pass it
  -l, --log DIRECTORY           Default location to save logs  [default:
                                ~/negmas/logs/tournaments]
  --recursive / --no-recursive  Whether to recursively look for tournament
                                results. --name should not be given if
                                --recursive  [default: True]
  --metric TEXT                 The statistical metric used for choosing the
                                winners. Possibilities are mean, median, std,
                                var, sum  [default: median]
  --config FILE                 Read configuration from FILE.
============================== =======================================================



negotiate Command
-----------------

The ``negotiate`` CLI provides a simple way for running negotiations, plotting them, and saving their statistics.

Basic Usage
~~~~~~~~~~~

Run a simple negotiation with default settings:

.. code-block:: console

    $ negotiate

Run with a specific scenario:

.. code-block:: console

    $ negotiate --scenario path/to/scenario.yml

You can find out about all the available options by running:

.. code-block:: console

    $ negotiate --help

Command Options
~~~~~~~~~~~~~~~

The ``negotiate`` command supports many options organized into several categories:

**Basic Options**

================================ =================================================================
 Option                           Description
================================ =================================================================
--scenario PATH                   The scenario to negotiate about [default: Generate A new Scenario]
-p, -m, --protocol, --mechanism   The protocol (Mechanism) to use [default: SAO]
-n, -a, --agent, --negotiator     Negotiator (agent) type. Use adapter/negotiator format for
                                  adapters (e.g. TAUAdapter/AspirationNegotiator)
                                  [default: AspirationNegotiator, NaiveTitForTatNegotiator]
-E, --extend-negotiators          Extend the negotiator list to cover all ufuns
-T, --truncate-ufuns              Use only the first n negotiator ufuns
--params TEXT                     Mechanism initialization parameters as comma-separated
                                  ``key=value`` pairs
--share-ufuns/--no-share-ufuns    Share partner ufuns using private-data [default: no-share-ufuns]
--share-reserved-values           Share partner reserved-values using private-data
================================ =================================================================

**Deadline Options**

=================== =================================================================
 Option              Description
=================== =================================================================
-s, --steps         Number of steps allowed in the negotiation [default: None]
-t, --time          Number of seconds allowed in the negotiation [default: None]
=================== =================================================================

**Scenario Overrides**

========================= =================================================================
 Option                    Description
========================= =================================================================
-r, --reserved            Reserved values to override the ones in the scenario
-f, --fraction            Rational fractions for generating reserved values
-d/--discount, -D         Load/skip discount factor [default: discount]
--normalize/-N            Normalize ufuns to range (0-1) [default: True]
========================= =================================================================

**Generated Scenario Options**

When no scenario is provided, these options control scenario generation:

================================ =================================================================
 Option                           Description
================================ =================================================================
-i, --issues                      Number of issues [default: None]
--values-min                      Minimum n. values per issue [default: 2]
--values-max                      Maximum n. values per issue [default: 50]
-z, --size                        Sizes of issues in order (overrides values-min/max)
--reserved-values-min             Min allowed reserved value [default: 0.0]
--reserved-values-max             Max allowed reserved value [default: 1.0]
-R/--rational, -I/--irrational-ok Guarantee some rational outcomes [default: rational]
-F, --rational-fraction           Reservation fractions [default: None]
--reservation-selector            Reservation value selector: min|max|first|last [default: min]
--issue-name                      Issue names [default: None]
--os-name                         Outcome space name [default: None]
--ufun-names                      Names of ufuns [default: None]
--numeric/--no-numeric            Numeric issues [default: no-numeric]
--linear/--non-linear             Linear ufuns [default: linear]
--pareto-generator                Pareto generator method(s)
================================ =================================================================

**Output Control**

================================ =================================================================
 Option                           Description
================================ =================================================================
-v, --verbose                     Make verbose
--verbosity INTEGER               Verbosity level (higher=more verbose) [default: 0]
--progress/--no-progress          Show progress bar [default: progress]
--history/--no-history            Print history [default: no-history]
--stats/--no-stats                Generate statistics [default: stats]
--rank-stats/--no-rank-stats      Generate rank statistics [default: no-rank-stats]
-c/--compact-stats, -C            Show/hide distances [default: compact-stats]
================================ =================================================================

**Plotting Options**

================================ =================================================================
 Option                           Description
================================ =================================================================
--plot/--no-plot                  Generate plot [default: plot]
-2/--only2d, -0/--with-offers     Only 2D plot vs with offers [default: with-offers]
--plot-backend TEXT               Backend used for plotting (see matplotlib backends)
--plot-interactive                Make the plot interactive [default: plot-interactive]
--plot-show/--no-plot-show        Show the plot [default: plot-show]
--simple-offers-view              Simple offers view [default: no-simple-offers-view]
--annotations/--no-annotations    Show annotations [default: no-annotations]
--agreement/--no-agreement        Show agreement [default: no-agreement]
--pareto-dist/--no-pareto-dist    Show Pareto distance [default: pareto-dist]
--nash-dist/--no-nash-dist        Show Nash distance [default: nash-dist]
--kalai-dist/--no-kalai-dist      Show Kalai distance [default: kalai-dist]
--max-welfare-dist                Show max welfare distance [default: max-welfare-dist]
--max-rel-welfare-dist            Show max relative welfare distance [default: no]
--end-reason/--no-end-reason      Show end reason [default: end-reason]
--show-reserved/--no-show-reserved Show reserved value lines [default: show-reserved]
--total-time/--no-total-time      Show time limit [default: total-time]
--relative-time/--no-relative-time Show relative time [default: relative-time]
--show-n-steps/--no-show-n-steps  Show n. steps [default: show-n-steps]
--plot-path PATH                  Path to save the plot to [default: None]
================================ =================================================================

**Saving to Disk**

================================ =================================================================
 Option                           Description
================================ =================================================================
--save-path PATH                  Path to save results to [default: Do not Save]
--save-history/--no-save-history  Save negotiation history [default: save-history]
--save-stats/--no-save-stats      Save statistics [default: save-stats]
--save-type TEXT                  Scenario format: yml|xml [default: yml]
--save-compact/--no-save-compact  Compact file [default: save-compact]
================================ =================================================================

**Advanced Options**

================================ =================================================================
 Option                           Description
================================ =================================================================
--fast/--no-fast                  Avoid slow operations [default: no-fast]
--path PATH                       Extra paths to look for negotiator and mechanism classes
--raise-exceptions                Raise exceptions on failure [default: no-raise-exceptions]
================================ =================================================================

Examples
~~~~~~~~

Run a negotiation with 3 issues and 100 steps:

.. code-block:: console

    $ negotiate -i 3 -s 100

Run with specific negotiators:

.. code-block:: console

    $ negotiate -n AspirationNegotiator -n TitForTatNegotiator

Run without plotting:

.. code-block:: console

    $ negotiate --no-plot

Save results to a specific path:

.. code-block:: console

    $ negotiate --save-path ./results

Run with a time limit of 60 seconds:

.. code-block:: console

    $ negotiate -t 60
