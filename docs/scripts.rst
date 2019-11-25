Command Line Scripts
====================

When installing NegMAS through the pip command, you get one command line tool that can be used to
aid your development and testing. This tool provides a unified interface to all negmas commands.

The set of supported commands are:

===============       ===================================================================
 Command                                  Meaning
===============       ===================================================================
genius-setup          Downloads the genius bridge and updates your settings.
jnegmas-setup         Downloads jnegmas and updates your settings
genius                Run a Genius Bridge. This bridge allows you to use GeniusNegotiator
                      agents. Please notice that this command by-default runs in the
                      foreground preventing further input to the terminal.\
jnegmas               Start the bridge to JNegMAS (to use Java agents in worlds)
tournament            Runs a tournament
version               Prints NegMAS version
===============       ===================================================================

The commands `genius-steup` and `jnegmas-setup`  have no parameters and will download genius and jnegmas (respectively)
for later use by `genius` and `jnegmas` commands.

Genius Bridge (negmas genius)
-----------------------------

The command ``genius`` can be used to start a JVM running the Genius_ platform allowing `GeniusNegotiator` objects
to interact with existing GENIUS agents (Thanks for Tim Baarslag Lead Developer of GENIUS for allowing us
to ship it within NegMAS).

.. _Genius: http://ii.tudelft.nl/genius/

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

JNegMAS bridge (negmas jnegmas)
-------------------------------

Runs a bridge to jnegmas (notice that `negmas jnegmas-setup` must have been run at least once before that). It has the
following parameters

===================== =================================================================
 Argument               Meaning
===================== =================================================================
  -p, --path TEXT     Path to jnegmas*.jar with. Use "auto" to read the path
                      from ~/negmas/config.json.  Config key is jnegmas_jar
                      You can download the latest version of this
                      jar from: http://www.yasserm.com/scml/jnegmas-all.jar
                      [default: auto]
  -r, --port INTEGER  Port to run the jnegmas on. Pass 0 for the default value
                      [default: 0]
  --config FILE       Read configuration from FILE.
===================== =================================================================

Tournament Command (negmas tournament)
--------------------------------------

The Tournament command (`tournament`) allows you to run a tournament between different agents in some world and
compare their relative performance. The tool is general enough to support several world types.


You can get help on this tool by running:

.. code-block:: console

    $ negmas tournament --help

The `tournament` command has a set of sub-commands for creating, running, and combining tournament results as follows:

========  ================================================
Command   Action
========  ================================================
 combine  Finds winners of an arbitrary set of tournaments
 create   Creates a tournament
 run      Runs/continues a tournament
 winners  Finds winners of a tournament or a set of
          tournaments sharing
========  ================================================


Creating a tournament
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

After creating a tournament using the `tournament create` command, it can be run using the `tournament run` command.
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
specified by the `--log` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing
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

Combining tournament results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can be used to combine the results of multiple tournaments runs using tournament `combine`.
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


Finding the winners of a tournament
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To report the winners of a tournament, you can use tournament `winners` . The parameters of this command are:

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


