Command Line Scripts
====================

When installing NegMAS through the pip command, you get one command line tool that can be used to
aid your development and testing. This tool provides a unified interface to all negmas commands.

The set of supported commands are:

===============       ===================================================================
 Command                                  Meaning
===============       ===================================================================
genius                Run a Genius Bridge. This bridge allows you to use GeniusNegotiator
                      agents. Please notice that this command by-default runs in the
                      foreground preventing further input to the terminal.
genius-setup          Downloads the genius bridge and updates your settings.
jnegmas               Start the bridge to JNegMAS (to use Java agents in worlds)
jnegmas-setup         Downloads jnegmas and updates your settings
scml                  Runs an SCML world
tournament            Runs a tournament
version               Prints NegMAS version
===============       ===================================================================

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


SCML World Runner (negmas scml)
-------------------------------

The SCML World Runner command (`scml`) runs an SCML world with default factory managers and reports
the results of this run.

You can get help on this tool by running:

.. code-block:: console

    $ negmas scml --help


These are the *optional* arguments of this tool:

=========================  =================================================
    Argument                     Meaning
=========================  =================================================
--steps INTEGER            Number of steps.  [default: 120]
--levels INTEGER           Number of intermediate production levels
                           (processes). -1 means a single product and no
                           factories.  [default: 3]
--neg-speedup INTEGER      Negotiation Speedup.  [default: 21]
--negotiator TEXT          Negotiator type to use for builtin agents.
                           [default: negmas.sao.AspirationNegotiator]
--min-consumption INTEGER  The minimum number of units consumed by each
                           consumer at every time-step.  [default: 3]
--max-consumption INTEGER  The maximum number of units consumed by each
                           consumer at every time-step.  [default: 5]
--agents INTEGER           Number of agents (miners/negmas.consumers) per
                           production level  [default: 5]
--horizon INTEGER          Consumption horizon.  [default: 20]
--transport INTEGER        Transportation Delay.  [default: 0]
--time INTEGER             Total time limit.  [default: 5400]
--neg-time INTEGER         Time limit per single negotiation  [default: 240]
--neg-steps INTEGER        Number of rounds per single negotiation
                           [default: 20]
--sign INTEGER             The default delay between contract conclusion and
                           signing  [default: 1]
--guaranteed TEXT          Whether to only sign contracts that are
                           guaranteed not to cause breaches  [default:
                           False]
--lines INTEGER            The number of lines per factory  [default: 10]
--retrials INTEGER         The number of times an agent re-tries on failed
                           negotiations  [default: 5]
--use-consumer TEXT        Use internal consumer object in factory managers
                           [default: True]
--max-insurance INTEGER    Use insurance against partner in factory managers
                           up to this premium  [default: 100]
--riskiness FLOAT          How risky is the default factory manager
                           [default: 0.0]
--log TEXT                 Default location to save logs (A folder will be
                           created under it)  [default: ~/negmas/logs]
--compact / --debug        If --compact, effort is exerted to reduce the memory
                           footprint whichincludes reducing logs
                           dramatically.  [default: --compact]
--log-ufuns                If given, ufuns are logged [default: False]
                           Only used if --debug is given
--log-negs                 If given, all negotiations and their offers are logged
                           [default: False]
--config FILENAME          configuration file name. If given all of the
                           parameters given above can be entered in this file
                           instead of being inputed on the command line.
--help                     Show help message and exit.
=========================  =================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs under the HOME directory). To avoid over-writing earlier
results, a new folder will be created for each run named by the current date and time (within an `scml` folder). The
folder will contain the following files:

=======================    =========    ====================================
File Name                  Format       Content
=======================    =========    ====================================
all_contracts.csv             CSV        A record of all contracts [filled only if --debug is specified]
contracts_full_info.csv       CSV        A record of all contracts with added information about the CFPs  [filled only if --debug is specified]
cancelled_contracts.csv       CSV        Contracts that were cancelled because one partner refused to sign it  [filled only if --debug is specified]
signed_contracts.csv          CSV        Contracts that were actually signed
negotiations.csv              CSV        A record of all negotiations  [filled only if --debug is specified]
breaches.csv                  CSV        A record of all breaches
stats.csv                     CSV        Helpful statistics about the state of the world at every timestep
                                         (e.g. N. negotiations, N. Contracts Executed, etc) in CSV format
stats.json                    JSON       Helpful statistics about the state of the world at every timestep
                                         (e.g. N. negotiations, N. Contracts Executed, etc) in JSON format
params.json                   JSON       The arguments used to run the world
logs.txt                      TXT        A log file giving details of most important events during the simulation
                                         [filled only if --debug is specified]
negotiation_info.csv          CSV        Negotiation information for all negotiation session logged (only if --log-negs
                                         is given).
negotiations                Folder       A folder containing a file for each negotiation giving all offers exchanged (only if --log-negs
                                         is given).
=======================    =========    ====================================



Tournament Command (negmas tournament)
--------------------------------------

The Tournament command (`tournament`) allows you to run a tournament between different agents in some world and
compare their relative performance. The tool is general enough to support several world types but currently only the
ANAC 2019 SCML (`anac2019`) configuration is supported.


You can get help on this tool by running:

.. code-block:: console

    $ negmas tournament --help


These are the *optional* arguments of this tool:

=================================== ==============================================================
Argument                             Meaning
=================================== ==============================================================
-n, --name TEXT                      The name of the tournament. The special
                                     value "random" will result in a random name [default: random]
-s, --steps INTEGER                  Number of steps.  [default: 60]
-f, --config TEXT                    The config to use. Default is ANAC 2019 [default: anac2019]
-t, --timeout INTEGER                Timeout after the given number of seconds (0 for infinite)
                                     [default: 0]
--runs INTEGER                       Number of runs for each configuration [default: 5]
--max-runs INTEGER                   Maximum total number of runs. Zero or negative numbers mean no
                                     limit  [default:-1]
--configs INTEGER                    Number of unique configurations to generate.
                                     [default: 5]
--runs INTEGER                       Number of runs for each configuration
                                     [default: 2]
--max-runs INTEGER                   Maximum total number of runs. Zero or
                                     negative numbers mean no limit  [default:
                                     -1]
--factories INTEGER                  Minimum numbers of factories to have per
                                     level.  [default: 5]
--competitors TEXT                   A semicolon (;) separated list of agent types to use for the
                                     competition.
                                     [default:negmas.apps.scml.DoNothingFactoryManager;
                                     negmas.apps.scml.GreedyFactoryManager]
--jcompetitors /--java-competitors    A semicolon (;) separated list of agent
                                      types to use for the competition.
--parallel / --serial                Run a parallel/serial tournament on a single machine
                                     [default: True]
--distributed / --single-machine     Run a distributed tournament using dask [default: False]
-l, --log TEXT                       Default location to save logs (A folder will be created under
                                     it)  [default:~/negmas/logs/tournaments]
--verbosity INTEGER                  verbosity level (from 0 == silent to 1 ==
                                     world progress)  [default: 1]
--configs-only / --run               configs_only  [default: False]
--reveal-names / --hidden-names      Reveal agent names (should be used only for debugging)
                                     [default: False]
--ip TEXT                            The IP address for a dask scheduler to run the distributed
                                     tournament.
                                     Effective only if --distributed  [default: 127.0.0.1]
--port INTEGER                       The IP port number a dask scheduler to run
                                     the distributed tournament. Effective only
                                     if --distributed  [default: 8786]
--compact / --debug                  If --compact, effort is exerted to reduce the memory
                                     footprint whichincludes reducing logs
                                     dramatically.  [default: --compact]
--log-ufuns                          If given, ufuns are logged [default: False]
                                     Only used if --debug is given
--log-negs                           If given, all negotiations and their offers are logged.
                                     Only used if --debug is given
                                     [default: False]
--config FILENAME                    configuration file name. If given all of the
                                     parameters given above can be entered in this file
                                     instead of being inputed on the command line.
--help                               Show help message and exit.
=================================== ==============================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing earlier
results, a new folder will be created for each run named by the current date and time. The
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
statistics/log of every world *but only if --debug is specified* (see the `SCML World Runner` section for the contents of
this folder.



