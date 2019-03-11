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
scml                  Runs an SCML world
tournament            Runs a tournament
===============       ==============================================================

Genius Bridge (genius negmas)
-----------------------------

The command ``genius`` can be used to start a JVM running the Genius_ platform allowing `GeniusNegotiator` objects
to interact with existing GENIUS agents (Thanks for Tim Baarslag Lead Developer of GENIUS for allowing us
to ship it within NegMAS).

.. _Genius: http://ii.tudelft.nl/genius/

You can get help on this tool by running:

.. code-block:: console

    $ rungenius --help


This tool supports the following *optional* arguments:

===============       ==============================================================
 Argument                                  Meaning
===============       ==============================================================
-p, --path TEXT       Path to genius-8.0.4.jar with embedded NegLoader [OPTIONAL]
1. -r, --port INTEGER Port to run the NegLoader on. Pass 0 for the default
                    value [OPTIONAL]
1. --force/--no-force Force trial even if an earlier instance exists [OPTIONAL]
1. --help             Show help message and exit.
===============       ==============================================================


SCML World Runner (negmas scml)
-------------------------------

The SCML World Runner command (`scml`) runs an SCML world with default factory managers and reports
the results of this run.

You can get help on this tool by running:

.. code-block:: console

    $ scml --help


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
--help                     Show help message and exit.
=========================  =================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs under the HOME directory). To avoid over-writing earlier
results, a new folder will be created for each run named by the current date and time (within an `scml` folder). The
folder will contain the following files:

=======================    ========     ====================================
File Name                  Format       Content
=======================    ========     ====================================
all_contracts.csv             CSV        A record of all contracts
contracts_full_info.csv       CSV        A record of all contracts with added information about the CFPs
cancelled_contracts.csv       CSV        Contracts that were cancelled because one partner refused to sign it
signed_contracts.csv          CSV        Contracts that were actually signed
negotiations.csv              CSV        A record of all negotiations
breaches.csv                  CSV        A record of all breaches
stats.csv                     CSV        Helpful statistics about the state of the world at every timestep
                                         (e.g. N. negotiations, N. Contracts Executed, etc) in CSV format
stats.json                    JSON       Helpful statistics about the state of the world at every timestep
                                         (e.g. N. negotiations, N. Contracts Executed, etc) in JSON format
params.json                   JSON       The arguments used to run the world
logs.txt                      TXT        A log file giving details of most important events during the simulation
=======================    ========     ====================================


Tournament Command (negmas tournament)
--------------------------------------

The Tournament command (`tournament`) allows you to run a tournament between different agents in some world and
compare their relative performance. The tool is general enough to support several world types but currently only the
ANAC 2019 SCML (`anac2019`) configuration is supported.


You can get help on this tool by running:

.. code-block:: console

    $ tournament --help


These are the *optional* arguments of this tool:

=================================   =================================================
    Argument                         Meaning
=================================   =================================================
-n, --name TEXT                     The name of the tournament. The special
                                    value "random" will result in a random name [default: random]
-s, --steps INTEGER                 Number of steps.  [default: 60]
-f, --config TEXT                   The config to use. Default is ANAC 2019 [default: anac2019]
-t, --timeout INTEGER               Timeout after the given number of seconds (0 for infinite)  [default: 0]
--runs INTEGER                      Number of runs for each configuration [default: 5]
--max-runs INTEGER                  Maximum total number of runs. Zero or negative numbers mean no limit  [default:-1]
--randomize / --permutations        Random worlds or try all permutations up to max-runs  [default: False]
-c, --competitors TEXT              A semicolon (;) separated list of agent types to use for the competition.
                                    [default:negmas.apps.scml.DoNothingFactoryManager;negmas.apps.scml.GreedyFactoryManager]
--parallel / --serial               Run a parallel/serial tournament on a single machine  [default: True]
--distributed / --single-machine    Run a distributed tournament using dask [default: False]
-l, --log TEXT                      Default location to save logs (A folder will be created under it)  [default:~/negmas/logs/tournaments]
--verbose INTEGER                   verbosity level (from 0 == silent to 1 == world progress)  [default: 0]
--configs-only / --run              configs_only  [default: False]
--reveal-names / --hidden-names     Reveal agent names (should be used only for debugging)  [default: False]
--ip TEXT                           The IP address for a dask scheduler to run the distributed tournament.
                                    Effective only if --distributed  [default: 127.0.0.1]
--port INTEGER                      The IP port number a dask scheduler to run
                                    the distributed tournament. Effective only
                                    if --distributed  [default: 8786]
--help                              Show help message and exit.
=================================   =================================================


Upon completion, a complete log and several statistics are saved in a new folder under the `log folder` location
specified by the `--log` argument (default is negmas/logs/tournaments under the HOME directory). To avoid over-writing earlier
results, a new folder will be created for each run named by the current date and time. The
folder will contain the following files:


=================           ========     ====================================
 File/Folder Name             Format         Content
=================           ========     ====================================
configs                     FOLDER       Contains one json file for each world run tried during the tournament. You can
                                         re-run this world using `run_world` function in the `tournament` module.
params.json                 JSON         The parameters used to create this tournament
scores.csv                  CSV          Scores of every agent in every world
total_scores.csv            CSV          Scores of every agent **type** averaged over all runs
winners.csv                 CSV          Winner *types* and their average scores
ttest.csv                   CSV          Results of a factorial TTEST comparing the performance of all agent *types*
=================           ========     ====================================

Other than these files, a folder with the same number as the corresponding config file in the configs folder, keeps full
statistics/log of every world (see the `SCML World Runner` section for the contents of this folder.



