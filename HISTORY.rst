History
=======

Release 0.5.1
-------------

- [situated] Adding graph construction and drawing
- [situated] renaming contracts in TimeInAgreement to contracts_per_step to avoid name clashes
- [situated] Adding fine control for when are contracts to be signed relative to different main events during the simulation
- [situated] adding basic support for partial contract signature (contracts that are signed by some of the partners are now treated as unsigned until the rest of the partners sign them).
- [situated] changing signatures into a dict inside Contract objects to simplify searching them

Release 0.5.0
-------------

- [genius] adding ParsCat as a Genius Agent
- [situated] added agent specific logs to situated
- [situated] adding simulation steps after and before entity/contract execution
- [situated] adding ignore_contract to ignore contracts completely as if they were never concluded
- [siutated] adding dropped contracts to the possible contract types. Now contracts can be concluded, signed, nullified, erred, breached, executed, and dropped
- [situated] Correcting the implementation of TimeInAgreementMixin taking into account batch signing
- [situated] Added aggregate management of contract signing through sign_all_contracts and on_contracts_finalized. We still support the older sign_contract and on_contract_signed/cancelled as a fallback if sign_all_contracts and on_contracts_finalized are not overriden
- [situated] Now contract related callbacks are called even for contracts ran through run_negotaiation(s)
- [situated] added batch_signing to control whether contracts are signed one by one or in batch. Default is batch (that is different from earlier versions)
- [situated] added force_signing. If set to true, the sign_* methods are never called and all concluded negotiations are immediately considered to be signed. The callbacks on_contracts_finalized (and by extension on_contract_signed/cancelled) will still be called so code that used them will still work as expected. The main difference is in timing.
- replacing -float("inf") with float("-inf") everywhere

Release 0.4.4
-------------

- replacing -float("inf") with float("-inf") everywhere
- [core] avoid importing elicitation in the main negmas __init__
- [concurrent] renaming nested module to chain
- [documentation] improving module listing
- [concurrent] Adding a draft implementation of MultiChainMechanism with the corresponding negotiator
- [elicitors] adding a printout if blist is not available.
- [documentation] improving the structure of module documentation
- [core] Defaulting reserved_value to -inf instead of None and removing unnecessary tests that it is not None
- [core] default __call__ of UtilityFunction now raises an exception if there is an error in evaluating the utility value of an offer instead or returning None
- [core] Adding utility_range and outcome_with_utility as members of UtilityFuction. Global functions of the same name are still there for backward compatibility
- [CLI] improving path management for windows environments.
- black formatting


Release 0.4.3
-------------

- [mechainsms] Allowing mechanisms to customize the AMI for each negotiator
- [concurrent] Adding ChainNegotiationMechanism as a first example of concurrent negotiation mechanisms.
- [core] avoiding an import error due to inability to compile blist in windows
- [core] removing the global mechanisms variable and using an internal _mechanism pointer in AMI instead.

Release 0.4.2
-------------

- [situated] Adding events to logging and added the main event types to the documentation of the situated module
- [situated] Do not create log folder if it is not going to be used.
- [negotiators] adding parent property to negotiator to access its controller

Release 0.4.1
-------------

- [Situated] adding accepted_negotiations and negotiation_requests to Agent (see the documentation for their use).
- [Situated] Now running_negotiations will contain both negotiations requested by the agent and negotiations accepted by it.
- [helpers] Adding microseconds to unique_name when add_time is True
- [Setup] separating requirements for elicitation and visualization to avoid an issue with compiling blist on windows machines unnecessarily if elicitation is not used.
- [core] adding is_discrete as an alias to is_countable in Issue
- [style] styling the mediated negotiators with black
- [core] resolving a bug in random generation of outcomes for issues with a single possible value
- [situated] resolving a bug that caused negotiations ran using run_negotiations() to run twice
- [core] making SAO mechanism ignore issue names by default (use tuples instead of dicts) for negotiation
- [core] allowed json dumping to work with numpy values
- [bug fix] Random Utility Function did not have a way to get a reserved value. Now it can.
- [core] Merging a pull request: Add mediated protocols
- [core] using num_outcomes instead of n_outcomes consistently when asking for n. outcomes of a set of issues
- [core] improving the robustness of Issue by testing against Integral, Real, and Number instead of int and float for interoperability with numpy
- [core] converted Issue.cardinality to a read-only property
- [core] converted Issue.values to a read-only property
- [core] improving the implementation of Issue class. It is now faster and supports Tuple[int, int] as values.
- [doc] preventing setting theme explicitly on RTD
- [doc] minor readme edit
- [doc] correcting readme type on pypi


Release 0.4.0
--------------

- Moving the SCML world to its own repository (https://github.com/yasserfarouk/scml)

Release 0.3.9
-------------

- Minor updates to documentation and requirements to avoid issues with pypi rendering and Travis-CI integration.

Release 0.3.8
-------------

- [Core][SAO] allowed AspirationNegotiator to work using sampling with infinite outcome spaces by not presorting.
- [Core][Outcome] bug fix in outcome_as_tuple to resolve an issue when the input is an iterable that is not a tuple.
- Documentation update for AspirationNegotiator

Release 0.3.7
-------------

- [Core][Tutorials] fix documentation of "Running existing negotiators"
- [Core][Utility] fixing a bug in xml() for UtilityFunction
- [Core][Documentation] adding documentation for elicitors, and modeling
- [Core][Genius] allowing Genius negotiators to be initialized using a ufun instead of files.
- [Core][Genius] Adding some built-in genius negotiators (Atlas3, AgentX, YXAgent, etc)
- [Core][Modeling] restructuring modeling into its own packages with modules for utility, strategy, acceptance and future modeling.
- [Core][Modeling] Adding regression based future modeling
- adding python 3.8 to tox
- [Core][Outcomes] adding functions to generate outcomes at a given utility, find the range of a utility function, etc
- [Core] restoring compatibility with python 3.6
- [Core][Elicitation, Modeling] Added utility elicitation and basic acceptance modeling (experimental)


Release 0.3.6
-------------

- Documentation Update.
- Adding LinearUtilityFunction as a simple way to implement linear utility functions without the need to use
  LinearUtilityAggregationFunction.
- [Setup] Removing dash dependency to get TravisCI to work
- [Core] Correcting the implementation of the aspiration equation to match Baarslag's equation.
- updating the requirements in setup.py
- [Visualizer] Adding visualizer basic interface. Very experimental
- Adding placeholders for basic builtin entities
- [Core] basic tests of checkpoints
- [Core] adding time to info when saving a checkpoint and smaller improvments
- [Core] updating the use of is_continuous to is_countable as appropriate (bug fix)
- [Core] exposing load from helpers
- [Core] testing is_countable
- [SingleText] renaming is_acceptable to is_acceptable_as_agreement
- [Core] Sampling with or without replacement from issues with values defined by a callable now return the same result
- [Core] Allowing creator of AspirationNegotiator to pass max/min ufun values
- [Core] Adding Negotiator.ufun as an alias to Negotiator.utility_function
- [Core] Allowing agreements from mechanisms to be a list of outcomes instead of one outcome
- [Core] adding current_state to MechanismState
- [Situated] [bug fix] run_negotiations was raising an exception if any partner refused to negotiation (i.e. passed a None negotiator).
- [Core][Outcomes] Adding support for issues without specified values. In this case, a callable must be given that can generate random values from the unknown issue space. Moreover, it is assumed that the issue space is uncountable (It may optionally be continuous but it will still be reported as uncountable).
- [Core] Implementing checkpoint behavior in mechanisms and worlds.
- Added checkpoint and from_checkpoint to NamedObject.
- Added CheckpointMixin in common to allow any class to automatically save checkpoints.
- [Core][Genius] Resolving a bug that prevented genius negotiators from starting.
- [SCML] converted InputOutput to a normal dataclass instead of it being frozen to simplify checkpoint implementation.
- [Core] Allow agents to run_negotiation or run_negotiations when they do not intend to participate in the negotiations.
- [Mechanisms] Adding Mechanism.runall to run several mechanisms concurrently
- [SAO] Added Waiting as a legal response in SAO mechanism
- [SAO] Added SAOSyncController which makes it easy to synchronize response in multiple negotiations
- [Situated] Correcting the implementation of run_negotiations (not yet tested)
- [SAO] adding the ability not to consider offering as acceptance. When enabled, the agent offering an outcome is not considered accepting it. It will be asked again about it if all other agents accepted it. This is a one-step free decommitment
- [Situated] exposing run_negotiation and run_negotiations in AgentWorldInterface
- [Situated] bug fix when competitor parameters are passed to a multistaged tournament
- [Situated] Avoiding an issue with competitor types that do not map directly to classes in tournament creation
- [Core][Situated] adding type-postfix to modify the name returned by type_name property in all Entities as needed. To be used to distinguish between competitors of the same type with different parameters in situated.
- [Core][Situated] using correct parameters with competitors in multistage tournaments
- [Core][Single Text] deep copying initial values to avoid overriding them.
- [Core][Common] Added results to all mechanism states which indicates after a negotiation is done, the final results. That is more general than agreement which can be a complete outcome only. A result can be a partial outcome, a list of outcomes, or even a list of issues. It is intended o be used in MechanismSequences to move from one mechanims to the next.
- added from_outcomes to create negotiation issues from outcomes
- updating nlevelscomparator mixin


Release 0.3.5
-------------

- [Core][SingleText] Adding single-text negotiation using Veto protocol
- [Core][Utilities] correcting the implementation of is_better
- [Core][Negotiators] Adding several extra honest negotiators that map functionality from the utility function. These are directly usable in mediated protocols
- bug fix: Making sure that step_time_limit is never None in the mechanism. If it is not given, it becomes -inf (the same as time_limit)
- [Core][Utilities] Adding several comparison and ranking methods to ufuns
- [Core][Event] improving the notification system by adding add_handler, remove_handler, handlers method to provide moduler notification handling.
- removing unnecessary warning when setting the ufun of a negotiator after creation but before the negotiation session is started


Release 0.3.4
-------------

- Adding NoResponsesMixin to situated to simplify development of the simplest possible agent for new worlds


Release 0.3.3
-------------

- time_limit is now set to inf instead of None to disable it
- improving handling of ultimatum avoidance
- a round of SAO now is a real round in the sense of Reyhan et al. instead of a single counteroffer
- improved handling of NO_RESPONSE option for SAO
- updates to help with generalizing tournaments
- updating dependencies to latest versions
- Bump notebook from 5.7.4 to 5.7.8 in /docs
- Bump urllib3 from 1.24.1 to 1.24.2 in /docs



Release 0.3.2
-------------

- updating dependencies to latest versions

Release 0.3.1
-------------

- [Situated] Correcting multistage tournament implementation.

Release 0.3.0
-------------
- [Situated] adding StatsMonitor and WorldMonitor classes to situated
- [Situated] adding a parameter to monitor stats of a world in real-time
- [Situated] showing ttest/kstest results in evaluation (negmas tournament commands)
- [SCML] adding total_balance to take hidden money into account for Factory objects and using it in negmas tournament and negmas scml
- [SCML] enabling --cw for collusion
- [SCML] adding hidden money to agent balance when evaluating it.
- [SCML] adding more debugging information to log.txt
- [Situated] adding multistage tournaments to tournament() function
- [Situated] adding control of the number of competitor in each world to create_tournament() and to negmas tournament create command
- [Core] avoid invalid or incomplete outcome proposals in SAOMechanism
- [Situated] adding metric parameter to evaluate_tournaments and corrsponding tournament command to control which metric is used for calculating the winner. Default is mean.
- [SCML] adding the ability to prevent CFP tampering and to ignore negotiated penalties to SCMLWorld
- [SCML] adding the possibility of ignore negotiated penalty in world simulation
- [SCML] saving bankruptcy events in stats (SCML)
- [SCML] improving bankruptcy processing
- [SCML] deep copying of parameters in collusion
- [Situated] saving extra score stats in evaluate_tournament
- [Core] avoiding a future warning in pandas
- [Situated] more printing in winners and combine commands
- [Situated] removing unnecessary balance/storage data from combine_tournament_stats
- [Situated] adding aggregate states to evaluate_tournament and negmas tournament commands
- [Situated] adding kstest
- [Situated] adding and disabling dependent t-tests to evaluate_tournament
- [Situated] adding negmas tournament combine to combine and evaluate multiple tournaments without a common root
- [Situated] avoiding an exception if combine_tournament is called with no scores
- [Situated] always save world stats in tournaments even in compact mode
- [SCML] reversing sabotage score
- [SCML] correcting factory number capping
- [SCML] more robust consumer
- [Core] avoid an exception if a ufun is not defined for a negotiator when logging
- [SCML] controlling number of colluding agents using --agents option of negmas tournament create
- [SCML] changing names of assigned worlds and multiple runs to have a unique log per world in tournament
- [SCML] controlling warnings and exception printing
- [SCML] increasing default world timeout by 50%
- [SCML] removing penalty processing from greedy
- [Core] avoid negotiation failure for negotiator exceptions
- [SCML] correcting sabotage implementation
- [CLI] adding winners subcommand to negmas tournament
- [CLI] saving all details of contracts
- [CLI] adding --steps-min and --steps-max to negmas tournament create to allow for tournaments with variable number of steps
- [CLI] removing the need to add greedy to std competition in anac 2019
- [CLI] saving log path in negmas tournament create
- [CLI] removing errroneous logs
- [CLI] enabling tournament resumption (bug fix)
- [CLI] avoiding a problem when trying to create two tournaments on the same place
- [CLI] fairer random assignment
- [CLI] more printing in negmas tournament
- [CLI] using median instead of mean for evaluating scores
- [CLI] Allowing for passing --world-config to tournament create command to change the default world settings
- [CLI] adding a print out of running competitors for verbose create_tournament
- [CLI] adding --world-config to negmas scml
- [CLI] displaying results of negmas tournament evaluate ordered by the choosen metric in the table.
- [CLI] preventing very long names
- [CLI] allowing for more configs/runs in the tournament by not trying all permutations of factory assignments.
- [CLI] adding --path to negmas tournament create
- [CLI] more printing in negmas tournament
- [CLI] reducing default n_retrials to 2
- [CLI] changing optimism from 0.0 to 0.5
- [CLI] setting reserved_value to 0.0
- [CLI] run_tournament does not call evaluate_tournament now
- [SCML] always adding greedy to std. competitions in negmas tournament
- [SCML] reducing # colluding agents to 3 by default
- [CLI] restructuring the tournament command in negmas to allow for pipelining and incremental running of tournaments.
- [SCML] adding DefaultGreedyManager to manage the behavior of default agents in the final tournament
- [CLI] avoiding overriding tournament folders if the name is repeated
- [SCML] avoiding missing reserved_value in some cases in AveragingNegotiatorUfun
- [CLI] adding the ability to control max-runs interactively to negmas tournament
- [CLI] adding the ability to use a fraction of all CPUs in tournament with parallel execution
- [SCML] exceptions in signing contracts are treated as refusal to sign them.
- [SCML] making contract execution more robust for edge cases (quantity or unit price is zero)
- [SCML] making collusion tournaments in SCML use the same number of worlds as std tournaments
- [Situated] adding ignore_contract_execution_excptions to situated and apps.scml
- [CLI] adding --raise-exceptions/ignore-exceptions to control behavior on agent exception in negmas tournament and negmas scml commands
- [SCML] adding --path to negmas scml command to add to python path
- [SCML] supporting ignore_agent_exceptions in situated and apps.scml
- [Situated] removing total timeout by default


Release 0.2.25
--------------
- [Debugging support] making negmas scml behave similar to negmas tournament worlds
- [Improved robustness] making insurance calculations robust against rounding errors.
- [Internal change with no behavioral effect] renaming pay_insurance member of InsuranceCompany to is_insured to better document its nature
- [Debugging support] adding --balance to negmas scml to control the balance


Release 0.2.24
--------------
- separating PassThroughNegotiator, PassThroughSAONegotiator. This speeds up all simulations at the expense
  of backward incompatibility for the undocumented Controller pattern. If you are using this pattern, you
  need to create PassThroughSAONegotiator instead of SAONegotiator. If you are not using Controller or you do not know
  what that is, you probably safe and your code will just work.
- adding logging of negotiations and offers (very slow)
- preventing miners from buying in case sell CFPs are posted.
- avoiding exceptions if the simulator is used to buy/sell AFTER simulation time
- adding more stats to the output of negmas scml command
- revealing competitor_params parameters for anac2019_std/collusion/sabotage. This parameter always existed
  but was not shown in the method signature (passed as part of kwargs).

Release 0.2.23
--------------

- Avoiding backward incompatibility issue in version 0.2.23 by adding INVALID_UTILITY back to both utilities
  and apps.scml.common

Release 0.2.22
--------------

- documentation update
- unifying the INVALID_UTILITY value used by all agents/negotiators to be float("-inf")
- Added reserved_value parameter to GreedyFactoryManager that allows for control of the reserved value used
  in all its ufuns.
- enable mechanism plotting without history and improving plotting visibility
- shortening negotiator names
- printing the average number of negotiation rounds in negmas scml command
- taking care of negotiation timeout possibility in SCML simulations

Release 0.2.21
--------------

- adding avoid_free_sales parameter to NegotiatorUtility to disable checks for zero price contracts
- adding an optional parameter "partner" to _create_annotation method to create correct contract annotations
  when response_to_negotiation_request is called
- Avoiding unnecessary assertion in insurance company evaluate method
- passing a copy of CFPs to on_new_cfp and on_cfp_removal methods to avoid modifications to them by agents.

Release 0.2.20
--------------

- logging name instead of ID in different debug log messages (CFP publication, rejection to negotiate)
- bug fix that caused GreedyFactoryManagers to reject valid negotiations

Release 0.2.19
--------------

- logging CFPs
- defaulting to buying insurance in negmas scml
- bug resolution related to recently added ability to use LinearUtilityFunction created by a dict with tuple
  outcomes
- Adding force_numeric to lead_genius_*

Release 0.2.18
--------------

- minor updates


Release 0.2.17
--------------

- allowing anac2019_world to receive keyword arguments to pass to chain_world
- bug fix: enabling parameter passing to the mechanism if given implicitly in MechanismFactory()
- receiving mechanisms explicitly in SCMLWorld and any other parameters of World implicitly

Release 0.2.16
--------------

- bug fix in GreedyFactoryManager to avoid unnecessary negotiation retrials.

Release 0.2.15
--------------

- Minor bug fix to avoid exceptions on consumers with None profile.
- Small update to the README file.


Release 0.2.14
--------------

- Documentation update
- simplifying continuous integration workflow (for development)

Release 0.2.13
--------------

- Adding new callbacks to simplify factory manager development in the SCM world: on_contract_executed,
  on_contract_breached, on_inventory_change, on_production_success, on_cash_transfer
- Supporting callbacks including onUfunChanged on jnegmas for SAONegotiator
- Installing jenegmas 0.2.6 by default in negmas jengmas-setup command

Release 0.2.12
--------------

- updating run scml tutorial
- tox setting update to avoid a break in latest pip (19.1.0)
- handling an edge case with both partners committing breaches at the same
  time.
- testing reduced max-insurance setting
- resolving a bug in contract resolution when the same agent commits
  multiple money breaches on multiple contracts simultaneously.
- better assertion of correct contract execution
- resolving a bug in production that caused double counting of some
  production outputs when multiple lines are executed generating the
  same product type at the same step.
- ensuring that the storage reported through awi.state or
  simulator.storage_* are correct for the current step. That involves
  a slight change in an undocumented feature of production. In the past
  produced products were moved to the factory storage BEFORE the
  beginning of production on the next step. Now it is moved AFTER the
  END of production of the current step (the step production was
  completed). This ensures that when the factory manager reads its
  storage it reflects what it actually have at all times.
- improving printing of RunningCommandInfo and ProductionReport
- regenerating setup.py
- revealing jobs in FactoryState
- handling a bug that caused factories to have a single line sometimes.
- revealing the dict jobs in FactoryState which gives the scheduled jobs
  for each time/line
- adding always_concede option to NaiveTitForTatNegotiator
- updating insurance premium percents.
- adding more tests of NaiveTitForTatNegotiator
- removing relative_premium/premium confusion. Now evaluate_premium will
  always return a premium as a fraction of the contract total cost not
  as the full price of the insurance policy. For a contract of value 30,
  a premium of 0.1 means 3 money units not 0.1 money units.
- adding --config option to tournament and scml commands of negmas CLI
  to allow users to set default parameters in a file or using
  environment variables
- unifying the meaning of negative numbers for max_insurance_premium to
  mean never buying insuance in the scheduler, manager, and app. Now you
  have to set max_insurance_premium to inf to make the system
- enforcing argument types in negmas CLI
- Adding DEFAULT_NEGOTIATOR constant to apps.scml.common to control the
  default negotiator type used by built-agents
- making utility_function a property instead of a data member of
  negotiator
- adding on_ufun_changed() callback to Negotiator instead of relying on
  on_nofitication() [relying on on_notification still works].
- deprecating passing dynamic_ufun to constructors of all negotiators
- removing special treatment of AspirationNegotiator in miners
- modifications to the implementation of TitForTatNegotiator to make it
  more sane.
- deprecating changing the utility function directly (using
  negotiator.utility_function = x) AFTER the negotiation starts. It is
  still possible to change it up to the call to join()
- adding negmas.apps.scml.DEFAULT_NEGOTIATOR to control the default negotiator used
- improved parameter settings (for internal parameters not published in the SCML document)
- speeding up ufun dumping
- formatting update
- adding ufun logging as follows:

  * World and SCMLWorld has now log_ufuns_file which if not None gives a file to log the funs into.
  * negmas tournament and scml commands receive a --log-ufuns or --no-log-ufuns to control whether
    or not to log the ufuns into the tournament/world stats directory under the name ufuns.csv

- adding a helper add_records to add records into existing csv files.


Release 0.2.11
--------------
- minor bug fix

Release 0.2.10
--------------

- adding more control to negmas tournaments:

   1. adding --factories argument to control how many factories (at least) should exist on each production
      level
   2. adding --agents argument to control how many agents per competitor to instantiate. For the anac2019std
      ttype, this will be forced to 1

- adding sabotage track and anac2019_sabotage to run it
- updating test assertions for negotiators.
- tutorial update
- completed NaiveTitForTatNegotiator implementation


Release 0.2.9
-------------

- resolving a bug in AspirationNegotiator that caused an exception for ufuns with assume_normalized
- resolving a bug in ASOMechanism that caused agreements only on boundary offers.
- using jnegmas-0.2.4 instead of jnegmas-0.2.3 in negmas jnegmas-setup command


Release 0.2.8
-------------

- adding commands to FactoryState.
- Allowing JNegMAS to use GreedyFactoryManager. To do that, the Java factory manager must inherit from
  GreedyFactoryManager and its class name must end with either GreedyFactoryManager or GFM


Release 0.2.7
-------------

- improving naming of java factory managers in log files.
- guaranteeing serial tournaments when java factory managers are involved (to be lifter later).
- adding links to the YouTube playlist in README
- adhering to Black style


Release 0.2.6
-------------

- documentation update
- setting default world runs to 100 steps
- rounding catalog prices and historical costs to money resolution
- better defaults for negmas tournaments
- adding warnings when running too many simulations.
- added version command to negmas
- corrected the way min_factories_per_level is handled during tournament config creation.
- added --factories to negmas tournament command to control the minimum number of factories per level.
- improving naming of managers and factories for debugging purposes
- forcing reveal-names when giving debug option to any negmas command
- adding short_type_name to all Entity objects for convenient printing

Release 0.2.5
-------------

- improvements to ufun representation to speedup computation
- making default factory managers slightly less risky in their behavior in long simulations and more risky
  in short ones
- adding jnegmas-setup and genius-setup commands to download and install jenegmas and genius bridge
- removing the logger mixin and replaced it with parameters to World and SCMLWorld
- added compact parameter to SCMLWorld, tournament, and world generators to reduce the memory footprint
- added --compact/--debug to the command line tools to avoid memory and log explosion setting the default to
  --compact
- improving implementation of consumer ufun for cases with negative schedule
- changing the return type of SCMLAWI.state from Factory to FactoryState to avoid modifying the original
  factory. For efficiency reasons, the profiles list is passed as it is and it is possible to modify it
  but that is forbidden by the rules of the game.
- Speeding up and correcting financial report reception.
- Making bankruptcy reporting system-wide
- avoiding execution of contracts with negative or no quantity and logging ones with zero unit price.
- documentation update
- bug fix to resolve an issue with ufun calculation for consumers in case of over consumption.
- make the default behavior of negmas command to reveal agent types in their names
- preventing agents from publishing CFPs with the ID of other agents
- documentation update
- improved Java support
- added option default_dump_extension to ~/negmas/config.json to enable changing the format of dumps from json to yaml.
  Currently json is the default. This included adding a helper function helpers.dump() to dump in the selected format
  (or overriding it by providing a file extension).
- completing compatibility with SCML description (minor change to the consumer profile)
- added two new options to negmas tournament command: anac2019std and anac2019collusion to simulate these two tracks of
  the ANAC 2019 SCML. Sabotage version will be added later.
- added two new functions in apps.scml.utils anac2019_std, anac2019_collusion to simulate these two tracks of the ANAC
  2019 SCML. Sabotage version will be added later.
- added assign_managers() method to SCMLWorld to allow post-init assignment of managers to factories.
- updating simulator documentation

Release 0.2.2
-------------

* modifications to achieve compatibility with JNegMAS 0.2.0
* removing the unnecessary ufun property in Negotiator

Release 0.2.0
-------------

* First ANAC 2019 SCML release
* compatible with JNegMAS 0.2.0

Release 0.1.45
--------------

* implemented money and inventory hiding
* added sugar methods to SCMLAWI that run execute for different commands: schedule_production, stop_production, schedule_job, hide_inventory, hide_money
* added a json file ~/negmas/config.json to store all global configs
* reading jar locations for both jnegmas and genius-bridge from config file
* completed bankruptcy and liquidation implementation.
* removed the unnecessary _world parameter from Entity
* Added parameters to the SCML world to control compensation parameters and default price for products with no catalog prices.
* Added contract nullification everywhere.
* updated documentation to show all inherited members of all classes and to show all non-private members
* Removing the bulletin-board from the public members of the AWI

Release 0.1.42
--------------

* documentation improvement
* basic bankruptcy implementation
* bug fixes

Release 0.1.40
--------------

* documentation update
* implementing bank and insurance company disable/enable switches
* implementing financial reports
* implementing checks for bankruptcy in all built-in agents in SCML
* implementing round timeout in SAOMechanism

Release 0.1.33
--------------

* Moving to Travis CI for continuous integration, ReadTheDocs for documentation and Codacy for code quality

Release 0.1.32
--------------

* Adding partial support to factory manager development using Java
* Adding annotation control to SCML world simulation disallowing factory managers from sending arbitrary information to
  co-specifics
* Removing some unnecessary dependencies
* Moving development to poetry. Now we do not keep a setup.py file and rely on poetry install

Release 0.1.3
-------------

* removing some unnecessary dependencies that may cause compilation issues

Release 0.1.2
-------------

* First public release
