History
=======

Release 0.10.8
--------------

* Adding exception handling and testing Cartesian Tournaments
* Adding ufun generators with controlled Pareto
* Stopping saving "stats.json" in tournaments. stats.csv contains the same information
* Allowing nash calculation without ufuns
* Exception handling in cartesian tournaments. Note that I assume that ignore_negotiator_exceptions can be passed to the mechanism class
* intin and floatin in the helpers to sample.
* Saving negotiator times in cartesian
* Adding execution_time to cartesian logs
* Finer conntrol on timing in Cartesian tournaments. Also recording negotiator times
* Adding negotiator_times to Mechanisms
* Finer control on mechanism printing
* Ignoring some typing errors
* Passing plot_params in cartesian_tournament

Release 0.10.7
--------------
* Saving path and controlling name-shortening in tournaments

Release 0.10.6
--------------
* Adding the ability to hide types in cartesian_tournament
* Name but not ID reveals type cartesian_tournament by default
* Removing neg/agent_names from NMI. Now negotiators have access to each other's ID but not name by default
* Correcting sorter negotiator

Release 0.10.5
--------------
* Minor bugfixes

Release 0.10.4
--------------
* Adding simple tournaments and testing them
* improving plotting and minor typing enhancement
* bugfix in time-based
* Reducing verbosity
* passing kwargs back to Negotitors
* Adding simple tournaments
* Fixing a pandas warning
* Renaming Scenario.agenda -> outcome_space

Release 0.10.3
--------------
* Switching to readthedocs.io
* doc update
* adding RandomAlwaysAcceptingNegotiator
* Upgrading CI tests to use python 3.12 by default
* avoid genius when calculating coverage
* Do not test notebooks before release (avoid genius)
* Empty results when a tournament has no scores

Release 0.10.2
--------------

* Adding RandomOfferGuaranteedAcceptance negotiator
* Fixing some failures in testing some genius agents
* [Snyk] Security upgrade pillow from 9.5.0 to 10.0.1
* [Snyk] Security upgrade werkzeug from 2.2.3 to 3.0.1
* [Snyk] Security upgrade pillow from 9.5.0 to 10.0.0
* fix: docs/requirements.txt to reduce vulnerabilities
* Updating tutorials, adding a tournament there
* Fixing an installation bug: hypothesis was needed to run test_situated under negmas/tests. This prevented users from running the fast set of tests after installation.
* cartesian_tournament to run a simple tournament
  - cartesian_tournament runs a simple tournament similar to Genius tournaments.
  - create_cartesian_tournament creates a simple Cartesian tournament but does not run it. To run the tournament, call run_tournament passing it the returned path from create_cartesian_tournament.
* fix: requirements-visualizer.txt to reduce vulnerabilities
* Group2 defaults to Y2015Group2 in gnegotaitors
* adding Ateamagent beside AteamAgent
* Correcting few gnegotiator names
* standardizing some gnegotiator names
* renaming ateamAgent -> AteamAgent in genius
* Adding some missing Genius negotiators to gnegotiators.py

Release 0.10.1
--------------

* various bugfixes
* Updating ginfo (Genius Information) with ANAC competition information up to the end of genius support and partial information for geniusweb years

Release 0.10.0
--------------

* removing offer from SAO's respond() method.
* allowing users to step worlds from the point of view of a set of agents ignoring simulation step boundaries and passing external actions if needed. See World.step() for details.

Release 0.9.8
-------------

* Restructuring tests
* Using Numba only with python 3.10
* Always using with when opening files
* Adding more info about anac results
* [SAO] Completely removing support for avoid_ultimatum
* [SAO] Adding fallbacks to respond() calls in SAO to support the API with and
  without source. The later API will be dropped later.
* [Preferences] Adding has_ufun to Rational to check if it has a `BaseUtilityFunction`
  as its preferences.
* [Genius] More details on errors from genius bridge
* [Genius] bugfix when starting genius negotitauions with no n-steps (sometims)
* [CLI] supporting genius negotiators in the negotiate.py cli
	Pass -n geinus.<agent-name> or genius:<agent-name>
	The agent-name can be just the full java class name, or a simplified
	version that is all lower without the word agent and without _

Release 0.9.7
-------------
* minor bugfixes

Release 0.9.6
-------------

* [python] Supporting 3.11 and dropping support for 3.8 and 3.9
* [test] Adding 3.11 to tests
* [major] Adding Generalized Bargaining Protocols
* [buffix] testing saving exceptions in SAO
* [bugfix] Avoid failure if a config folder for negmas does not exist
* [minor] avoid a warning when setting preferences explicitly
* [minor] Moving shortest_unique_names to strings.py from misc.py
* [cli] renaming the 50% column to median in scores
* [feature] Adjustable config paths. Now all paths and configs are adjustable using environement variables, a global json file or a local json file. See `negmas_config` under `negmas.config` for more details.
* [feature] Adding calculation of Kalai-points, max-welfare-points and max-relative-welfare points and making nash_points return all nash points (previously we had nash_point() which returned just one)

Release 0.9.5
-------------

* defaulting to full type name in NamedObject
* Removing a couple of warnings

Release 0.9.4
-------------

* removing dependence on tqdm and printing by rich
* using rich progressbar in run_with_progress

Release 0.9.3
-------------

* feature: added serialization to yaml and json in Scenario
* feature: adding shorten_type_field to serialize()
* feature: Adding future annotations for 3.8 compatibility   (tests)
* bugfix: resetting() controllers now kills negs.
* bugfix: Ensuring that counter_all() is called every step for SAOSyncController
* enhancement: extra check in SyncController
* enhancement: Rejects offers for unregistered negotiators
* bugfix: SAOSyncController not receiving first_proposals before counter_all
* enhancement: SAOMechanism extra assertions
* enhancement: improved type annotations
* feature: Adding ExpAspiration time curve
* feature: Adding more acceptance strategies
* enhancement: Restructuring the situated module

Release 0.9.2
-------------

* Improving caching
* Renaming modeling advanced module to models
* optimizing imports
* removing the need for extra_state()
* changing some of the core classes to use attrs
* switching to setup.cfg and adding pytoml.yml
* performance improvement and code sorting
* more basic acceptance strategies

Release 0.9.1
-------------

* caching offer in the offering strategy
* Avoids repeated calls to the offering strategy in SAOModuler if it was
  called for example by the acceptance strategy then again by the mechanism.
* Purifying protocols
* correcting info for ANAC 2014
* Implementing not for AcceptanceStrategy and adding RejectionStrategy to invert the decision of an AcceptanceStrategy
* Supporting normalized ufuns in TFT
* Added ZeroSumModel as a simple opponent model (assumes a zero-sum negotiation)
* Refactored NTFT to use this model
* Removed the unnecesasry ConcessionEstimator classes

Release 0.9.0
-------------

This is a major release and it is **not** backward compatible. Please reference
the upgrade guide at the upgrdade guide_.

.. _guide: http://yasserm.com/negmas/upgrade_guide.html

Some of the most important changes are:

* Introduces the `ModularNegotiator` and `Component` objects to simplify reuse of negotiation strategies through composition instead of inheritance.
* Restructures most of the code-base for readability.
* Completed the tutorial.
* Simplified several key methods.
* Introduced the `SAOModularNegotiator`, `MAPNegotiator`, `BOANegotiator` as basic modular negotiators for the SAO mechanism as well as reusable components like `AcceptanceStrategy`, and `OfferingStrategy`


Release 0.8.9
-------------

* [sao] improvement to the plot() method of SAOMechanism
* [genius] Almost complete rewriting of the genius-bridge. Now we are
  compatible with genius*bridge v0.2.0
* [genius] Renaming get_genius_agents() to get_anac_agents()
* [genius] Updating TEST_FAILING_NEGOTIATORS and adding ALL_GENIUS_NEGOTIATORS,
  ALL_BASIC_GENIUS_NEGOTIATORS to ginfo
* [core] Adding nash_point() to find the nash point of a set of ufuns (within
  the pareto frontier)
* [bugfix] plotting SAOMechanism instances with continuous Issue spaces work
  now
* [genius] Stricter GeniusNegotiator.  If strict=True is given to a
  GeniusNegotiator (or in an n_steps limited negotaition with strict not given
  at all), more tests are incorporated to make sure that the Genius agent is
  getting what it expects all the time.
* [sao] relative_time matches Genius behavior.  relative_time was equal to
  step/n_steps now it is (step+1)/(n_steps+1) This is only in the case of using
  n_steps as a limit of a mechanism.
* [tests] Extracting long genius tests out and running genius tests in CI
* [genius] Added is_installed to GeniusBridge and genius_bridge_is_installed()
* [bugfix] Handling wrong time perception in Genius agents
* [genius] Adding wxtra warnings for common timinig problems in SAO
    * A warning is now raised in either of the following cases:
        1. A mechanism is created with neither a time_limit nor n_step set
        2. A Genius agent tries to join a mechanism with both time_limit and
           n_steps set
    * We stopped using timeline.increment() inside the genius bridge and now
      pass the round number (step in negmas terms) directly from negmas.
      This should avoid any possibility of double counting
* [sao] Adding enforce_outcome_type to SAOMechanism
* [sao] Adding enforcement of issue value types SAOP
* [sao] Adding the ability to cast_outcome to Mechanism
* [genius] Adding relative_time to GeniusNegotiator which checks the time as perceived by the Genius Agent inside the JVM
* [genius] Improving the way tuple ouctomes are handled in GeniusNegotiator
* [tournament] Allowing truncated_mean in eval
* [cli] adding truncated_mean as a possible metric


Release 0.8.8
-------------

* [sao] Treating `None` as `(REJECT_OFFER, None)` in responses from counter_all()

Release 0.8.7
-------------

* [core] better normalization for random Linear*UFun
* [helpers] single_thread() context manager
* [bugfix] Partner params incorrectly passed in NegWorld

Release 0.8.6
-------------

* [core] Adding to_dict/from_dict to all ufun types
* [core] Better random LinearAdditiveUtilityFunction
* [core] better implementation of stepall and runall
* [core] implementing keep_order=False for stepall()
* [tournaments] Adding negotiation tournaments.
* [situated] shuffle_negotiations option in World
* [bugfix] SAOSyncController never loses offers

Release 0.8.5
-------------

*  [sao] Avoiding an issue with avoid-ultimatum if all agents sent None as their first offer
*  [situated] bugfix in reporting mechanism exceptions
*  [helpers] Adding one-thread mode
*  [situated] enable agent printing by default
*  [tournament] not setting log_negotiations for forced logs

Release 0.8.4
-------------

* [tournaments] udpating log_negotiations when forced to save logs
* [tournaments] saving negotiations
* [sao] bugfix AsporationController best_outcome
* [sao] avoiding repetition in trace and offers at the end
* [genius] disabling AgentTD
* [genius] disabling GeneKing
* [genius] testing only confirmed passing negotiators
* [genius] correcting some genius class names
* [testing] stronger genius testing
* [testing] shortening the time allowed for genius negotiators in tests

Release 0.8.3
-------------

* [genius] allowing the ufun of genius agents to be set anytime before negotiation start
* [core] bugfix. Type of issue value may be incorrect when exporting to xml
* formatting
* [bugfix] correcting getting partner agent names in controllers
* [elicitation] pandora unknowns sometimes were not set
* [helpers] bugfix in serialization: correctly serializing cloud pickalable objects
* [bugfix] some SAO mechanisms where timeouting without timeout set
* [genius] updating the set of tested genius agents

Release 0.8.2
-------------

* [sao] adding the ability to use sync-calls in SAOMechanism
* [situated] fixing not showing last step's conracts in draw

Release 0.8.1
-------------

*  [sao][bugfix] correctly handling unexpected timeouts (Usually Genius)
*  [minor] using warnings.warn instead or print whne appropriate
*  [sao] improving synchronous controller handling
*  [sao] correcting history storage. Avoiding repetition of the last offer sometimes
*  [core] better handling of extra state in Mechanism
*  [sao] default waiting is now 0 step and correcting times calculation
*  [tournament] [bugfix] correcting str conversion for TournamentResults
*  [sao] [bugfix] correcting storage of history in state
*  [core] Supporting python 3.9
*  [situated] bugfix when agents make exceptions (time was ignored)
*  [situated] forcing all agents not to print anything
*  [situated] forcing all agents not to print anything

Release 0.8.0
-------------

* [minor] ignoring some intentionally broken type checks
* [setup] Adding cloudpickle as a requirement for setup
* [situated] revealing all  methods of Agent in the AWI
* [genius] bugfix, forcing time_limit to be an int in genius
* [situated] Adding RunningNegotiationInfo to situated.__all__

Release 0.7.4
-------------

* [core] making the core SAONegotiator robust to missing ufuns.
* [core] allowing controllers to control the ID of negotiators
* [core] adding reset_timer to EventLogger and logging time
* [core] passing AMI to minmax [situated] reversing adapter and adapted
         names in Adapter to make sure that split(".")[-1] still gets the
         adapted name not the adapter name.
* [core] making Controller.negotiators return NegotiatorInfo
* [genius] bug fix in saving xml utils that broke the bridge
* [genius] get_genius_agents in genius.ginfo to find genius agents
* [situated] adding event logging to situated (unstable)
* [bugfix] removing color codes in log file (log.txt)
* [situated] adding more events (contracts/breaches)
* [testing] getting some genius related tests to pass
* [testing] avoiding failure on genius agents that cannot agree

Release 0.7.3
-------------

* [core] making the core SAONegotiator robust to missing ufuns.
* [core] allowing controllers to control the ID of negotiators
* [core] adding methods to find partner IDs and names
* [sao] Adding global_ufun to SAOSyncController
* [core] removing all all_contracts.csv from output keeping only contracts.csv withe full information.
* [core] Added serialization module for serializing objects in human readable format.
* [core] Added id as a parameter to all constructors of NamedObjects
* [core] dividing utilities.py into multiple modules
* This should not affect any external users.
* [core] removing an issue when deepcopying utility fucntions.
* [core] adding inverse_utility support
* [core] adding inverse ufun support
* [cli] removing unnecessry force flag
* [sao] adding allow_offering_just_rejected_offers
* [core] adding max_n_outcomes to Issue.sample
* adding parameters to mechanisms and worlds.
* [genius] improved the information on ANAC competition
* [genius] restructuring the module into a package
* [core] bugfix in LinearUtilityFunciton that calculated the weights
* incorrectly sometimes
* [genius] Adding close_gateway to GeniusBridge to close all connections
* [genius] Adding close_gateway to GeniusBridge to close all connections
* [genius] Added GeniusBridge with methods to control a bridge
* [genius] Now all GeniusNegotiator classes share the same bridge to avoid too much resource allocation but this may not be safe when running tournaments.
* [genius] compatible with bridge version 0.5
* [genius] compatible with bridge v0.3
* [genius] more exhaustive testing and resolving ending issue
* [genius] adding the skeleton to cancel unending agents
* [sao] allowing load_genius_domain to use any kwargs
* [core] adding imap to all mechanisms
* [core] Maps between issue name and index and back
* [core] Speeding issue enumeration
* [core] Enumerating faster for large outcome spaces.
* [core] Adding max_n_outcomes to functions that use outcome enumeration more consistently.
* [core] adding a warning for infinity ufun values
* [inout] bugfix a failure when reading some genius files

Release 0.6.15
--------------

* [tournaments] Default to faster tournaments
* [testing] Avoid failure on PyQT not installed
* [situated] agreement and contract validation:
  Agreement validation (is_valid_agreement) and contract validation
  (is_valis_valid_contract) are added to the World class. Using them
  a world designer can decide that an agreement (before signing) or
  a contract (after signing) is invalid and drop it so it is never
  executed. These contracts appear as 'dropped_contracts' in stats.
* [tournaments] Adding max_attempts parameter when running worlds.

Release 0.6.14
--------------

* [tournaments] Possible exclusion of competitors from dyn. non-comp.
* [tournaments] Adding dynamic non_competitors
* [situated] Allowing more return types from sign_all_contacts
* [tournaments] Avoid different stat lengths
* [situated, tournaments] Early break if time-limit is exceeded.
* [situated, tournaments] Early break if time-limit is exceeded.
* [situated, mechanisms, tournaments] Using perf_counter consistently to measure time.
* [situated,mechanisms] more robust relative time
* [setup] Removing installation of visualizer components in CI
* [tournaments] Avoid failure for empty stat files when combining tournaments
* [helpers] avoid trying to load empty files
* [tournament][bugfix] Error in concatenating multiple exceptions.
* [tournament][bugfix] Serial run was failing
* [situated] Avoiding relative_time > 1
* [mechanisms] Avoiding relative_time > 1
* [tournament] Saving temporary scores in tournaments by default
* [tournaments][bugfix] Tuples were causing exceptions when combining agent exceptions
* [bugfix] correcting NotImplementedError exception
* [situated] Avoid failure when returning non-iterable from sign_all_contracts
* [tournaments] better handling of continuation
* [tournament] Randomizing assigned config runs
* [tournament] adding extra exception and timing information to tournaments
* [docs] Documentation update
* [situated] Keeping details of who committed exceptions.
* [situated] For negotiation exceptions, the exception is registered for the agents
  owning all negotiators as it is not possible in World to know the
  negotiator from whom the exception originated.

Release 0.6.13
--------------

* [tournaments] defaulting to no logs or videos in tournaments.
* [base] bugfix: avoid calling parent in passthrough negotiator when it does not exist.
* [base] making PyQT optional

Release 0.6.12
--------------

* [docs] more tutorials and overview revampment
* [sao] Allowing max_wait to be passed as None defaulting to inf
* [sao] Passing the ufun to the meta-negotiator in SAOMetaNegotiatorController
* [base] unsetting the controller when killing a negotiator
* [base] setting default max_waits to infinity
* [base] defaulting to auto-kill negotiators in all controllers.py
* [base] Adding max_wait to void infinite loops with sync controllers

Release 0.6.11
--------------

* [base] removing a warning caused by passing dynamic_ufun
* [base] correctly passing ufun to all rational types
* [base] placeholder to support parallel runall in mechanism
* [base] LimitedOutcomesNegotiator does not offer what it will not accept
* [base] Bug fixes in Utilities and LimitedOutcomesNegotiator
* [performance] Caching first offers in SyncController.
* [performance] Reducing memory consumption of AspirationNegotiator
* [performance] Speeding up Mechanism.state
* [performance] Adding eval_all to UtilitityFunction to speedup multiple evaluations
* [docs] Improving the overview part of the documentation
* [docs] Documentation update
* [elicitation] Fixing documentation after renaming elicitors -> elicitation
* [elicitation] Adding AMI to elicitaition.User to know the step
* [elicitation] restructuring elicitors module and renaming it to elicitation
* [elicitation] correcting a bug in base elicitor
* [installation] Resolving an issue when blist is not installed
* [installation] Adding gif to requirements
* [installation] warn if gif generation failed
* reformatting and import optimization
* Removing eu from SAONegotiator because we have no opponent_models yet

Release 0.6.10
--------------

* [base] Refactoring to allow Negotiators, Controllers and Agents to have UFuns. Introduced the Rational type wich is a NamedObject with a ufun. Now Negotiators, Controllers, and Agents are all Rational types. This makes it easier to define ufuns for any of these objects.
  on_ufun_changed is now called immediately when the ufun is set but if an AMI is not found, the _ufun_modified flag is set and the rational object is responsible of calling on_ufun_changed after the nmi is defined. For Negotiators, this happen automatically
* [situated] Making negotiation requests with an empty output-space fail
* [testing] Correcting some testing edge casease
* [base] converting outcome_type in UtilityFunction to a property. To allow complex ufuns to set the outcome_type of their children
  recursively.
* [docs]. Using "Outocme" instead of Outcome for type hints. To avoid the nonsensical long types that were appearing in the
  documentation because Sphinx cannot find the Outcome type alias and
  rolls it to a long Union[.....] thing.
* [docs] documentation update

Release 0.6.9
-------------

- [sao] always calculating best outcome in AspirationNegotiator
- [utilities] making the calculation of utility ranges in minmax more robust
- [sao] Making SyncController default to the outcome with maximum utility in the first round instead of sending no response.
- [chain] moved to relative imports
- [negotiators] Removed the outcomes/reserved_value parameters when constructing RandomNegotiator
- [negotiators] Improvements to the implementation of Controller
- [sao] Adding SAOAspirationSingleAgreementController, SAOMetaController, SAORandomSyncController and improving the implementation of SAOSyncController and SAOSingleAgreementController
- adding more tests

Release 0.6.8
-------------

- [situated] Improving the description of partners and handling in
  request/run negotiations by having the caller being added to the
  partners list automatically if it has one item.
- adding a helper to find shortest_unique_names.
- Better adherence to the black format
- Documentation Update
- Separating configuration into config.py
- Moving CI to Github Actions
- Removing negotiation_info.csv and keeping only negotiations.csv
  Now negotiation.csv contains all the information about the negotiation
  that was scattered between it an negotiation_info.csv
- [situated] Adding the concept of a neg. group
- [bugfix] correcting the implementation of joining in
  SAOControlledNegotiator
- [negotiators] Making it possible to use the `AspirationMixin`
  for controllers.

Release 0.6.7
-------------

- Adding information about the agent in SAOState
- Preliminary GUI support
- Correcting the import of json_normalize to match
- Pandas 1.0
- Correcting the types of offers in SingleAgreement
- Documentation update (removing inherited members)

Release 0.6.6
-------------

- [tournament] Adding a string conversion to TournamentResults
- [sao] Adding SAOSingleAgreementController that is guaranteed to get
  at most one agreement only.
- [helperrs] Supporting dumping csv files in dump/load
- [situated] making _type_name add the module name to the class name
  before snake-casing it
- [situated] [bug] correcting cancellation_fraction implementation to
  take into account non-negotiated contracts

Release 0.6.5
-------------

- [helpers] making add_records more robust to input
- [bugfix] Resolving a bug in creating graphs while running a tournament

Release 0.6.4
-------------

- [situated] Cancellation fraction and Agreement fraction now consider only
  negotiated contracts

Release 0.6.3
-------------

- [situated] never fail for gif generation (just pass the exception)
- [CLI] Fixing a bug that prevented negmas tournament create from failing
  gracefully when not given a scorer/assigner/world-config or world-generator.

Release 0.6.2
-------------

- [mechanism] triggering a negotiator_exception even on negotiator exceptions
- [situated] adding a count of exceptions per agent
- [situated] counting exceptions in negotiations as exceptions by the owner agent
- [mechanism] adding mechanism abortion

Release 0.6.1
-------------

- [situated] Adding the method call to World and using it always
  when calling agents to count exceptions
- [situated] Adding n_*_exceptions to count exceptions happening in
  agents, simulation and negotiations
- [tournaments] Adding n_*_exceptions to the tournament Results
  structure (TournamentResults) reporting the number of exceptions
  that happened during the tournament from different types
- [tournament] adding more details to tournament results and andding world_stats.csv to the saved data
- [situated] handling compact world running better:
  - added a no_logs option to World that disables all logging including agent logging
  - Corrected the tournament running functions to deal correctly with worlds with no logs
- [tournament] adding path to tournament results

Release 0.6.0
-------------

- [situated] adding negotiation quotas and setting negotiator owner
- [base] adding accessor to negotiator's nmi and a setter for the owner
- [sao] removing deadlocks in SAOSyncController
- [tournament] allowing round-robin tournaments to have zero stage winners (which will resolve to one winner)
- [tournament] making median the default metric
- [base] on_negotiation_end is always sent to negotiators
- [base] Adding owner to negotiators to keep track of the agent owning a negotiator.
- [situated] Resolving a possible bug if the victims of a breach were more than one agent

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
- [core] Adding minmax and outcome_with_utility as members of UtilityFuction. Global functions of the same name are still there for backward compatibility
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
  LinearAdditiveUtilityFunction.
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
- [Core] Adding Negotiator.ufun as an alias to Negotiator.ufun
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
- separating ControlledNegotiator, ControlledSAONegotiator. This speeds up all simulations at the expense
  of backward incompatibility for the undocumented Controller pattern. If you are using this pattern, you
  need to create ControlledSAONegotiator instead of SAONegotiator. If you are not using Controller or you do not know
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
  negotiator.ufun = x) AFTER the negotiation starts. It is
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
