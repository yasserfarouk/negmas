History
=======

Release 0.2.19
--------------

- logging CFPs
- defaulting to buying insurance in negmas scml
- bug resolution related to recently added ability to use LinearUtilityFunction created by a dict with tuple outcomes
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

- Adding new callbacks to simplify factory manager development in the SCM world
  - on_contract_executed, on_contract_breached
  - on_inventory_change, on_production_success, on_cash_transfer
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
   1. adding --factories argument to control how many factories (at least) should exist on each production level
   2. adding --agents argument to control how many agents per competitor to instantiate. For the anac2019std ttype,
      this will be forced to 1
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
- making default factory managers slightly less risky in their behavior in long simulations and more risky in short ones
- adding jnegmas-setup and genius-setup commands to download and install jenegmas and genius bridge
- removing the logger mixin and replaced it with parameters to World and SCMLWorld
- added compact parameter to SCMLWorld, tournament, and world generators to reduce the memory footprint
- added --compact/--debug to the command line tools to avoid memory and log explosion setting the default to --compact
- improving implementation of consumer ufun for cases with negative schedule
- changing the return type of SCMLAWI.state from Factory to FactoryState to avoid modifying the original factory. For
  efficiency reasons, the profiles list is passed as it is and it is possible to modify it but that is forbidden by the
  rules of the game.
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
