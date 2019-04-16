History
=======

Release 0.2.5
-------------

- improvements to ufun representation to speedup computation
- making default factory managers slightly less risky in their behavior in long simulations and more risky in short ones
- adding jnegmas-setup and genius-setup commands to download and install jenegmas and genius bridge
- removing the logger mixin and replaced it with parameters to World and SCMLWorld
- added compact parameter to SCMLWorld, tournament, and world generators to reduce the memory footprint
- added --compact/--debug to the command line tools to avoid memory and log explosion setting the default to --compact
- improving implementation of consumer ufun for cases with negative schedule
- changing the return type of SCMLAWI.state from Factory to FactoryState to avoid modifying the original factory. For efficiency reasons, the profiles list is passed as it is and it is possible to modify it but that is forbidden by the laws of the game.
- Speeding up and correcting financial report reception.
- Making bankruptcy reporting system-wide
- avoiding execution of contracts with negative or no quantity and logging ones with zero unit price.
- documentation update
- bug fix to resolve an issue with ufun calculation for consumers in case of over consumption.
- make the default behavior of negmas command to reveal agent types in their names
- preventing agents from publishing CFPs with the ID of other agents
- documentation update
- improved Java support
- added option default_dump_extension to ~/negmas/config.json to enable changing the format of dumps from json to yaml. Currently json is the default. This included adding a helper function helpers.dump() to dump in the selected format (or overriding it by providing a file extension).
- completing compatibility with SCML description (minor change to the consumer profile)
- added two new options to negmas tournament command: anac2019std and anac2019collusion to simulate these two tracks of the ANAC 2019 SCML. Sabotage version will be added later.
- added two new functions in apps.scml.utils anac2019_std, anac2019_collusion to simulate these two tracks of the ANAC 2019 SCML. Sabotage version will be added later.
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
