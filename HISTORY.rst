History
=======

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
