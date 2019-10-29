"""
The implementation file for all entities needed for ANAC-SCML 2019.

Participants need to provide a class inherited from `FactoryManager` that implements all of its abstract functions.

Participants can optionally override any other methods of this class or implement new `NegotiatorUtility` class.

Simulation steps:
-----------------

    #. prepare custom stats (call `_pre_step_stats`)
    #. sign contracts that are to be signed at this step calling `on_contract_signed` as needed
    #. step all existing negotiations `negotiation_speed_multiple` times handling any failed negotiations and creating
       contracts for any resulting agreements
    #. run all `Entity` objects registered (i.e. all agents). `Consumer` s run first then `FactoryManager` s then
       `Miner` s
    #. execute contracts that are executable at this time-step handling any breaches
    #. Custom Simulation Steps:
        #. step all factories (`Factory` objects) running any pre-scheduled commands
        #. Apply interests and pay loans
        #. remove expired `CFP` s
        #. Deliver any products that are in transportation
    #. remove any negotiations that are completed!
    #. update basic stats
    #. update custom stats (call `_post_step_stats`)

Remarks about re-negotiation on breaches:
-----------------------------------------

    - The victim is asked first to specify the negotiation agenda (issues) then the perpetrator
    - renegotiations for breaches run immediately to completion independent from settings of
      `negotiation_speed_multiplier` and `immediate_negotiations`. That include conclusion and signing of any resulting
      agreements.

Remarks about timing:
---------------------

    - The order of events within a single time-step are as follows:

        #. Contracts scheduled to be signed are signed.
        #. Scheduled negotiations run for the predefined number of steps. Any negotiation that result in a contract or
           fail may trigger other negotiations.
        #. If `immediate_negotiations`, some of the newly added negotiations may be concluded/failed.
        #. Any newly concluded contracts that are to be signed at this step are signed
        #. Contracts are executed including full execution of any re-negotiations and breaches are handled. Notice that
           if re-negotiation leads to new contracts, these will be concluded and signed immediately at this step. Please
           note the following about contract execution:

           - Products are moved from the seller's storage to a temporary *truck* as long as they are available at the
             time of contract execution. Because contract execution happens *before* actual production, outputs from
             production processes *CANNOT* be sold at the same time-step.

        #. Production is executed on all factories. For a `Process` to start/continue on a `Line`, all its inputs
           required at this time-step **MUST** be available in storage of the corresponding factory *by this point*.
           This implies that it is impossible for any processes to start at time-step *0* except if initial storage was
           nonzero. `FactoryManager` s are informed about processes that cannot start due to storage or fund shortage
           (or cannot continue due to storage shortage) through an `on_production_failure` call.
        #. Outputs of the `Process` are generated at *the end* of the corresponding time-step. It is immediately moved
           to storage. Because outputs are generated at the *end* of the step and inputs are consumed at the beginning,
           a factory cannot use outputs of a process as inputs to another process that starts at the same time-step.
        #. Products are moved from the temporary *truck* to the buyer's storage after the `transportation_delay` have
           passed at the *end* of the time-step. Transportation completes at the *end* of the time-step no matter
           what is the value for `transportation_delay`. This means that if a `FactoryManager` believes
           that it can produce some product at time *t*, it should never contract to sell it before *t+d + 1* where
           *d* is the `transportation_delay` (the *1* comes from the fact that contract execution happens *before*
           production). Even for a zero transportation delay, you cannot produce something and sell it in the same
           time-step. Moreover, the buyer should never use the product to be delivered at time *t* as an input to a
           production process that needs it before step *t+1*.
        #. When contracts are executed, the funds are deducted from the buyer's wallet at the *beginning* of the
           simulation step and deposited in the seller's wallet at the *end* of that step (similar to what happens to
           the products). This means that a factory manager cannot use funds it receives from sales at time *t* for
           buying products before *t + 1*.


Remarks about ANAC 2019 SCML League:
------------------------------------

    Given the information above, and settings for the ANAC 2019 SCML you can confirm for yourself that the following
    rules are all correct:

        #. No agents except miners should contract on delivery at time *0*.
        #. `FactoryManager` s should never sign contracts to sell the output of their production with delivery at *t*
           except if this production starts at step *t* and the contract is signed no later than than *t-1*.
        #. If not all inputs are available in storage, `FactoryManager` s should never sign contracts to sell the output
           of production with delivery at *t* later than *t-2* (and that is optimistic).


"""
from .common import *
from .schedulers import *
from .awi import *
from .agent import *
from .bank import *
from .insurance import *
from .world import *
from .factory_managers import *
from .consumers import *
from .miners import *
from . import utils
from . import helpers
from .utils import *
from .helpers import *
from .simulators import *


__all__ = (
    common.__all__
    + awi.__all__
    + factory_managers.__all__
    + bank.__all__
    + insurance.__all__
    + agent.__all__
    + simulators.__all__
    + utils.__all__
    + helpers.__all__
    + schedulers.__all__
    + world.__all__
    + consumers.__all__
    + miners.__all__
    + ["utils", "helpers"]
)
