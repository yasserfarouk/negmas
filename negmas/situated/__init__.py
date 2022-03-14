# -*- coding: utf-8 -*-
"""
This module defines the base classes for worlds within which multiple agents engage in situated negotiations


The `Agent` class encapsulates the managing entity that creates negotiators to engage in negotiations within a world
`Simulation` in order to maximize its own total utility.

Remarks:
--------

    - When immediate_negotiations is true, negotiations start in the same step they are registered in (they may also end
      in that step) and `negotiation_speed_multiple` steps of it are conducted. That entails that requesting a
      negotiation may result in new contracts in the same time-step only of `immediate_negotiations` is true.

Simulation steps:
-----------------
    It is possible to control the order of the simulation steps differently using the `operations` parameter to the world
    constructor. this is the default order

    #. prepare custom stats (call `_pre_step_stats`)
    #. step all existing negotiations `negotiation_speed_multiple` times handling any failed negotiations and creating
       contracts for any resulting agreements
    #. Allow custom simulation (call `simulation_step` ) with stage 0
    #. run all `Entity` objects registered (i.e. all agents) in the predefined `simulation_order`.
    #. sign contracts that are to be signed at this step calling `on_contracts_finalized`  / `on_contract_signed` as
       needed
    #. execute contracts that are executable at this time-step handling any breaches
    #. allow custom simulation steps to run (call `simulation_step` ) with stage 1
    #. remove any negotiations that are completed!
    #. update basic stats
    #. update custom stats (call `_post_step_stats`)


Monitoring a simulation:
------------------------

You can monitor a running simulation using a `WorldMonitor` or `StatsMonitor` object. The former monitors events in the
world while the later monitors the statistics of the simulation.
This is a list of some of the events that can be monitored by `WorldMonitor` . World designers can add new events
either by announcing them using `announce` or as a side-effect of logging them using any of the log functions.

=================================   ===============================================================================
 Event                               Data
=================================   ===============================================================================
extra-step                           none
mechanism-creation-exception         exception: `Exception`
zero-outcomes-negotiation            caller: `Agent`, partners: `List[Agent]` , annotation: `Dict` [ `str`, `Any` ]
entity-exception                     exception: `Exception`
contract-exception                   contract: `Contract`, exception: `Exception`
agent-exception                      method: `str` , exception: `Exception`
agent-joined                         agent: `Agent`
negotiation-request                  caller: `Agent` , partners: `List` [ `Agent` ], issues: `List` [ `Issue` ]
                                     , mechanism_name: `str` , annotation: `Dict` [ `str`, `Any` ], req_id: `str`
negotiation-request-immediate        caller: `Agent` , partners: `List` [ `Agent` ], issues: `List` [ `Issue` ]
                                     , mechanism_name: `str` , annotation: `Dict` [ `str`, `Any` ]
negotiation-request-rejected         caller: `Agent` , partners: `List` [ `Agent` ] , req_id: `str`
                                     , rejectors: `List` [ `Agent` ] , annotation: `Dict` [ `str`, `Any` ]
negotiation-request-accepted         caller: `Agent` , partners: `List` [ `Agent` ] , req_id: `str`
                                     , mechanism: `Mechanism` , annotation: `Dict` [ `str`, `Any` ]
negotiation-success                  mechanism: `Mechanism` , contract: `Contract` , partners: `List` [ `Agent` ]
negotiation-failure                  mechanism: `Mechanism` , partners: `List` [ `Agent` ]
contract-executing                   contract: `Contract`
contract-nullified                   contract: `Contract`
contract-breached                    contract: `Contract`, breach: `Breach`
breach-resolved                      contract: `Contract`, breaches: `List[Breach]`, resolution
contract-executed                    contract: `Contract`
dropped-contract                     contract: `Contract`
=================================   ===============================================================================

"""
from __future__ import annotations

from .common import *
from .action import *
from .adapter import *
from .agent import *
from .entity import *
from .awi import *
from .breaches import *
from .bulletinboard import *
from .contract import *
from .helpers import *
from .mechanismfactory import *
from .mixins import *
from .monitors import *
from .save import *
from .world import *
from .simple import *
from .neg import *

__all__ = (
    common.__all__
    + action.__all__
    + adapter.__all__
    + agent.__all__
    + entity.__all__
    + awi.__all__
    + breaches.__all__
    + bulletinboard.__all__
    + contract.__all__
    + helpers.__all__
    + mechanismfactory.__all__
    + mixins.__all__
    + monitors.__all__
    + save.__all__
    + world.__all__
    + simple.__all__
    + neg.__all__
)
