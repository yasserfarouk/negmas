import functools
import itertools
import math
from abc import ABC
from random import random, randint
from typing import TYPE_CHECKING

from dataclasses import dataclass
from numpy.random import dirichlet

from negmas.events import Notification
from negmas.helpers import ConfigReader, get_class
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.sao import AspirationNegotiator
from negmas.situated import Contract, Breach
from negmas.situated import RenegotiationRequest
from negmas.utilities import normalize, ComplexWeightedUtilityFunction, MappingUtilityFunction
from .common import SCMLAgent, CFP
from .helpers import pos_gauss

if True: #
    from typing import Dict, Any, List, Optional, Union, Tuple
    from .common import Loan

if TYPE_CHECKING:
    from .awi import SCMLAWI

__all__ = [
    'Consumer',
    'ConsumptionProfile',
    'ScheduleDrivenConsumer'
]


@dataclass
class ConsumptionProfile:
    schedule: Union[int, List[int]] = 0
    underconsumption: float = 0.1
    overconsumption: float = 0.01
    dynamicity: float = 0.0
    cv: float = 0.1

    alpha_q: float = 0.5
    alpha_u: float = 1.0

    beta_q: float = 10.0
    beta_u: float = 10.0

    tau_q: float = 2
    tau_u: float = 0.25

    @classmethod
    def random(cls):
        return ConsumptionProfile(schedule=randint(1, 20), overconsumption=2 * random(), underconsumption=2 * random()
                                  , cv=random(), beta_q=99 * random() + 1, beta_u=99 * random() + 1, tau_q=3 * random()
                                  , tau_u=random())

    def schedule_at(self, time: int) -> int:
        if isinstance(self.schedule, int):
            return self.schedule
        else:
            return self.schedule[time % len(self.schedule)]

    def schedule_within(self, time: Union[int, List[int], Tuple[int, int]]) -> int:
        if isinstance(time, int):
            return self.schedule_at(time)
        if isinstance(time, tuple):
            times = list(range(time[0], time[1] + 1))
        else:
            times = time
        if isinstance(self.schedule, int):
            return self.schedule * len(times)
        return sum(self.schedule_at(t) for t in times)

    def set_schedule_at(self, time: int, value: int, n_steps: int) -> None:
        if isinstance(self.schedule, int):
            self.schedule = [self.schedule] * n_steps
        elif len(self.schedule) < n_steps:
            self.schedule = list(itertools.chain(*([self.schedule] * int(math.ceil(n_steps / len(self.schedule))))))
        self.schedule[time % len(self.schedule)] = value


class Consumer(SCMLAgent, ABC):
    """Base class of all consumer classes"""
    pass


class ScheduleDrivenConsumer(Consumer):
    """Consumer class"""

    def on_contract_nullified(self, contract: Contract, bankrupt_partner: str, compensation: float) -> None:
        pass

    def on_agent_bankrupt(self, agent_id: str) -> None:
        pass

    def confirm_partial_execution(self, contract: Contract, breaches: List[Breach]) -> bool:
        pass

    def on_remove_cfp(self, cfp: 'CFP'):
        pass

    MAX_UNIT_PRICE = 1e5
    RELATIVE_MAX_PRICE = 2

    def __init__(self, profiles: Dict[int, ConsumptionProfile] = None
                 , negotiator_type='negmas.sao.AspirationNegotiator'
                 , consumption_horizon: Optional[int] = 20
                 , immediate_cfp_update: bool = True
                 , name=None):
        super().__init__(name=name)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.profiles: Dict[int, ConsumptionProfile] = dict()
        self.secured_quantities: Dict[int, int] = dict()
        if profiles is not None:
            self.set_profiles(profiles=profiles)
        self.consumption_horizon = consumption_horizon
        self.immediate_cfp_update = immediate_cfp_update

    def on_new_cfp(self, cfp: 'CFP') -> None:
        pass  # consumers never respond to CFPs

    def init(self):
        super().init()
        if self.consumption_horizon is None:
            self.consumption_horizon = self.awi.n_steps
        self.awi.register_interest(list(self.profiles.keys()))

    def set_profiles(self, profiles: Dict[int, ConsumptionProfile]):
        self.profiles = profiles if profiles is not None else dict()
        self.secured_quantities = dict(zip(profiles.keys(), itertools.repeat(0)))

    def register_product_cfps(self, p: int, t: int, profile: ConsumptionProfile):
        current_schedule = profile.schedule_at(t)
        product = self.products[p]
        awi: SCMLAWI = self.awi
        if current_schedule <= 0:
            awi.bb_remove(section='cfps', query={'publisher': self.id, 'time': t, 'product_index': p})
            return
        max_price = ScheduleDrivenConsumer.RELATIVE_MAX_PRICE * product.catalog_price if product.catalog_price is not None \
            else ScheduleDrivenConsumer.MAX_UNIT_PRICE
        cfps = awi.bb_query(section='cfps', query={'publisher': self.id, 'time': t, 'product': p})
        if cfps is not None and len(cfps) > 0:
            for _, cfp in cfps.items():
                if cfp.max_quantity != current_schedule:
                    cfp = CFP(is_buy=True, publisher=self.id, product=p
                              , time=t, unit_price=(0, max_price), quantity=(1, current_schedule))
                    awi.bb_remove(section='cfps', query={'publisher': self.id, 'time': t
                        , 'product': p})
                    awi.register_cfp(cfp)
                    break
        else:
            cfp = CFP(is_buy=True, publisher=self.id, product=p
                      , time=t, unit_price=(0, max_price), quantity=(1, current_schedule))
            awi.register_cfp(cfp)

    def step(self):
        super().step()
        if self.consumption_horizon is None:
            horizon = self.awi.n_steps
        else:
            horizon = min(self.awi.current_step + self.consumption_horizon + 1, self.awi.n_steps)
        for p, profile in self.profiles.items():
            for t in range(self.awi.current_step, horizon):  # + self.transportation_delay
                self.register_product_cfps(p=p, t=t, profile=profile)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    @staticmethod
    def _qufun(outcome: Dict[str, Any], tau: float, profile: ConsumptionProfile):
        """The ufun value for quantity"""
        q, t = outcome['quantity'], outcome['time']
        y = profile.schedule_within(t)
        o = profile.overconsumption
        u = profile.underconsumption
        if q == 0 and y != 0:
            return 0.0
        if y == 0:
            result = -o
        elif q > y:
            result = - o * (((q - y) / y) ** tau)
        elif q < y:
            result = - u * (((y - q) / y) ** tau)
        else:
            result = 1.0
        return math.exp(result)

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[Negotiator]:
        profile = self.profiles[cfp.product]
        if profile.cv == 0:
            alpha_u, alpha_q = profile.alpha_u, profile.alpha_q
        else:
            alpha_u, alpha_q = tuple(dirichlet((profile.alpha_u, profile.alpha_q), size=1)[0])
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)
        ufun = normalize(ComplexWeightedUtilityFunction(ufuns=[
            MappingUtilityFunction(mapping=lambda x: 1 - x['unit_price'] ** tau_u / beta_u)
            , MappingUtilityFunction(mapping=functools.partial(ScheduleDrivenConsumer._qufun
                                                               , tau=tau_q, profile=profile))]
            , weights=[alpha_u, alpha_q], name=self.name + '_' + partner)
            , outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        if self.negotiator_type == AspirationNegotiator:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + partner
                                              , aspiration_type='boulware')
        else:
            negotiator = self.negotiator_type(name=self.name + '*' + partner)
        negotiator.name = self.name + '_' + partner
        negotiator.utility_function = ufun
        return negotiator

    def set_renegotiation_agenda(self, contract: Contract, breaches: List[Breach]) -> Optional[RenegotiationRequest]:
        """
        Received by partners in ascending order of their total breach levels in order to set the
        renegotiation agenda when contract execution fails

        Args:
            contract:
            breaches:

        Returns:

        """
        return None

    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Breach]
                                         , agenda: RenegotiationRequest) -> Optional[Negotiator]:
        """
        Called to respond to a renegotiation request

        Args:
            agenda:
            contract:
            breaches:

        Returns:

        """
        return None

    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return False

    def sign_contract(self, contract: Contract) -> Optional[str]:
        if contract is None:
            return
        cfp: CFP = contract.annotation['cfp']
        agreement = contract.agreement  # type: ignore
        schedule = self.profiles[cfp.product].schedule_at(agreement['time'])
        if schedule - agreement['quantity'] < 0:
            return None
        return super().sign_contract(contract=contract)

    def on_contract_signed(self, contract: Contract):
        super().on_contract_signed(contract)
        if contract is None:
            return
        cfp: CFP = contract.annotation['cfp']
        agreement = contract.agreement  # type: ignore
        self.secured_quantities[cfp.product] += agreement['quantity']
        old_quantity = self.profiles[cfp.product].schedule_at(agreement['time'])
        new_quantity = old_quantity - agreement['quantity']
        t = agreement['time']
        self.profiles[cfp.product].set_schedule_at(time=t
                                                   , value=new_quantity
                                                   , n_steps=self.awi.n_steps)
        if self.immediate_cfp_update and new_quantity != old_quantity:
            self.register_product_cfps(p=cfp.product, t=t, profile=self.profiles[cfp.product])
        for negotiation in self._running_negotiations.values():
            self.notify(negotiation.negotiator, Notification(type='ufun_modified', data=None))



