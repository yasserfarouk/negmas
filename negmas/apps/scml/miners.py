from abc import ABC
from collections import defaultdict
from random import random
from typing import TYPE_CHECKING

from dataclasses import dataclass
from numpy.random import dirichlet

from negmas.common import MechanismState, MechanismInfo
from negmas.helpers import ConfigReader, get_class
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue
from negmas.sao import AspirationNegotiator
from negmas.situated import Contract, Breach
from negmas.situated import RenegotiationRequest
from negmas.utilities import LinearUtilityAggregationFunction, normalize
from .common import SCMLAgent
from .helpers import pos_gauss

if True:
    from typing import Dict, Any, List, Optional
    from .common import Loan

__all__ = [
    'Miner', 'MiningProfile', 'ReactiveMiner'
]


@dataclass
class MiningProfile:
    cv: float = 0.05

    alpha_t: float = 1.0
    alpha_q: float = 1.0
    alpha_u: float = 1.0

    beta_t: float = 1.0
    beta_q: float = 100.0
    beta_u: float = 100.0

    tau_t: float = -0.25
    tau_q: float = 0.25
    tau_u: float = 1.0

    @classmethod
    def random(cls):
        alpha_t, alpha_q, alpha_u = dirichlet((1, 1, 1), size=1)[0]
        tau_t, tau_q, tau_u = 2 * random() - 1, 2 * random() - 1, 2 * random() - 1
        return MiningProfile(cv=random(), alpha_t=alpha_t, alpha_q=alpha_q, alpha_u=alpha_u
                             , tau_t=tau_t, tau_q=tau_q, tau_u=tau_u
                             , beta_t=1.5 * random(), beta_q=99 * random() + 1, beta_u=99 * random() + 1)


class Miner(SCMLAgent, ABC):
    """Base class of all miners"""


class ReactiveMiner(Miner):
    """Raw Material Generator"""

    def on_contract_nullified(self, contract: Contract, bankrupt_partner: str, compensation: float) -> None:
        pass

    def on_agent_bankrupt(self, agent_id: str) -> None:
        pass

    def confirm_partial_execution(self, contract: Contract, breaches: List[Breach]) -> bool:
        pass

    def on_remove_cfp(self, cfp: 'CFP'):
        pass

    def __init__(self, profiles: Dict[int, MiningProfile] = None, negotiator_type='negmas.sao.AspirationNegotiator'
                 , n_retrials=0, reactive=True, name=None):
        super().__init__(name=name)
        self.negotiator_type = get_class(negotiator_type, scope=globals())
        self.profiles: Dict[int, MiningProfile] = {}
        self.n_neg_trials: Dict[str, int] = defaultdict(int)
        self.n_retrials = n_retrials
        self.reactive = reactive
        if profiles is not None:
            self.set_profiles(profiles=profiles)

    def init(self):
        super().init()
        self.awi.register_interest(list(self.profiles.keys()))

    def on_negotiation_failure(self, partners: List[str], annotation: Dict[str, Any], mechanism: MechanismInfo
                               , state: MechanismState) -> None:
        # noinspection PyUnusedLocal
        cfp = annotation['cfp']
        super().on_negotiation_failure(partners=partners, annotation=annotation, mechanism=mechanism, state=state)
        thiscfp = self.awi.bb_query(section='cfps', query=cfp.id, query_keys=True)
        if cfp.publisher != self.id and thiscfp is not None and len(thiscfp) > 0 \
            and self.n_neg_trials[cfp.id] < self.n_retrials:
            self.awi.logdebug(f'Renegotiating {self.n_neg_trials[cfp.id]} on {cfp}')
            self.on_new_cfp(cfp=annotation['cfp'])

    def set_profiles(self, profiles: Dict[int, MiningProfile]):
        self.profiles = profiles if profiles is not None else dict()

    def _process_cfp(self, cfp: 'CFP'):
        if not self.can_expect_agreement(cfp=cfp, margin=0):
            return
        profile = self.profiles.get(cfp.product, None)
        if profile is None:
            return
        if profile.cv == 0:
            alpha_u, alpha_q, alpha_t = profile.alpha_u, profile.alpha_q, profile.alpha_t
        else:
            alpha_u, alpha_q, alpha_t = tuple(
                dirichlet((profile.alpha_u, profile.alpha_q, profile.alpha_t), size=1)[0])
        beta_u = pos_gauss(profile.beta_u, profile.cv)
        beta_t = pos_gauss(profile.beta_t, profile.cv)
        beta_q = pos_gauss(profile.beta_q, profile.cv)

        tau_u = pos_gauss(profile.tau_u, profile.cv)
        tau_t = pos_gauss(profile.tau_t, profile.cv)
        tau_q = pos_gauss(profile.tau_q, profile.cv)

        ufun = normalize(LinearUtilityAggregationFunction(issue_utilities={
            'time': lambda x: x ** tau_t / beta_t,
            'quantity': lambda x: x ** tau_q / beta_q,
            'unit_price': lambda x: x ** tau_u / beta_u if x > 1e-7 else -2000.0,
        }, weights={'time': alpha_t, 'quantity': alpha_q, 'unit_price': alpha_u})
            , outcomes=cfp.outcomes, infeasible_cutoff=-1500)
        if self.negotiator_type == AspirationNegotiator:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + cfp.publisher
                                              , dynamic_ufun=False, aspiration_type='boulware')
        else:
            negotiator = self.negotiator_type(assume_normalized=True, name=self.name + '*' + cfp.publisher)
        negotiator.utility_function = normalize(ufun, outcomes=cfp.outcomes, infeasible_cutoff=None)
        self.n_neg_trials[cfp.id] += 1
        self.request_negotiation(partners=[cfp.publisher, self.id], issues=cfp.issues, negotiator=negotiator
                                 , annotation=self._create_annotation(cfp=cfp), extra=None
                                 , mechanism_name='negmas.sao.SAOMechanism', roles=None)

    def on_new_cfp(self, cfp: 'CFP'):
        if self.reactive:
            if not cfp.satisfies(query={'products': list(self.profiles.keys())}):
                return
            self._process_cfp(cfp)

    def step(self):
        super().step()
        if not self.reactive:
            cfps = self.awi.bb_query(section='cfps', query_keys=False
                                                 , query={'products': list(self.profiles.keys())})
            if cfps is None:
                return
            cfps = cfps.values()
            for cfp in cfps:
                self._process_cfp(cfp)

    def confirm_contract_execution(self, contract: Contract) -> bool:
        return True

    def on_negotiation_request(self, cfp: "CFP", partner: str) -> Optional[Negotiator]:
        raise ValueError('Miners should never receive negotiation requests as they publish no CFPs')

    def set_renegotiation_agenda(self, contract: Contract, breaches: List[Breach]) -> Optional[RenegotiationRequest]:
        return None

    def respond_to_renegotiation_request(self, contract: Contract, breaches: List[Breach]
                                         , agenda: RenegotiationRequest) -> Optional[Negotiator]:
        return None

    def confirm_loan(self, loan: Loan) -> bool:
        """called by the world manager to confirm a loan if needed by the buyer of a contract that is about to be
        breached"""
        return False
