"""
Implements an agent-world-interface (see `AgentWorldInterface`) for the SCM world.
"""
from negmas import Issue
from negmas.situated import AgentWorldInterface, Contract
from negmas.apps.scml.common import *
from typing import Optional, List, Dict, Any

__all__ = [
    'SCMLAWI',
]


class SCMLAWI(AgentWorldInterface):
    """A single contact point between SCML agents and the world simulation."""

    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        self.bb_record(section='cfps', key=cfp.id, value=cfp)

    def register_interest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self._world.register_interest(agent=self.agent, products=products)

    def unregister_interest(self, products: List[int]) -> None:
        """registers interest in receiving callbacks about CFPs related to these products"""
        self._world.unregister_interest(agent=self.agent, products=products)

    def remove_cfp(self, cfp: CFP) -> bool:
        """Removes a CFP"""
        if self.agent.id != cfp.publisher:
            return False
        return self.bb_remove(section='cfps', key=str(hash(cfp)))

    def evaluate_insurance(self, contract: Contract, t: int = None) -> Optional[float]:
        """Can be called to evaluate the premium for insuring the given contract against breachs committed by others

        Args:

            contract: hypothetical contract
            t: time at which the policy is to be bought. If None, it means current step
        """
        return self._world.evaluate_insurance(contract=contract, agent=self.agent, t=t)

    def buy_insurance(self, contract: Contract) -> bool:
        """Buys insurance for the contract by the premium calculated by the insurance company.

        Remarks:
            The agent can call `evaluate_insurance` to find the premium that will be used.
        """
        return self._world.buy_insurance(contract=contract, agent=self.agent)

    def _create_annotation(self, cfp: 'CFP'):
        """Creates full annotation based on a cfp that the agent is receiving"""
        partners = [self.agent.id, cfp.publisher]
        annotation = {'cfp': cfp, 'partners': partners}
        if cfp.is_buy:
            annotation['seller'] = self.agent.id
            annotation['buyer'] = cfp.publisher
        else:
            annotation['buyer'] = self.agent.id
            annotation['seller'] = cfp.publisher
        return annotation

    def request_negotiation(self, cfp: CFP, req_id: str, roles: List[str] = None, mechanism_name: str = None
                            , mechanism_params: Dict[str, Any] = None):
        default_annotation = self._create_annotation(cfp)
        super().request_negotiation_about(issues=cfp.issues, req_id=req_id, partners=default_annotation['partners']
                                          , roles=roles, annotation=default_annotation, mechanism_name=mechanism_name
                                          , mechanism_params=mechanism_params)

    def request_negotiation_about(self
                                  , issues: List[Issue]
                                  , partners: List[str]
                                  , req_id: str
                                  , roles: List[str] = None
                                  , annotation: Optional[Dict[str, Any]] = None
                                  , mechanism_name: str = None
                                  , mechanism_params: Dict[str, Any] = None
                                  ):
        raise RuntimeError('request_negotiation_about should never be called directly in the SCM world'
                           ', call request_negotiation instead.')

    @property
    def products(self) -> List[Product]:
        """Products in the world"""
        return self._world.products

    @property
    def processes(self) -> List[Process]:
        """Processes in the world"""
        return self._world.processes


