from negmas.situated import AgentWorldInterface, Contract
from .common import *
from typing import Optional


__all__ = [
    'SCMLAWI',
]


class SCMLAWI(AgentWorldInterface):
    def register_cfp(self, cfp: CFP) -> None:
        """Registers a CFP"""
        self._world.n_new_cfps += 1
        cfp.money_resolution = self._world.money_resolution
        self._world.bulletin_board.record(section='cfps', key=cfp.id, value=cfp)

    def remove_cfp(self, cfp: CFP) -> bool:
        """Removes a CFP"""
        if self.agent.id != cfp.publisher:
            return False
        return self._world.bulletin_board.remove(section='cfps', key=str(hash(cfp)))

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

    @property
    def products(self):
        """Products in the world"""
        return self._world.products

    @property
    def processes(self):
        """Processes in the world"""
        return self._world.processes

