from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Collection

from negmas.serialization import to_flat_dict

from .world import World

if TYPE_CHECKING:
    from .breaches import Breach
    from .contract import Contract

__all__ = ["SimpleWorld"]


class SimpleWorld(World, ABC):
    """
    Represents a simple world simulation with sane values for most callbacks and methods.
    """

    def delete_executed_contracts(self) -> None:
        pass

    def post_step_stats(self):
        pass

    def pre_step_stats(self):
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        return contracts

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        return to_flat_dict(contract, deep=True)

    def breach_record(self, breach: Breach) -> dict[str, Any]:
        return to_flat_dict(breach, deep=True)

    def contract_size(self, contract: Contract) -> float:
        return 1.0
