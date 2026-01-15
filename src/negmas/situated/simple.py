"""Simplified World implementation with common defaults."""

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
    Represents a simple world with no simulation and sane values for most callbacks and methods.
    """

    def delete_executed_contracts(self) -> None:
        """Delete executed contracts."""
        pass

    def post_step_stats(self):
        """Post step stats."""
        pass

    def pre_step_stats(self):
        """Pre step stats."""
        pass

    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        """Order contracts for execution.

        Args:
            contracts: Contracts.

        Returns:
            Collection[Contract]: The result.
        """
        return contracts

    def contract_record(self, contract: Contract) -> dict[str, Any]:
        """Contract record.

        Args:
            contract: Contract.

        Returns:
            dict[str, Any]: The result.
        """
        return to_flat_dict(contract, deep=True)

    def breach_record(self, breach: Breach) -> dict[str, Any]:
        """Breach record.

        Args:
            breach: Breach.

        Returns:
            dict[str, Any]: The result.
        """
        return to_flat_dict(breach, deep=True)

    def contract_size(self, contract: Contract) -> float:
        """Contract size.

        Args:
            contract: Contract.

        Returns:
            float: The result.
        """
        return 1.0
