"""Mixins providing optional behaviors for World and Agent implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Collection, Any


if TYPE_CHECKING:
    from negmas.common import MechanismState, NegotiatorMechanismInterface
    from negmas.negotiators import Negotiator

    from .breaches import Breach
    from .common import RenegotiationRequest
    from .contract import Contract
    from .world import World

__all__ = ["TimeInAgreementMixin", "NoContractExecutionMixin", "NoResponsesMixin"]


class TimeInAgreementMixin:
    """Mixin that tracks contracts by their execution time step for efficient retrieval."""

    def init(self, time_field="time"):
        """Initialize contract tracking by time step.

        Args:
            time_field: Name of the field in contract agreements that specifies execution time.
        """
        self._time_field_name = time_field
        self.contracts_per_step: dict[int, list[Contract]] = defaultdict(list)

    def on_contract_signed(self: World, contract: Contract):
        """Register newly signed contract for execution at its scheduled time step.

        Args:
            contract: The contract that was just signed.
        """
        result = super().on_contract_signed(contract=contract)
        if result:
            self.contracts_per_step[contract.agreement[self._time_field_name]].append(
                contract
            )
        return result

    def executable_contracts(self: World) -> Collection[Contract]:
        """Called at every time-step to get the contracts that are `executable` at this point of the simulation"""
        if {
            _["id"]
            for _ in self._saved_contracts.values()
            if _["delivery_time"] == self.current_step and _["signed_at"] >= 0
        } != {_.id for _ in self.contracts_per_step.get(self.current_step, [])}:
            saved = {
                _["id"]
                for _ in self._saved_contracts.values()
                if _["delivery_time"] == self.current_step and _["signed_at"] >= 0
            }
            used = {_.id for _ in self.contracts_per_step.get(self.current_step, [])}
            err = (
                f"Some signed contracts due at {self.current_step} are not being executed: {saved - used} "
                f"({used - saved}):\n"
            )
            for c in saved - used:
                err += f"Saved Only:{str(self._saved_contracts[c])}\n"
            for c in used - saved:
                con = None
                for _ in self.contracts_per_step.get(self.current_step, []):
                    if _.id == c:
                        con = _
                        break
                err += f"Executable Only:{con}\n"
            raise ValueError(err)
        return self.contracts_per_step.get(self.current_step, [])

    def delete_executed_contracts(self: World) -> None:
        """Delete executed contracts."""
        self.contracts_per_step.pop(self.current_step, None)

    def get_dropped_contracts(self) -> Collection[Contract]:
        """Return contracts scheduled for this step that were signed but not executed, breached, nullified, or errored."""
        return [
            _
            for _ in self.contracts_per_step.get(self.current_step, [])
            if self._saved_contracts[_.id]["signed_at"] >= 0
            and self._saved_contracts[_.id].get("breaches", "") == ""
            and self._saved_contracts[_.id].get("nullified_at", -1) < 0
            and self._saved_contracts[_.id].get("erred_at", -1) < 0
            and self._saved_contracts[_.id].get("executed_at", -1) < 0
        ]


class NoContractExecutionMixin:
    """
    A mixin to add when there is no contract execution
    """

    def delete_executed_contracts(self: World) -> None:
        """Delete executed contracts."""
        pass

    def executable_contracts(self) -> Collection[Contract]:
        """Return empty collection since this mixin disables contract execution."""
        return []

    def start_contract_execution(self, contract: Contract) -> set[Breach]:
        """No-op since this mixin disables contract execution."""
        return set()

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        """No-op since this mixin disables contract execution."""
        pass


class NoResponsesMixin:
    """A mixin that can be added to Agent to minimize the number of abstract methods"""

    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        """Called when a negotiation request is rejected.

        Args:
            req_id: Unique identifier of the rejected request.
            by: List of agent IDs that rejected the request, or None if unknown.
        """
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        """Called when a negotiation request is accepted and the mechanism starts.

        Args:
            req_id: Unique identifier of the accepted request.
            mechanism: Interface to the negotiation mechanism that was created.
        """
        pass

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """Called when a negotiation ends without reaching an agreement.

        Args:
            partners: List of agent IDs that participated in the negotiation.
            annotation: Metadata associated with the negotiation.
            mechanism: Interface to the negotiation mechanism.
            state: Final state of the mechanism when it ended.
        """
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """Called when a negotiation ends with a successful agreement.

        Args:
            contract: The resulting contract from the negotiation.
            mechanism: Interface to the negotiation mechanism.
        """
        pass

    def on_contract_signed(self, contract: Contract) -> bool:
        """Called when all parties have signed a contract.

        Args:
            contract: The fully signed contract.

        Returns:
            True to accept the contract, False to reject it.
        """
        return True

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        """Called when a contract is cancelled due to rejections.

        Args:
            contract: The cancelled contract.
            rejectors: List of agent IDs that rejected the contract.
        """
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        """Create a renegotiation request when a contract is breached.

        Args:
            contract: The breached contract.
            breaches: List of breaches that occurred.

        Returns:
            A renegotiation request if renegotiation is desired, None otherwise.
        """
        pass

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        """Decide whether to participate in renegotiation of a breached contract.

        Args:
            contract: The breached contract being renegotiated.
            breaches: List of breaches that triggered renegotiation.
            agenda: The proposed renegotiation agenda and terms.

        Returns:
            A negotiator to participate in renegotiation, None to refuse.
        """
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        """Called when a contract has been successfully executed.

        Args:
            contract: The executed contract.
        """
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        """Called when a contract has been breached.

        Args:
            contract: The breached contract.
            breaches: List of breaches that occurred.
            resolution: New contract from renegotiation if any, None otherwise.
        """
        pass
