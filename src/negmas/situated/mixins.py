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
    """TimeInAgreementMixin implementation."""

    def init(self, time_field="time"):
        """Init.

        Args:
            time_field: Time field.
        """
        self._time_field_name = time_field
        self.contracts_per_step: dict[int, list[Contract]] = defaultdict(list)

    def on_contract_signed(self: World, contract: Contract):
        """On contract signed.

        Args:
            contract: Contract.
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
        """Get dropped contracts.

        Returns:
            Collection[Contract]: The result.
        """
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
        """Executable contracts.

        Returns:
            Collection[Contract]: The result.
        """
        return []

    def start_contract_execution(self, contract: Contract) -> set[Breach]:
        """Start contract execution.

        Args:
            contract: Contract.

        Returns:
            set[Breach]: The result.
        """
        return set()

    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        """Complete contract execution.

        Args:
            contract: Contract.
            breaches: Breaches.
            resolution: Resolution.
        """
        pass


class NoResponsesMixin:
    """A mixin that can be added to Agent to minimize the number of abstract methods"""

    def on_neg_request_rejected(self, req_id: str, by: list[str] | None):
        """On neg request rejected.

        Args:
            req_id: Req id.
            by: By.
        """
        pass

    def on_neg_request_accepted(
        self, req_id: str, mechanism: NegotiatorMechanismInterface
    ):
        """On neg request accepted.

        Args:
            req_id: Req id.
            mechanism: Mechanism.
        """
        pass

    def on_negotiation_failure(
        self,
        partners: list[str],
        annotation: dict[str, Any],
        mechanism: NegotiatorMechanismInterface,
        state: MechanismState,
    ) -> None:
        """On negotiation failure.

        Args:
            partners: Partners.
            annotation: Annotation.
            mechanism: Mechanism.
            state: Current state.
        """
        pass

    def on_negotiation_success(
        self, contract: Contract, mechanism: NegotiatorMechanismInterface
    ) -> None:
        """On negotiation success.

        Args:
            contract: Contract.
            mechanism: Mechanism.
        """
        pass

    def on_contract_signed(self, contract: Contract) -> bool:
        """On contract signed.

        Args:
            contract: Contract.

        Returns:
            bool: The result.
        """
        return True

    def on_contract_cancelled(self, contract: Contract, rejectors: list[str]) -> None:
        """On contract cancelled.

        Args:
            contract: Contract.
            rejectors: Rejectors.
        """
        pass

    def set_renegotiation_agenda(
        self, contract: Contract, breaches: list[Breach]
    ) -> RenegotiationRequest | None:
        """Set renegotiation agenda.

        Args:
            contract: Contract.
            breaches: Breaches.

        Returns:
            RenegotiationRequest | None: The result.
        """
        pass

    def respond_to_renegotiation_request(
        self, contract: Contract, breaches: list[Breach], agenda: RenegotiationRequest
    ) -> Negotiator | None:
        """Respond to renegotiation request.

        Args:
            contract: Contract.
            breaches: Breaches.
            agenda: Agenda.

        Returns:
            Negotiator | None: The result.
        """
        pass

    def on_contract_executed(self, contract: Contract) -> None:
        """On contract executed.

        Args:
            contract: Contract.
        """
        pass

    def on_contract_breached(
        self, contract: Contract, breaches: list[Breach], resolution: Contract | None
    ) -> None:
        """On contract breached.

        Args:
            contract: Contract.
            breaches: Breaches.
            resolution: Resolution.
        """
        pass
