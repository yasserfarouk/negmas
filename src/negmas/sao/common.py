"""
Common data-structures for supporting the Stacked Alternating Offers Protocol
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from attrs import define, field

from negmas.common import NegotiatorMechanismInterface, MechanismAction
from negmas.gb.common import GBState, ResponseType, ExtendedResponseType
from negmas.outcomes.common import ExtendedOutcome

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.sao.negotiators.base import SAONegotiator

__all__ = ["ResponseType", "SAOResponse", "SAOState", "SAONMI", "all_negotiator_types"]


@define
class SAOResponse(MechanismAction):
    """A response to an offer given by a negotiator in the alternating offers protocol.

    This class encapsulates both the response decision and an optional offer,
    along with any additional data (such as text explanations).

    Attributes:
        response: The response type (ACCEPT_OFFER, REJECT_OFFER, END_NEGOTIATION, etc.).
        outcome: The proposed outcome/offer, or None if not proposing.
        data: Optional dictionary of additional data, such as text explanations.

    Example:
        Creating a simple response::

            >>> from negmas.sao.common import SAOResponse
            >>> from negmas.gb.common import ResponseType
            >>> response = SAOResponse(ResponseType.REJECT_OFFER, (5, 10))
            >>> response.response
            <ResponseType.REJECT_OFFER: 1>

        Creating a response from extended types with text::

            >>> from negmas.outcomes.common import ExtendedOutcome
            >>> from negmas.gb.common import ExtendedResponseType
            >>> ext_response = ExtendedResponseType(
            ...     ResponseType.REJECT_OFFER,
            ...     data={"text": "Price too high"}
            ... )
            >>> ext_offer = ExtendedOutcome(
            ...     outcome=(5, 10),
            ...     data={"text": "Counter offer"}
            ... )
            >>> combined = SAOResponse.from_extended(ext_response, ext_offer)
            >>> "Price too high" in combined.data["text"]
            True

    To customize how text is combined when both response and offer have text,
    subclass SAOResponse and override the `text_combiner` class method.
    """

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Outcome | None = None
    data: dict[str, Any] | None = None

    @classmethod
    def text_combiner(cls, response_text: str, offer_text: str) -> str:
        """Combine text from response and offer.

        Override this method in subclasses to customize text combination behavior.

        Args:
            response_text: Text from the response.
            offer_text: Text from the offer.

        Returns:
            Combined text string.
        """
        return f"{response_text}\n{offer_text}"

    @classmethod
    def from_extended(
        cls,
        response: ResponseType | ExtendedResponseType,
        offer: Outcome | ExtendedOutcome | None,
    ) -> SAOResponse:
        """Create SAOResponse from extended response and offer types.

        Args:
            response: The response, either ResponseType or ExtendedResponseType.
            offer: The offer, either Outcome, ExtendedOutcome, or None.

        Returns:
            SAOResponse with merged data from both response and offer.

        Remarks:
            - If both response and offer have data with the same key, the keys are
              renamed with "_response" and "_offer" postfixes respectively.
            - If "text" exists in only one of them, it is copied normally.
            - If "text" exists in both, the `text_combiner` class method is used to combine them.
            - To customize text combination, subclass SAOResponse and override `text_combiner`.
        """
        # Extract response type and response data
        if isinstance(response, ExtendedResponseType):
            response_type = response.response
            response_data = response.data
        else:
            response_type = response
            response_data = None

        # Extract outcome and offer data
        if isinstance(offer, ExtendedOutcome):
            outcome = offer.outcome
            offer_data = offer.data
        else:
            outcome = offer
            offer_data = None

        # Merge data
        merged_data: dict[str, Any] | None = None

        if response_data is not None or offer_data is not None:
            merged_data = {}

            response_data = response_data or {}
            offer_data = offer_data or {}

            # Handle text field specially
            response_text = response_data.get("text")
            offer_text = offer_data.get("text")

            if response_text is not None and offer_text is not None:
                # Both have text - combine them using the class method
                merged_data["text"] = cls.text_combiner(response_text, offer_text)
            elif response_text is not None:
                merged_data["text"] = response_text
            elif offer_text is not None:
                merged_data["text"] = offer_text

            # Merge other keys, handling conflicts
            all_keys = set(response_data.keys()) | set(offer_data.keys())
            for key in all_keys:
                if key == "text":
                    continue  # Already handled

                in_response = key in response_data
                in_offer = key in offer_data

                if in_response and in_offer:
                    # Conflict - rename both with postfixes
                    merged_data[f"{key}_response"] = response_data[key]
                    merged_data[f"{key}_offer"] = offer_data[key]
                elif in_response:
                    merged_data[key] = response_data[key]
                else:
                    merged_data[key] = offer_data[key]

            # If merged_data is empty, set to None
            if not merged_data:
                merged_data = None

        return cls(response=response_type, outcome=outcome, data=merged_data)


@define
class SAOState(GBState):
    """The `MechanismState` of SAO"""

    current_offer: Outcome | None = None
    current_proposer: str | None = None
    current_proposer_agent: str | None = None
    n_acceptances: int = 0
    new_offers: list[tuple[str, Outcome | None]] = field(factory=list)
    new_offerer_agents: list[str | None] = field(factory=list)
    last_negotiator: str | None = None
    current_data: dict[str, Any] | None = None
    new_data: list[tuple[str, dict[str, Any] | None]] = field(factory=list)


@define(frozen=True)
class SAONMI(NegotiatorMechanismInterface):
    """The `NegotiatorMechanismInterface` of SAO"""

    end_on_no_response: bool = True
    """End the negotiation if any agent responded with None"""

    one_offer_per_step: bool = False
    """If true, a step should be atomic with only one action from one negotiator"""

    offering_is_accepting: bool = True
    """If true, offering is considered an acceptance of that offer which means that the offerer need not accept the offer again to make it an agreement"""

    @property
    def state(self) -> SAOState:
        """Current state of the SAO negotiation mechanism.

        Returns:
            SAOState: State containing step number, current offer, and negotiation status
        """
        return self._mechanism.state  # type: ignore

    @property
    def history(self) -> list[SAOState]:
        """Complete history of all SAO negotiation states.

        Returns:
            list[SAOState]: Chronological list of states from negotiation start to current
        """
        return self._mechanism.history

    @property
    def extended_trace(self) -> list[tuple[int, str, Outcome]]:
        """Returns the negotiation history as a list of step, negotiator, offer tuples"""
        return self._mechanism.extended_trace  # type: ignore

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Returns the negotiation history as a list of negotiator, offer tuples"""
        return self._mechanism.trace  # type: ignore

    @property
    def offers(self) -> list[Outcome]:
        """Returns offers exchanged in order"""
        return self._mechanism.offers  # type: ignore

    def negotiator_offers(self, negotiator_id: str) -> list[Outcome]:
        """Get all offers made by a specific negotiator.

        Args:
            negotiator_id: ID of the negotiator whose offers to retrieve

        Returns:
            list[Outcome]: List of all outcomes proposed by this negotiator
        """
        return self._mechanism.negotiator_offers(negotiator_id)  # type: ignore


@lru_cache(1)
def all_negotiator_types() -> list[type[SAONegotiator]]:
    """
    Returns all the negotiator types defined in negmas.sao.negotiators
    """
    import negmas
    from negmas.helpers import get_class
    from negmas.sao import SAONegotiator
    from negmas.gb import GBNegotiator
    from negmas.gb.negotiators.utilbased import UtilBasedNegotiator
    from negmas.gb.negotiators.cab import CABNegotiator, CARNegotiator, CANNegotiator
    from negmas.gb.negotiators.war import WABNegotiator, WARNegotiator, WANNegotiator

    excluded = {
        UtilBasedNegotiator,
        CABNegotiator,
        CARNegotiator,
        CANNegotiator,
        WABNegotiator,
        WARNegotiator,
        WANNegotiator,
    }

    results = []
    for _ in dir(negmas.sao.negotiators):
        try:
            type = get_class(f"negmas.sao.negotiators.{_}")
            type()
        except Exception:
            continue
        if issubclass(type, SAONegotiator) or issubclass(type, GBNegotiator):
            results.append(type)
    return list(set(results) - excluded)
