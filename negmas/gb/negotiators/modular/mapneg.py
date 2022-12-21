from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from ...common import ResponseType
from ...components import AcceptancePolicy, Model, OfferingPolicy
from .modular import GBModularNegotiator

if TYPE_CHECKING:
    from negmas.gb import GBState

    from ....outcomes import Outcome
    from ...components import GBComponent
    from ..base import GBNegotiator

__all__ = ["MAPNegotiator"]


class MAPNegotiator(GBModularNegotiator):
    """
    A negotiator that is  constructed from three components:

    1. A `Model` (or a list of `Model` s) that are used to model the partner or the negotiation context
    2. An `AcceptanceStrategy` That is used for responding to offers.
    3. An `OfferingStrategy` That is used for generating offers.


    For all callbacks, models are called first, followed by the acceptance strategy followed by the offering strategy

    """

    def __init__(
        self,
        *args,
        acceptance: AcceptancePolicy | GBNegotiator | None = None,
        offering: OfferingPolicy | GBNegotiator | None = None,
        models: list[Model] | None = None,
        model_names: None | list[str] = None,
        extra_components: list[GBComponent] | None = None,
        extra_component_names: list[str] | None = None,
        **kwargs,
    ):
        components, names = [], []
        if models:
            for i, (m, n) in enumerate(
                zip(models, model_names if model_names else itertools.repeat(None))
            ):
                if not m:
                    continue
                components.append(m)
                names.append(n if n else f"model{i}")
        if acceptance:
            components.append(acceptance)
            names.append("acceptance")
        if offering:
            components.append(offering)
            names.append("offering")
        if extra_components:
            extra_components = list(extra_components)
            components += extra_components
            if not extra_component_names:
                extra_component_names = [
                    f"extra{i}" for i in range(len(extra_components))
                ]
            names += extra_component_names
        self._acceptance = acceptance
        self._offering = offering
        self._models = models
        super().__init__(*args, components=components, component_names=names, **kwargs)
        if not self._acceptance:
            self.capabilities["respond"] = False
        if not self._offering:
            self.capabilities["propose"] = False

    def generate_response(
        self, state: GBState, offer: Outcome, source: str
    ) -> ResponseType:
        if not self._acceptance:
            return ResponseType.REJECT_OFFER
        return self._acceptance.respond(state, offer, source)

    def generate_proposal(self, state: GBState) -> Outcome | None:
        if not self._offering:
            return None
        return self._offering.propose(state)
