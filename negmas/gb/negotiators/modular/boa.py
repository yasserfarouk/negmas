from __future__ import annotations

from typing import TYPE_CHECKING

from ...components import AcceptancePolicy, Model, OfferingPolicy
from .mapneg import MAPNegotiator

if TYPE_CHECKING:
    from ...components import GBComponent
    from ...negotiators.base import GBNegotiator

__all__ = ["make_boa", "BOANegotiator"]


def make_boa(
    acceptance: AcceptancePolicy | GBNegotiator | None = None,
    offering: OfferingPolicy | GBNegotiator | None = None,
    model: Model | None = None,
    extra_components: list[GBComponent] | None = None,
    extra_component_names: list[str] | None = None,
    **kwargs,
):
    """
    A negotiator that is  constructed from three components:

    1. A `PartnerUFunModel` to learn the ufun of the partner
    2. An `AcceptanceStrategy` That is used for responding to offers.
    3. An `OfferingStrategy` That is used for generating offers.


    For all callbacks, partner-model is called first, followed by the acceptance strategy followed by the offering strategy

    """

    return MAPNegotiator(
        acceptance=acceptance,
        offering=offering,
        models=[model] if model else None,
        model_names=["model"] if model else None,
        extra_components=extra_components,
        extra_component_names=extra_component_names,
        type_name="BOANegotiator",
        **kwargs,
    )


class BOANegotiator(MAPNegotiator):
    """
    A negotiator that is  constructed from three components:

    1. A `PartnerUFunModel` to learn the ufun of the partner
    2. An `AcceptanceStrategy` That is used for responding to offers.
    3. An `OfferingStrategy` That is used for generating offers.


    For all callbacks, partner-model is called first, followed by the acceptance strategy followed by the offering strategy

    """

    def __init__(
        *args,
        acceptance: AcceptancePolicy | GBNegotiator | None = None,
        offering: OfferingPolicy | GBNegotiator | None = None,
        model: Model | None = None,
        extra_components: list[GBComponent] | None = None,
        extra_component_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            acceptance=acceptance,
            offering=offering,
            models=[model] if model else None,
            model_names=["model"] if model else None,
            extra_components=extra_components,
            extra_component_names=extra_component_names,
            type_name="BOANegotiator",
            **kwargs,
        )
