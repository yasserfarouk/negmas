from __future__ import annotations

from collections import namedtuple
from enum import Enum
from typing import TYPE_CHECKING, Any

from attr import define, field

if TYPE_CHECKING:
    from negmas.mechanisms import Mechanism
    from negmas.outcomes import Issue

    from .agent import Agent


__all__ = [
    "PROTOCOL_CLASS_NAME_FIELD",
    "EDGE_TYPES",
    "DEFAULT_EDGE_TYPES",
    "EDGE_COLORS",
    "RunningNegotiationInfo",
    "NegotiationRequestInfo",
    "RenegotiationRequest",
    "NegotiationInfo",
    "Operations",
]

PROTOCOL_CLASS_NAME_FIELD = "__mechanism_class_name"

EDGE_TYPES = [
    "negotiation-requests-rejected",
    "negotiation-requests-accepted",
    "negotiations-rejected",
    "negotiations-started",
    "negotiations-failed",
    "negotiations-succeeded",
    "contracts-concluded",
    "contracts-cancelled",
    "contracts-signed",
    "contracts-nullified",
    "contracts-erred",
    "contracts-breached",
    "contracts-executed",
]
"""All types of graphs that can be drawn for a world"""

DEFAULT_EDGE_TYPES = [
    "negotiation-requests-rejected",
    "negotiation-requests-accepted",
    "negotiations-rejected",
    "negotiations-started",
    "negotiations-failed",
    "contracts-concluded",
    "contracts-cancelled",
    "contracts-signed",
    "contracts-breached",
    "contracts-executed",
]
"""Default set of edge types to show by defaults in draw"""

EDGE_COLORS = {
    "negotiation-requests-rejected": "silver",
    "negotiation-requests-accepted": "gray",
    "negotiations-rejected": "tab:pink",
    "negotiations-started": "tab:blue",
    "negotiations-failed": "tab:red",
    "negotiations-succeeded": "tab:green",
    "contracts-concluded": "tab:green",
    "contracts-cancelled": "fuchsia",
    "contracts-nullified": "yellow",
    "contracts-erred": "darksalmon",
    "contracts-signed": "blue",
    "contracts-breached": "indigo",
    "contracts-executed": "black",
}


RunningNegotiationInfo = namedtuple(
    "RunningNegotiationInfo",
    ["negotiator", "annotation", "uuid", "extra", "my_request"],
)
"""Keeps track of running negotiations for an agent"""

NegotiationRequestInfo = namedtuple(
    "NegotiationRequestInfo",
    ["partners", "issues", "annotation", "uuid", "negotiator", "requested", "extra"],
)
"""Keeps track to negotiation requests that an agent sent"""


@define(frozen=True)
class RenegotiationRequest:
    """A request for renegotiation."""

    publisher: Agent
    issues: list[Issue]
    annotation: dict[str, Any] = field(factory=dict)


@define(frozen=True)
class NegotiationInfo:
    """Saves information about a negotiation"""

    mechanism: Mechanism | None
    partners: list[Agent]
    annotation: dict[str, Any]
    issues: list[Issue]
    requested_at: int
    rejectors: list[Agent] | None = None
    caller: Agent | None = None
    group: str | None = None


class Operations(Enum):
    Negotiations = 1
    ContractSigning = 2
    AgentSteps = 3
    ContractExecution = 4
    SimulationStep = 5
    StatsUpdate = 6
