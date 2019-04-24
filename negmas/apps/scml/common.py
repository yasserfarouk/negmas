"""Common data-structures and objects used throughout the SCM world implementation"""
import itertools
import math
import sys
import uuid
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field, InitVar
from typing import Dict, Union, Tuple, Iterable, List, Optional, Any

import numpy as np

from negmas.outcomes import OutcomeType, Issue
from negmas.situated import Contract

INVALID_STEP = -1000
NO_PRODUCTION = -1
INVALID_UTILITY = -2000

g_last_product_id = 0
g_last_process_id = 0

DEFAULT_NEGOTIATOR = "negmas.sao.AspirationNegotiator"


__all__ = [
    "Product",
    "Process",
    "InputOutput",
    "RunningCommandInfo",
    "INVALID_STEP",
    "NO_PRODUCTION",
    "INVALID_UTILITY",
    "ManufacturingProfile",
    "ManufacturingProfileCompiled",
    "ProductManufacturingInfo",
    "FactoryStatusUpdate",
    "Job",
    "ProductionNeed",
    "MissingInput",
    "ProductionReport",
    "ProductionFailure",
    "FinancialReport",
    "SCMLAgreement",
    "SCMLAction",
    "CFP",
    "Loan",
    "InsurancePolicy",
    "Factory",
    "FactoryState",
    "DEFAULT_NEGOTIATOR",
]


@dataclass
class Product:
    __slots__ = ["id", "production_level", "name", "expires_in", "catalog_price"]
    """A product that can be transacted in."""
    id: int
    """Product index. Must be set during construction and **MUST** be unique for products in the same world"""
    production_level: int
    """The level of this product in the production graph."""
    name: str
    """Object name"""
    expires_in: Optional[int]
    """Number of steps within which the product must be consumed. None means never"""
    catalog_price: Optional[float]
    """Catalog price of the product."""

    def __str__(self):
        """String representation is simply the name"""
        return self.name + (
            f"(cp:{self.catalog_price:0.02f})" if self.catalog_price is not None else ""
        )

    def __post_init__(self):
        global g_last_product_id
        if self.id is None:
            self.id = g_last_product_id
            g_last_product_id += 1
        if self.name is None:
            self.name = str(self.id)

    def __hash__(self):
        return self.id.__hash__()


@dataclass(frozen=True)
class InputOutput:
    """An input/output to a production process"""

    __slots__ = ["product", "quantity", "step"]
    product: int
    """Index of the product used as input or output"""
    quantity: int
    """Quantity needed/produced"""
    step: float
    """Relative time within the production at which the input is needed (output is produced)"""


@dataclass
class Process:
    __slots__ = [
        "id",
        "production_level",
        "name",
        "inputs",
        "outputs",
        "historical_cost",
    ]
    id: int
    """A manufacturing process."""
    production_level: int
    """The level of this process in the production graph"""
    name: str
    """Object name"""
    inputs: List[InputOutput]
    """list of input product name + quantity required and time of consumption relative to the time required for 
    production (value from 0 to 1)"""
    outputs: List[InputOutput]
    """list of output product names, quantity required and when it becomes available relative to the time required for 
    production (value from 0 to 1)"""
    historical_cost: Optional[float]
    """Average cost for running this process in some world. Filled by the world"""

    def __str__(self):
        """String representation is simply the name"""
        return self.name + (
            f"(cp:{self.historical_cost})" if self.historical_cost is not None else ""
        )

    def __post_init__(self):
        global g_last_process_id
        if self.id is None:
            self.id = g_last_process_id
            g_last_process_id += 1
        if self.name is None:
            self.name = str(self.id)

    def __hash__(self):
        """The hash depends only on the name"""
        return str(self).__hash__()


@dataclass
class ManufacturingProfile:
    """The costs/time required for running a process on a line (with associated cancellation costs etc). This
    data-structure carries full information about the `Process` es instead of just its index as in
    `ManufacturingProfileCompiled`. It is intended to be used to construct factories

    See Also:
        `Factory`

    """

    __slots__ = [
        "n_steps",
        "cost",
        "initial_pause_cost",
        "running_pause_cost",
        "resumption_cost",
        "cancellation_cost",
        "line",
        "process",
    ]
    n_steps: int
    """Number of steps needed to complete the manufacturing"""
    cost: float
    """Cost of manufacturing"""
    initial_pause_cost: float
    """Cost of pausing incurred only at the step a pause is started"""
    running_pause_cost: float
    """Running cost of pausing"""
    resumption_cost: float
    """Cost of resuming a process"""
    cancellation_cost: float
    """Cost of cancelling the process before the last step"""
    line: int
    """The line index"""
    process: Process
    """The `Process` associated with this profile"""


@dataclass
class FactoryStatusUpdate:
    __slots__ = ["balance", "storage"]
    balance: float
    """The update to the balance"""
    storage: Dict[int, int]
    """The updates to be applied to the storage after this step"""

    def __post_init__(self):
        if not isinstance(self.storage, defaultdict):
            self.storage = defaultdict(int, self.storage)

    def make_empty(self) -> None:
        """Makes the update an empty one."""
        self.balance = 0.0
        self.storage = defaultdict(int)

    def combine(self, other: "FactoryStatusUpdate") -> None:
        """
        Combines this status update with another one in place

        Args:
            other: The other status update

        Returns:
            None
        """
        if other is None:
            return
        self.balance += other.balance
        to_remove = []
        for k in set(list(self.storage.keys()) + list(other.storage.keys())):
            self.storage[k] += other.storage.get(k, 0)
            if self.storage[k] == 0:
                to_remove.append(k)
        for k in to_remove:
            self.storage.pop(k, None)

    @classmethod
    def combine_sets(
        cls,
        dst: Dict[int, "FactoryStatusUpdate"],
        src: Dict[int, "FactoryStatusUpdate"],
    ):
        """
        Combines a set of updates over time with another in place (overriding `first`)
        Args:
            dst: First set of updates to be combined into
            src: second set of updates to be combined from

        Returns:

        """
        to_remove = []
        for i, update in src.items():
            dst[i].combine(update)
            if dst[i].balance == 0 and len(dst[i].storage) == 0:
                to_remove.append(i)
        for i in to_remove:
            dst.pop(i, None)
        return None

    @property
    def is_empty(self):
        return self.balance == 0 and (
            len(self.storage) == 0 or sum(self.storage.values()) == 0
        )

    @classmethod
    def empty(cls):
        return FactoryStatusUpdate(balance=0.0, storage={})

    def __str__(self):
        return (
            f"balance: {self.balance}, "
            + f'{str({k: v for k, v in self.storage.items()}) if self.storage is not None else ""}'
        )


@dataclass
class RunningCommandInfo:
    __slots__ = ["profile", "beg", "end", "action", "updates", "step", "paused"]
    profile: ManufacturingProfile
    """The manufacturing profile associated with this command. Most importantly, it gives the process and line"""
    beg: int
    """The time the command is to be executed"""
    end: int
    """The number of steps starting at `beg` for this command to end (it ends at end - 1)"""
    step: int
    """The time-step relative to `beg` at the factory is currently executing the `Process` indicated in `profile`.
    `step` will always go up by one every simulation step except if the command is paused where it does not change
    """
    paused: bool
    """True if the command is paused"""
    action: str
    """The command type. For the current implementation it will always be run or none for no command"""
    updates: Dict[int, "FactoryStatusUpdate"]
    """The status updates implied by this command with their times relative to `beg`"""

    @property
    def n_steps(self) -> int:
        return self.end - self.beg

    def ended_before(self, t: int):
        return self.end <= t

    def started_on_or_after(self, t: int):
        return self.beg >= t

    def __str__(self):
        # if self.is_none:
        #     return 'No command'
        return (
            f"{self.action if self.action != 'none' else 'running'} process "
            f"{self.profile.process.id} @ {self.profile.line} steps: {self.beg}~{self.end - 1}"
        )

    @property
    def is_none(self):
        return self.action == "none"

    @is_none.setter
    def is_none(self, is_none):
        self.action = "none"

    @classmethod
    def do_nothing(cls):
        # noinspection PyTypeChecker
        return cls(
            profile=None,
            beg=-1,
            end=-1,
            action="none",
            updates={},
            step=0,
            paused=False,
        )


@dataclass
class Job:
    """Describes a job to be run on one production line of a `Factory`."""

    __slots__ = ["profile", "time", "line", "action", "contract", "override"]
    profile: int
    """The process for run commands"""
    time: int
    """The time the command is to be executed"""
    line: int
    """Index of the line on which the job is to be scheduled. Notice that it will be ignored for `run` actions."""
    action: str
    """The command type. For the current implementation it can be run/pause/resume/stop/cancel with `cancel` cancelling 
    any other command type."""
    contract: Optional[Contract]
    """The sell contract associated with the command"""
    override: bool
    """Whether to override existing commands when the job is to be executed."""

    def __str__(self):
        s = f'{self.action} {self.profile if self.action == "run" else ""} at {self.time} on {self.line}'
        s += f'{" override" if self.override else ""}'
        if self.contract is not None:
            s += f" for {self.contract.id}"
        return s

    def is_cancelling(self, job: "Job") -> bool:
        """
        Determines if the given jobs cancels this one

        Args:
            job:

        Returns:

        """
        if self.line != job.line:
            return False
        return (
            job.action == "cancel"
            or (self.action in ("run", "start") and job.action == "stop")
            or (self.action == "pause" and job.action == "resume")
            or (self.action == "resume" and job.action == "pause")
        )


@dataclass
class ProductionNeed:
    """Describes some quantity of a product that is needed to honor a (sell) contract."""

    __slots__ = [
        "product",
        "needed_for",
        "quantity_to_buy",
        "quantity_in_storage",
        "step",
    ]
    product: int
    """The product needed"""
    needed_for: Contract
    """The contract for which the product is needed"""
    quantity_to_buy: int
    """The quantity need to be bought"""
    quantity_in_storage: int
    """The quantity already found in storage"""
    step: int
    """The time step at which the product is needed"""

    def __str__(self):
        """String representation is simply the name"""
        return (
            f"Need {self.quantity_to_buy} ({self.quantity_in_storage} exist) of {self.product} at "
            + f" {self.step} for {self.needed_for}"
        )


@dataclass
class MissingInput:
    __slots__ = ["product", "quantity"]
    product: int
    quantity: int

    def __str__(self):
        return f"{self.product}: {self.quantity}"


@dataclass
class ProductionFailure:
    __slots__ = ["line", "command", "missing_inputs", "missing_money", "missing_space"]
    line: int
    """ID of the line that failed"""
    command: RunningCommandInfo
    """Information about the command that failed"""
    missing_inputs: List[MissingInput]
    """The missing inputs if any with their quantities"""
    missing_money: float
    """The amount of money needed for production that is not available"""
    missing_space: int
    """The amount space needed in storage but not found"""

    def __str__(self):
        s = f"{str(self.command)} @ {self.line} failed:"
        if self.missing_money > 0:
            s += f" money {self.missing_money}"
        if len(self.missing_inputs) > 0:
            s += f" inputs: {[str(_) for _ in self.missing_inputs]}"
        if self.missing_space > 0:
            s += f" space {self.missing_space}"
        return s


@dataclass
class ProductionReport:
    line: int
    """ID of the line"""
    started: Optional[RunningCommandInfo]
    """Commands started"""
    continuing: Optional[RunningCommandInfo]
    """Command that is continuing"""
    finished: Optional[RunningCommandInfo]
    """Command finished"""
    failure: Optional[ProductionFailure]
    """Failures"""
    updates: FactoryStatusUpdate
    """Updates applied to the factory"""

    @property
    def failed(self):
        return self.failure is not None

    @property
    def is_empty(self):
        return self.no_production and self.updates.is_empty

    @property
    def no_production(self):
        return self.started is None and self.finished is None and self.failure is None

    def __str__(self):
        if self.is_empty:
            return ""
        s = f"{self.line}: " if self.line >= 0 else f"Updates: "
        if self.failed:
            s += f"{str(self.failure)} "
        else:
            if self.started is not None and self.finished is not None:
                s += f"started/finished {str(self.started)} "
            elif self.started is not None:
                s += f"started {str(self.started)} "
            elif self.finished is not None:
                s += f"finished {str(self.finished)} "
        if not self.updates.is_empty:
            s += f"{str(self.updates)}"
        return s


@dataclass
class SCMLAgreement(OutcomeType):
    time: int
    """delivery time"""
    unit_price: float
    """unit price"""
    quantity: int
    """quantity"""
    penalty: Optional[float] = None
    """penalty"""
    signing_delay: int = -1
    """Delay between agreement conclusion and signing it to be binding"""


@dataclass
class CFP(OutcomeType):
    """A Call for proposal upon which a negotiation can start"""

    is_buy: bool
    """If true, the author wants to buy otherwise to sell. Non-negotiable."""
    publisher: str
    """the publisher name. Non-negotiable."""
    product: int
    """product ID. Non-negotiable."""
    time: Union[int, Tuple[int, int], List[int]]
    """delivery time. May be negotiable."""
    unit_price: Union[float, Tuple[float, float], List[float]]
    """unit price. May be negotiable."""
    quantity: Union[int, Tuple[int, int], List[int]]
    """quantity. May be negotiable."""
    penalty: Optional[Union[float, Tuple[float, float], List[float]]] = None
    """penalty per missing item in case the seller cannot provide the required quantity. May be negotiable."""
    signing_delay: Optional[Union[int, Tuple[int, int], List[int]]] = None
    """The grace period after which the agents are asked to confirm signing the contract"""
    money_resolution: Optional[float] = None
    """If not None then it is the minimum unit of money (e.g. 1 for dollar, 0.01 for cent, etc)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Unique CFP ID"""

    def __str__(self):
        s = f'{"buy" if self.is_buy else "sell"} '
        s += f"{self.product} "
        s += f"(t: {self.time}, u: {self.unit_price}, q: {self.quantity}"
        if self.penalty is not None:
            s += f", penalty: {self.penalty}"
        if self.signing_delay is not None:
            s += f", sign after: {self.signing_delay}"
        s += ")"
        return s

    def satisfies(self, query: Dict[str, Any]) -> bool:
        """
        Tests whether the CFP satisfies the conditions set by the query

        Args:

            query: A dictionary given the conditions. See `Remarks` for details

        Remarks:

           - The query dictionary can be used to specify any conditions that are required in the CFP. Only CFPs that
             satisfy ALL the conditions specified in the query are considered satisfying the query. The following keys
             can be set with corresponding meanings:

             is_buy
               True or False. If both are OK, just do not add this key

             publisher
               A string or `SCMLAgent` specifying a specific publisher

             publishers
               A list of publishers (see publisher key)

             product
              A string specifying a product name

             products
               A list of products (see product key)

             time
               A number, list or 2-items-tuple (range) specifying possible times to consider satisfactory

             unit_price
               A number, list or 2-items-tuple (range) specifying possible prices to consider satisfactory

             quantity
                A number, list or 2-items-tuple (range) specifying possible quantities to consider OK

             penalty
               A number, list or 2-items-tuple (range) specifying possible penalties to consider satisfactory

        """

        def _overlap(
            a: Union[
                int, float, Tuple[float, float], List[float], Tuple[int, int], List[int]
            ],
            b: Union[
                float, Tuple[float, float], List[float], int, Tuple[int, int], List[int]
            ],
        ):
            def _test_single(a, b):
                if not isinstance(b, Iterable):
                    return a == b
                if isinstance(b, tuple):
                    return b[0] <= a <= b[1]
                return a in b

            if not isinstance(b, Iterable):
                a, b = b, a

            if not isinstance(a, Iterable):
                return _test_single(a, b)

            if isinstance(a, tuple):
                if isinstance(b, tuple):
                    return b[0] <= a[0] <= b[1] or b[0] <= a[1] <= b[1]
                return any(_test_single(_, a) for _ in b)  # type: ignore

            return any(_test_single(_, b) for _ in a)

        for k, v in query.items():
            if k == "is_buy" and self.is_buy != v:
                return False
            if k == "publisher" and self.publisher != v:
                return False
            if k == "publishers" and self.publisher not in v:
                return False
            if k == "products" and self.product not in v:
                return False
            if k == "product_ids" and self.product not in v:
                return False
            if k == "product_indices" and self.product not in v:
                return False
            if k == "product" and self.product != v:
                return False
            if k == "product_id" and self.product != v:
                return False
            if k == "product_index" and self.product != v:
                return False
            if k == "time" and not _overlap(v, self.time):
                return False
            if k == "unit_price" and not _overlap(v, self.unit_price):
                return False
            if k == "penalty":
                if self.penalty is None and v is None:
                    return True
                if self.penalty is None or v is None:
                    return False
                if not _overlap(v, self.penalty):
                    return False
            if k == "quantity" and not _overlap(v, self.quantity):
                return False
        return True

    @property
    def issues(self):
        """Returns the set of issues associated with this CFP. Notice that some of the issues may have a single value"""

        def _values(x, ensure_list=False, ensure_int=False):
            if isinstance(x, tuple) and ensure_list:
                if x[0] == x[1]:
                    if ensure_list and self.money_resolution is not None:
                        if ensure_int:
                            return [
                                int(
                                    math.floor(x[0] / self.money_resolution)
                                    * self.money_resolution
                                )
                            ]
                        return [
                            math.floor(x[0] / self.money_resolution)
                            * self.money_resolution
                        ]
                    else:
                        if ensure_int:
                            return [int(x[0])]
                        return [x[0]]
                if isinstance(x[0], float) or isinstance(x[1], float):
                    xs = (
                        int(math.floor(x[0] / self.money_resolution)),
                        int(math.floor(x[1] / self.money_resolution)),
                    )
                    xs = list(
                        _ * self.money_resolution for _ in range(xs[0], xs[1] + 1)
                    )
                elif isinstance(x[0], int):
                    xs = list(range(x[0], x[1] + 1))
                else:
                    xs = list(range(int(x[0]), int(x[1]) + 1))
                if len(xs) == 0:
                    if ensure_list and self.money_resolution is not None:
                        if ensure_int:
                            return [
                                int(
                                    math.floor(x[0] / self.money_resolution)
                                    * self.money_resolution
                                )
                            ]
                        return [
                            math.floor(x[0] / self.money_resolution)
                            * self.money_resolution
                        ]
                    if ensure_int:
                        return [int(x[0])]
                    return [x[0]]
                if ensure_int:
                    return list(set(int(_) for _ in xs))
                return list(set(xs))
            if isinstance(x, Iterable):
                if ensure_int:
                    return list(set(int(_) for _ in x))
                return list(set(x))
            if ensure_int:
                return [int(x)]
            return [x]

        issues = [
            Issue(
                name="time",
                values=_values(self.time, ensure_list=True, ensure_int=True),
            ),
            Issue(
                name="quantity",
                values=_values(self.quantity, ensure_list=True, ensure_int=True),
            ),
            Issue(
                name="unit_price",
                values=_values(
                    self.unit_price, ensure_list=self.money_resolution is not None
                ),
            ),
        ]
        if self.penalty is not None:
            issues.append(
                Issue(
                    name="penalty",
                    values=_values(
                        self.penalty, ensure_list=self.money_resolution is not None
                    ),
                )
            )
        if self.signing_delay is not None:
            issues.append(
                Issue(
                    name="signing_delay",
                    values=_values(self.quantity, ensure_list=True, ensure_int=True),
                )
            )
        return issues

    @property
    def outcomes(self):
        return Issue.enumerate(issues=self.issues, max_n_outcomes=1000)

    @property
    def min_time(self):
        if isinstance(self.time, tuple):
            return self.time[0]
        elif isinstance(self.time, list):
            return min(self.time)
        return self.time

    @property
    def max_time(self):
        if isinstance(self.time, tuple):
            return self.time[1]
        elif isinstance(self.time, list):
            return max(self.time)
        return self.time

    @property
    def min_quantity(self):
        if isinstance(self.quantity, tuple):
            return self.quantity[0]
        elif isinstance(self.quantity, list):
            return min(self.quantity)
        return self.quantity

    @property
    def max_quantity(self):
        if isinstance(self.quantity, tuple):
            return self.quantity[1]
        elif isinstance(self.quantity, list):
            return max(self.quantity)
        return self.quantity

    @property
    def min_unit_price(self):
        if isinstance(self.unit_price, tuple):
            return self.unit_price[0]
        elif isinstance(self.unit_price, list):
            return min(self.unit_price)
        return self.unit_price

    @property
    def max_unit_price(self):
        if isinstance(self.unit_price, tuple):
            return self.unit_price[1]
        elif isinstance(self.unit_price, list):
            return max(self.unit_price)
        return self.unit_price

    @property
    def min_signing_delay(self):
        if self.signing_delay is None:
            return None
        if isinstance(self.signing_delay, tuple):
            return self.signing_delay[0]
        elif isinstance(self.signing_delay, list):
            return min(self.signing_delay)
        return self.signing_delay

    @property
    def max_signing_delay(self):
        if self.signing_delay is None:
            return None
        if isinstance(self.signing_delay, tuple):
            return self.signing_delay[1]
        elif isinstance(self.signing_delay, list):
            return max(self.signing_delay)
        return self.signing_delay

    @property
    def min_penalty(self):
        if self.penalty is None:
            return None
        if isinstance(self.penalty, tuple):
            return self.penalty[0]
        elif isinstance(self.penalty, list):
            return min(self.penalty)
        return self.penalty

    @property
    def max_penalty(self):
        if self.penalty is None:
            return None
        if isinstance(self.penalty, tuple):
            return self.penalty[1]
        elif isinstance(self.penalty, list):
            return max(self.penalty)
        return self.penalty

    def to_java(self):
        d = {
            "is_buy": self.is_buy,
            "publisher": self.publisher,
            "product": self.product,
            "id": self.id,
            "money_resolution": float(self.money_resolution)
            if self.money_resolution is not None
            else 0.0,
            "min_time": int(self.min_time),
            "max_time": int(self.max_time),
            "min_quantity": int(self.min_quantity),
            "max_quantity": int(self.max_quantity),
            "min_unit_price": float(self.min_unit_price),
            "max_unit_price": float(self.max_unit_price),
            "min_penalty": float(self.min_penalty)
            if self.min_penalty is not None
            else None,
            "max_penalty": float(self.max_penalty)
            if self.max_penalty is not None
            else None,
            "min_signing_delay": int(self.min_signing_delay)
            if self.min_signing_delay is not None
            else None,
            "max_signing_delay": int(self.max_signing_delay)
            if self.max_signing_delay is not None
            else None,
        }
        return d

    @classmethod
    def from_java(
        cls, idict: Dict[str, Any], class_name: Optional[str] = None
    ) -> "CFP":
        if idict["min_time"] == idict["max_time"]:
            t = idict["min_time"]
        else:
            t = (idict["min_time"], idict["max_time"])
        if idict["min_quantity"] == idict["max_quantity"]:
            q = idict["min_quantity"]
        else:
            q = (idict["min_quantity"], idict["max_quantity"])
        if idict["min_unit_price"] == idict["max_unit_price"]:
            up = idict["min_unit_price"]
        else:
            up = (idict["min_unit_price"], idict["max_unit_price"])
        if not idict.get("min_penalty", None) or not idict.get("max_penalty", None):
            p = None
        else:
            if idict["min_penalty"] == idict["max_penalty"]:
                p = idict["min_penalty"]
            else:
                p = (idict["min_penalty"], idict["max_penalty"])
        if not idict.get("min_signing_delay", None) or not idict.get(
            "max_signing_delay", None
        ):
            s = None
        else:
            if idict["min_signing_delay"] == idict["max_signing_delay"]:
                s = idict["min_signing_delay"]
            else:
                s = (idict["min_signing_delay"], idict["max_signing_delay"])

        return cls(
            is_buy=idict["is_buy"],
            publisher=idict["publisher"],
            product=idict["product"],
            time=t,
            unit_price=up,
            quantity=q,
            penalty=p,
            signing_delay=s,
            money_resolution=idict.get("money_resolution", None),
            id=idict.get("id", None),
        )


@dataclass
class SCMLAction:
    line: str
    """Line to execute the action on (need not be given if the profile is given"""
    profile: Optional[int]
    """Index of the profile to execute"""
    action: str
    """The action which may be start, stop, pause, resume"""
    time: int = 0
    """Time to execute the action at"""


@dataclass
class ManufacturingProfileCompiled:
    """The costs/time required for running a process on a line (with associated cancellation costs etc).

    See Also:
        `Factory`

    """

    __slots__ = [
        "n_steps",
        "cost",
        "initial_pause_cost",
        "running_pause_cost",
        "resumption_cost",
        "cancellation_cost",
        "line",
        "process",
    ]
    n_steps: int
    """Number of steps needed to complete the manufacturing"""
    cost: float
    """Cost of manufacturing"""
    initial_pause_cost: float
    """Cost of pausing incurred only at the step a pause is started"""
    running_pause_cost: float
    """Running cost of pausing"""
    resumption_cost: float
    """Cost of resuming a process"""
    cancellation_cost: float
    """Cost of cancelling the process before the last step"""
    line: int
    """The line index"""
    process: int
    """The `Process` index"""

    @classmethod
    def from_manufacturing_profile(
        cls, profile: ManufacturingProfile, process2ind: Dict[Process, int]
    ):
        return ManufacturingProfileCompiled(
            n_steps=profile.n_steps,
            cost=profile.cost,
            initial_pause_cost=profile.initial_pause_cost,
            running_pause_cost=profile.running_pause_cost,
            resumption_cost=profile.resumption_cost,
            cancellation_cost=profile.cancellation_cost,
            line=profile.line,
            process=process2ind[profile.process],
        )


@dataclass
class ProductManufacturingInfo:
    """Gives full information about a manufacturing process that can generate or consume a product.

    See Also:
        `consuming` and `producing` of `Factory`
    """

    __slots__ = ["profile", "quantity", "step"]
    profile: int
    """The `ManufacturingProfile` index"""
    quantity: int
    """The quantity generated/consumed by running this manufacturing info"""
    step: int
    """The step from the beginning at which the `Product` is received/consumed"""


@dataclass
class FinancialReport:
    """Reports that financial standing of an agent at a given time in the simulation"""

    agent: str
    """Agent ID"""
    step: int
    """Time of the report"""
    cash: float
    """Cash at hand"""
    liabilities: float
    """Total liabilities (loans)"""
    inventory: float
    """Value of everything in the inventory priced at catalog prices."""
    credit_rating: float
    """The agent's credit rating as a fraction of the maximum credit rating (1 indicates highest credit rating)."""

    @property
    def balance(self):
        """The balance of the agent defined as the difference between its available cash + inventory and its liabilities

        Remarks:

            - If the inventory was not calculated (due to having at least one product with unknown catalog price),
              it is used as zero in the equation.
        """
        return (
            self.cash + self.inventory - self.liabilities
            if self.inventory is not None
            else self.cash - self.liabilities
        )


@dataclass
class Loan:
    amount: float
    """Loan amount"""
    starts_at: int
    """The time-step at which payment starts"""
    total: float
    """The total to be paid including the amount + interests"""
    interest: float
    """The interest rate per step"""
    installment: float
    """The amount to be paid in one installment"""
    n_installments: int
    """The number of installments"""

    def __str__(self):
        return (
            f"{self.amount} @ {self.interest} paid in {self.n_installments} [{self.installment} each] "
            f"for a total {self.total} [starts at {self.starts_at}]"
        )

    class Java:
        implements = ["jnegmas.apps.scml.common.Loan"]


RunningNegotiationInfo = namedtuple(
    "RunningNegotiationInfo", ["negotiator", "annotation", "uuid", "extra"]
)
"""Keeps track of running negotiations for an agent"""

NegotiationRequestInfo = namedtuple(
    "NegotiationRequestInfo",
    ["partners", "issues", "annotation", "uuid", "negotiator", "extra"],
)
"""Keeps track to negotiation requests that an agent sent"""


@dataclass
class InsurancePolicy:
    premium: float
    contract: Contract
    at_time: int
    against: "SCMLAgent"


@dataclass
class FactoryState:
    """Read Only State of a factory"""

    max_storage: int
    """Maximum storage allowed in this factory"""
    line_schedules: np.array
    """An array of n_lines * n_steps giving the line schedules"""
    storage: Dict[int, int]
    """Mapping from product index to the amount available in the inventory"""
    wallet: float
    """Money available for purchases"""
    hidden_money: float
    """Amount of money hidden by the agent"""
    hidden_storage: Dict[int, int]
    """Mapping from product index to the amount hidden by the agent"""
    loans: float
    """The total money owned as loans"""
    n_lines: int
    """The number of lines in the factory, will be set using the `profiles` input"""
    profiles: List[ManufacturingProfile]
    """A list of profiles used to initialize the factory"""
    next_step: int
    """Next simulation step for this factory"""
    commands: np.array
    """The production command currently running"""
    jobs: Dict[Tuple[int, int], Job]
    """The jobs waiting to be run on the factory indexed by (time, line) tuples"""


@dataclass
class Factory:
    """Represents a factory within an SCML world. It is only accessed by the World so it need not be made public."""

    initial_storage: InitVar[Dict[int, int]]
    """Initial storage"""
    initial_wallet: InitVar[float] = 0.0
    """Initial Wallet"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=True)
    """Object name"""
    profiles: List[ManufacturingProfile] = field(default_factory=list)
    """A list of profiles used to initialize the factory"""
    max_storage: int = sys.maxsize
    """Maximum storage allowed in this factory"""
    min_storage: int = 0
    """Minimum allowed storage per product"""
    min_balance: int = 0
    """Minimum allowed balance"""
    initial_balance: float = field(init=False, default=0.0)
    """Initial balance of the factory"""
    _commands: np.array = field(init=False)
    """The production command currently running"""
    _line_schedules: np.array = field(init=False)
    _storage: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    """Mapping from product index to the amount available in the inventory"""
    _total_storage: int = field(init=False, default=0)
    """Total storage"""
    _wallet: float = field(default=0, init=False)
    """Money available for purchases"""
    _hidden_money: float = field(default=0, init=False)
    """Amount of money hidden by the agent"""
    _hidden_storage: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )
    """Mapping from product index to the amount hidden by the agent"""
    _loans: float = field(default=0.0, init=False)
    """The total money owned as loans"""
    _n_lines: int = field(init=False)
    """The number of lines in the factory, will be set using the `profiles` input"""
    _jobs: Dict[Tuple[int, int], Job] = field(default_factory=dict)
    """The jobs waiting to be run on the factory indexed by (time, line) tuples"""

    _next_step: int = field(init=False, default=0)
    """Current simulation step"""
    _carried_updates: FactoryStatusUpdate = field(
        init=False, default_factory=lambda: FactoryStatusUpdate.empty()
    )
    """Carried updates from last executed command"""

    def __post_init__(self, initial_storage: Dict[int, int], initial_wallet=0.0):
        # no matter what are the line indices in the given profiles, the lines used by the factory
        # will be numbered from 0 to `n_lines` - 1
        if self.max_storage is None or self.max_storage < 0:
            self.max_storage = sys.maxsize
        given_lines = sorted(list(set(p.line for p in self.profiles)))
        mapping = dict(zip(given_lines, range(len(given_lines))))
        for profile in self.profiles:
            profile.line = mapping[profile.line]
        self._n_lines = len(given_lines)
        self._commands = np.array(
            [RunningCommandInfo.do_nothing() for _ in range(self._n_lines)]
        )
        self._line_schedules = np.ones(self._n_lines, dtype=int) * NO_PRODUCTION
        self._storage = defaultdict(int)
        self._total_storage = 0
        for k, v in initial_storage.items():
            self._storage[k] = v
            self._total_storage += v
        self._wallet = initial_wallet
        self._carried_updates = FactoryStatusUpdate.empty()
        self.initial_balance = initial_wallet

    @property
    def hidden_money(self) -> float:
        return self._hidden_money

    @property
    def hidden_storage(self) -> Dict[int, int]:
        return self._hidden_storage

    @property
    def n_lines(self) -> int:
        return self._n_lines

    @property
    def jobs(self) -> Dict[Tuple[int, int], Job]:
        return self._jobs

    @property
    def commands(self) -> np.array:
        return self._commands

    @property
    def line_schedules(self) -> np.array:
        return self._line_schedules

    @property
    def wallet(self) -> float:
        return self._wallet

    @property
    def storage(self) -> Dict[int, int]:
        return self._storage

    @property
    def loans(self) -> float:
        return self._loans

    @property
    def total_storage(self) -> int:
        # assert len(self._storage) == 0 or min(self._storage.values()) >= self.min_storage, f'min {min(self._storage.values()) if len(self._storage) > 0 else 0} < {self.min_storage} storage {self._storage}'
        # assert (len(self._storage) == 0 and self._total_storage == 0) or \
        #     self._total_storage == sum(self._storage.values()), f'Total storage {self._total_storage} but the sum is {sum(self._storage.values())}'
        return self._total_storage

    @property
    def balance(self) -> float:
        """The total balance of the factory"""
        return self._wallet - self._loans

    @property
    def next_step(self) -> int:
        return self._next_step

    def add_loan(self, total: float) -> None:
        self._loans += total

    def receive(self, payment: float) -> None:
        self.pay(-payment)

    def pay(self, payment: float) -> None:
        if self._wallet - payment < self.min_balance:
            raise ValueError(f"Cannot pay {payment} as  we have only {self._wallet}")
        self._wallet -= payment

    def transport_to(self, product: int, quantity: int) -> None:
        if self._storage[product] + quantity < self.min_storage:
            raise ValueError(
                f"Cannot transfer {quantity} of {product} as  we have only {self._storage[product]} "
                f"(min {self.min_storage}, max {self.max_storage})"
            )
        if self._total_storage + quantity > self.max_storage:
            raise ValueError(
                f"Cannot transfer {quantity} of {product} as  we have only {self._storage[product]} "
                f"(min {self.min_storage}, max {self.max_storage})"
            )
        self._storage[product] += quantity
        self._total_storage += quantity

    # @todo Schedulers and simulators do not know about transportation or transfer delays. They should
    # @todo Factory buy and sell functions do not take transportation and transfer delays into account

    def buy(self, product: int, quantity: int, price: float) -> None:
        if self._wallet < price or self._total_storage + quantity > self.max_storage:
            raise ValueError(
                f"Cannot buy {quantity} (total {self._total_storage}/{sum(self._storage.values())}) of "
                f"{product} for {price} (wallet {self._wallet} / balance {self.balance})"
            )
        self._wallet -= price
        self._storage[product] += quantity
        self._total_storage += quantity

    def sell(self, product: int, quantity: int, price: float) -> None:
        if self._storage[product] < quantity + self.min_storage:
            raise ValueError(
                f"Cannot sell {quantity} (have {self._storage[product]}) of "
                f"{product} for {price} (wallet {self._wallet} / balance {self.balance})"
            )
        self._storage[product] -= quantity
        self._total_storage -= quantity
        self._wallet += price

    def transport_from(self, product: int, quantity: int) -> None:
        self.transport_to(product=product, quantity=-quantity)

    def hide_funds(self, amount: float) -> None:
        to_hide = min(amount, self._wallet)
        self._hidden_money += to_hide
        self._wallet -= to_hide

    def hide_product(self, product: int, quantity: int) -> None:
        to_hide = min(quantity, self._storage.get(product, 0))
        self._hidden_storage[product] += to_hide
        self._storage[product] -= to_hide

    def unhide_funds(self, amount: float) -> None:
        to_hide = min(amount, self._hidden_money)
        self._hidden_money -= to_hide
        self._wallet += to_hide

    def unhide_product(self, product: int, quantity: int) -> None:
        to_hide = min(quantity, self._hidden_storage.get(product, 0))
        self._hidden_storage[product] -= to_hide
        self._storage[product] += to_hide

    def schedule(self, job: Job, override=False) -> None:
        """
        Schedules the given job at its `time` and `line` optionally overriding whatever was already scheduled
        Args:
            job:
            override:

        Returns:
            Success/failure
        """
        # you can only schedule jobs at the following simulation step
        t, line, profile = job.time, job.line, self.profiles[job.profile]
        if job.action in ("run", "start"):
            line = profile.line
        if t < self._next_step - 1 or line >= self._n_lines or line < 0:
            raise ValueError(
                f"cannot schedule at time {t} (current {self._next_step - 1}) on line {line} "
                f"of {self._n_lines}"
            )
        existing_job = self._jobs.get((t, line), None)
        if existing_job is None:
            self._jobs[(t, line)] = job
            return
        if existing_job.is_cancelling(job):
            del self._jobs[(t, line)]
            return
        if not override:
            raise ValueError(
                f"Job {str(existing_job)} is scheduled at {t} and overriding is not allowed"
            )
        self._jobs[(t, line)] = job

    def _apply_updates(self, updates: FactoryStatusUpdate) -> None:
        if updates.balance != 0.0:
            self._wallet += updates.balance
        if updates.storage is not None:
            for k, v in updates.storage.items():
                self._storage[k] += v
                self._total_storage += v

    def step(self) -> List[ProductionReport]:
        reports = []
        for line in range(self._n_lines):
            # step the current production process
            if self._commands[line].ended_before(self._next_step):
                self._commands[line].action = "none"
            report = self._step_line(line=line)
            reports.append(report)
            self._apply_updates(report.updates)
        if not self._carried_updates.is_empty:
            reports.append(
                ProductionReport(
                    line=-1,
                    started=None,
                    continuing=None,
                    finished=None,
                    failure=None,
                    updates=self._carried_updates,
                )
            )
            self._apply_updates(self._carried_updates)
            self._carried_updates = FactoryStatusUpdate.empty()
        self._next_step += 1
        return reports

    def _run(self, profile: ManufacturingProfile, override=True) -> None:
        """running is executed at the beginning of the step t

        Args:
            profile: the profile to start giving both the line and process
            override: If true, override any running processes paying cancellation cost for these processes

        Remarks:

            - The output of a process that runs from step t to step t + n - 1 will only be in storage at step t + n

        """

        # if I am not allowed to override, then this command has no effect and I return an empty status update
        t = self._next_step
        line = profile.line
        running_command = self._commands[line]
        if not running_command.is_none and not override:
            return
        process = profile.process
        n, cost = profile.n_steps, profile.cost
        updates = defaultdict(lambda: FactoryStatusUpdate.empty())
        command = RunningCommandInfo(
            action="run",
            profile=profile,
            beg=t,
            end=t + n,
            updates=updates,
            paused=False,
            step=0,
        )
        for need in process.inputs:
            updates[int(math.floor(need.step * n))].storage[
                need.product
            ] -= need.quantity
        for output in process.outputs:
            updates[int(math.ceil(output.step * n))].storage[
                output.product
            ] += output.quantity
        updates[0].balance -= cost

        # cancel the running command by stopping it and then run the new command
        if not running_command.is_none:
            self._stop(line=profile.line)
        self._commands[line] = command
        self._line_schedules[line] = process.id

    def _pause(self, line: int) -> None:
        """pausing is executed at the end of the step

        Args:
            line: the line on which the process is running

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - Not implemented yet
            - pausing when nothing is running is not an error and will return an empty status update

        """
        running_command = self._commands[line]
        if running_command.is_none:
            return
        running_command.updates[running_command.step].combine(
            FactoryStatusUpdate(
                balance=-running_command.profile.initial_pause_cost, storage={}
            )
        )
        running_command.paused = True

    def _resume(self, line: int) -> None:
        """resumption is executed at the end of the step (starting next step count down)


        Args:
            line: the line on which the process is running

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - Not implemented yet
            - resuming when nothing is paused is not an error and will return an empty status update

        """
        running_command = self._commands[line]
        if running_command.is_none:
            return
        profile = running_command.profile
        running_command.updates[running_command.step].combine(
            FactoryStatusUpdate(balance=-profile.resumption_cost, storage={})
        )
        running_command.paused = False

    def _stop(self, line: int) -> None:
        """stopping is executed at the beginning of the current step

        Args:
            line: the line on which the process is running

        Returns:

            Optional[Dict[int, FactoryStatusUpdate]]: The status updated for all times that need to be updated to cancel
            the command if it is not None. If None is returned then scheduling failed.

        Remarks:

            - stopping when nothing is running is not an error and will just return an empty schedule
        """

        # stopping a no-action command is always successful
        running_command = self._commands[line]
        if running_command.is_none:
            return
        profile = running_command.profile
        t = self._next_step
        running_command.paused = False
        running_command.end = t + 1
        running_command.updates = {
            running_command.step: running_command.updates[running_command.step].combine(
                FactoryStatusUpdate(balance=-profile.cancellation_cost, storage={})
            )
        }

    def _step_line(self, line: int) -> ProductionReport:
        """
        Steps the line to the time-step `t` assuming that it is already stepped to time-step t-1 given the storage

        Args:
            line: the line to step

        Returns:
            ProductionReport
        """
        t = self._next_step
        running_command = self._commands[line]
        job = self._jobs.get((t, line), None)
        updates = FactoryStatusUpdate.empty()
        if job is None and running_command.is_none:
            return ProductionReport(
                updates=updates,
                continuing=None,
                started=None,
                finished=None,
                failure=None,
                line=line,
            )
        if job is not None:
            if job.action in ("run", "start"):
                self._run(profile=self.profiles[job.profile], override=job.override)
            elif job.action == "pause":
                self._pause(line=job.line)
            elif job.action == "resume":
                self._resume(line=job.line)
            elif job.action == "stop":
                self._stop(line=job.line)
            else:
                raise ValueError(f"action {job.action} is unknown")
            del self._jobs[(t, line)]

        # now all updates in the command are correct except for the running pause cost which we add here
        running_command = self._commands[line]
        profile = running_command.profile
        if running_command.paused:
            running_command.end += 1
            running_command.updates = {
                k + 1: v for k, v in running_command.updates.items()
            }
            running_command.updates[running_command.step].combine(
                FactoryStatusUpdate(balance=-profile.running_pause_cost, storage={})
            )
        updates = running_command.updates.get(running_command.step, None)
        if not running_command.paused:
            running_command.step += 1
        if updates is None or updates.is_empty:
            return ProductionReport(
                updates=FactoryStatusUpdate.empty(),
                continuing=running_command if running_command.beg < t else None,
                started=running_command if running_command.beg == t else None,
                finished=running_command if running_command.end <= t + 1 else None,
                failure=None,
                line=line,
            )
        if updates is not None:
            del running_command.updates[running_command.step - 1]
        available_storage = self.max_storage - self._total_storage
        missing_inputs = []
        missing_money = 0
        failed = False
        missing_space = 0
        if updates.balance < 0 and self._wallet < -updates.balance:
            failed = True
            missing_money = -updates.balance - self._wallet
        for product_id, quantity in updates.storage.items():
            if quantity < 0 and self._storage.get(product_id, 0) < -quantity:
                failed = True
                missing_inputs.append(
                    MissingInput(product=product_id, quantity=-quantity)
                )
            elif quantity > 0:
                available_storage -= quantity
                if available_storage < 0:
                    failed = True
                    missing_space -= available_storage
                    available_storage = 0
        if failed:
            running_command.action = "none"
            failure = ProductionFailure(
                line=line,
                command=running_command,
                missing_money=missing_money,
                missing_inputs=missing_inputs,
                missing_space=missing_space,
            )
            return ProductionReport(
                updates=FactoryStatusUpdate.empty(),
                continuing=running_command if running_command.beg < t else None,
                started=running_command if running_command.beg == t else None,
                finished=running_command if running_command.end <= t + 1 else None,
                failure=failure,
                line=line,
            )
        if running_command.ended_before(t + 1):
            self._carried_updates.combine(
                running_command.updates.get(
                    running_command.step, FactoryStatusUpdate.empty()
                )
            )
        return ProductionReport(
            updates=updates,
            continuing=running_command if running_command.beg < t else None,
            started=running_command if running_command.beg == t else None,
            finished=running_command if running_command.end <= t + 1 else None,
            failure=None,
            line=line,
        )
