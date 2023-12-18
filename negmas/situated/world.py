from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import random
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Literal

import numpy as np
import pandas as pd
import yaml

from negmas.checkpoints import CheckpointMixin
from negmas.common import Action, NegotiatorMechanismInterface
from negmas.config import negmas_config
from negmas.events import Event, EventLogger, EventSink, EventSource
from negmas.genius import ANY_JAVA_PORT, DEFAULT_JAVA_PORT, get_free_tcp_port
from negmas.helpers import (
    create_loggers,
    exception2str,
    get_class,
    humanize_time,
    unique_name,
)
from negmas.helpers.inout import ConfigReader, add_records
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.outcomes import Issue, Outcome, outcome2dict
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.preferences import Preferences
from negmas.serialization import to_flat_dict
from negmas.types import NamedObject
from negmas.warnings import NegmasImportWarning, warn

from .agent import Agent
from .breaches import Breach, BreachProcessing
from .bulletinboard import BulletinBoard
from .common import (
    DEFAULT_EDGE_TYPES,
    EDGE_COLORS,
    EDGE_TYPES,
    NegotiationInfo,
    Operations,
)
from .contract import Contract
from .entity import Entity
from .helpers import deflistdict
from .mechanismfactory import MechanismFactory
from .monitors import StatsMonitor, WorldMonitor
from .save import save_stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    import networkx as nx
except ImportError:
    nx = None

__all__ = ["World"]

LOG_BASE = negmas_config("log_base", Path.home() / "negmas" / "logs")


def _path(path) -> Path:
    """Creates an absolute path from given path which can be a string"""
    if isinstance(path, str):
        if path.startswith("~"):
            path = Path.home() / ("/".join(path.split("/")[1:]))
    return Path(path).absolute()


class World(EventSink, EventSource, ConfigReader, NamedObject, CheckpointMixin, ABC):
    """Base world class encapsulating a world that runs a simulation with several agents interacting within some
    dynamically changing environment.

    A world maintains its own session.

    Args:

        * General *

        name: World Name
        bulletin_board: A bulletin board object to use. If not given one will be created
        awi_type: The type used for agent world interfaces (must descend from or behave like `AgentWorldInterface` )
        info: A dictionary of key-value pairs that is kept within the world but never used. It is useful for storing
              contextual information. For example, when running tournaments.

        * Simulation parameters *

        n_steps: Total simulation time in steps
        time_limit: Real-time limit on the simulation
        operations: A list of `Operations` to run in order during every simulation step

        * Negotiation Parameters *

        negotiation_speed: The number of negotiation steps per simulation step. None means infinite
        neg_n_steps: Maximum number of steps allowed for a negotiation.
        neg_step_time_limit: Time limit for single step of the negotiation protocol.
        neg_time_limit: Real-time limit on each single negotiation.
        shuffle_negotiations: Whether negotiations are shuffled everytime when stepped.
        negotiation_quota_per_step: Number of negotiations an agent is allowed to start per step
        negotiation_quota_per_simulation: Number of negotiations an agent is allowed to start in the simulation
        start_negotiations_immediately: If true negotiations start immediately when registered rather than waiting
                                        for the next step
        mechanisms: The mechanism types allowed in this world associated with each keyward arguments to be passed
                    to it.

        * Signing parameters *

        default_signing_delay: The default number of steps between contract conclusion and signing it. Only takes
                               effect if `force_signing` is `False`
        force_signing: If true, agents are not asked to sign contracts. They are forced to do so. In this
                       case, `default_singing_delay` is not effective and signature is immediate
        batch_signing: If true, contracts are signed in batches not individually


        * Breach Processing *

        breach_processing: How to handle breaches. Can be any of `BreachProcessing` values

        * Logging *

        log_folder: Folder to save all logs
        log_to_file: If true, will log to a file
        log_file_name: Name of the log file
        log_file_level: The log-level to save to file (WARNING, ERROR, INFO, DEBUG, CRITICAL, ...)
        log_ufuns: Log utility functions
        log_negotiations: Log all negotiation events
        log_to_screen: Whether to log to screen
        log_screen_level: The log-level to show on screen (WARNING, ERROR, INFO, DEBUG, CRITICAL, ...)
        no_logs: If True, All logging will be disabled no matter what other options are given.
        log_stats_every: If nonzero and positive, the period of saving stats
        construct_graphs: If true, information needed to draw graphs using `draw` method are kept.
        event_file_name: If not None, the file-name to store events into.
        event_types: Types of events to log

        * What to save *

        save_signed_contracts: Save all signed contracts
        save_cancelled_contracts: Save all cancelled contracts
        save_negotiations: Save all negotiation records
        save_resolved_breaches: Save all resolved breaches
        save_unresolved_breaches: Save all unresolved breaches

        * Exception Handling *

        ignore_agent_exceptions: Ignore agent exceptions and keep running
        ignore_negotiation_exceptions: If true, all mechanism exceptions are ignored and the mechanism is aborted
        ignore_simulation_exceptions: Ignore simulation exceptions and keep running
        ignore_contract_execution_exceptions: Ignore contract execution exceptions and keep running
        safe_stats_monitoring: Never throw an exception for a failure to save stats or because of a Stats Monitor
                               object

        * Checkpoints *

        checkpoint_every: The number of steps to checkpoint after. Set to <= 0 to disable
        checkpoint_folder: The folder to save checkpoints into. Set to None to disable
        checkpoint_filename: The base filename to use for checkpoints (multiple checkpoints will be prefixed with
                             step number).
        single_checkpoint: If true, only the most recent checkpoint will be saved.
        extra_checkpoint_info: Any extra information to save with the checkpoint in the corresponding json file as
                               a dictionary with string keys
        exist_ok: IF true, checkpoints override existing checkpoints with the same filename.
        genius_port: the port used to connect to Genius for all negotiators in this mechanism (0 means any).
    """

    def __init__(
        self,
        bulletin_board: BulletinBoard | None = None,
        n_steps: int = 10000,
        time_limit: int | float | None = 60 * 60,
        negotiation_speed: int | None = None,
        neg_n_steps: int | None = 100,
        neg_time_limit: int | float | None = None,
        neg_step_time_limit: int | float | None = 60,
        shuffle_negotiations=True,
        negotiation_quota_per_step: int | float = sys.maxsize,
        negotiation_quota_per_simulation: int | float = sys.maxsize,
        default_signing_delay=1,
        force_signing=False,
        batch_signing=True,
        breach_processing=BreachProcessing.NONE,
        mechanisms: dict[str, dict[str, Any]] | None = None,
        awi_type: str = "negmas.situated.AgentWorldInterface",
        start_negotiations_immediately: bool = False,
        log_folder=None,
        log_to_file=True,
        log_ufuns=False,
        log_negotiations: bool = False,
        log_to_screen: bool = False,
        log_stats_every: int = 0,
        log_file_level=logging.DEBUG,
        log_screen_level=logging.ERROR,
        no_logs=False,
        event_file_name="events.json",
        event_types=None,
        log_file_name="log.txt",
        save_signed_contracts: bool = True,
        save_cancelled_contracts: bool = True,
        save_negotiations: bool = True,
        save_resolved_breaches: bool = True,
        save_unresolved_breaches: bool = True,
        ignore_agent_exceptions: bool = False,
        ignore_negotiation_exceptions: bool = False,
        ignore_contract_execution_exceptions: bool = False,
        ignore_simulation_exceptions: bool = False,
        safe_stats_monitoring: bool = False,
        construct_graphs: bool = False,
        checkpoint_every: int = 1,
        checkpoint_folder: str | Path | None = None,
        checkpoint_filename: str | None = None,
        extra_checkpoint_info: dict[str, Any] | None = None,
        single_checkpoint: bool = True,
        exist_ok: bool = True,
        operations: Collection[Operations] = (
            Operations.StatsUpdate,
            Operations.Negotiations,
            Operations.ContractSigning,
            Operations.AgentSteps,
            Operations.ContractExecution,
            Operations.SimulationStep,
            Operations.ContractSigning,
            Operations.StatsUpdate,
        ),
        info: dict[str, Any] | None = None,
        genius_port: int = DEFAULT_JAVA_PORT,
        disable_agent_printing: bool = False,
        debug: bool = False,
        name: str | None = None,
        id: str | None = None,
    ):
        self._debug = debug
        if debug:
            ignore_agent_exceptions = False
            ignore_negotiation_exceptions = False
            ignore_contract_execution_exceptions = False
            ignore_simulation_exceptions = False
        self.info = None
        self.disable_agent_printing = disable_agent_printing
        self.ignore_simulation_exceptions = ignore_simulation_exceptions
        self.ignore_negotiation_exceptions = ignore_negotiation_exceptions
        if force_signing:
            batch_signing = False
        super().__init__()
        self.__next_operation_index = 0
        NamedObject.__init__(self, name, id=id)
        CheckpointMixin.checkpoint_init(
            self,
            step_attrib="current_step",
            every=checkpoint_every,
            folder=checkpoint_folder,
            filename=checkpoint_filename,
            info=extra_checkpoint_info,
            exist_ok=exist_ok,
            single=single_checkpoint,
        )
        self.name = (
            name.replace("/", ".")
            if name is not None
            else unique_name(base=self.__class__.__name__, add_time=True, rand_digits=5)
        )
        self.id = unique_name(self.name, add_time=True, rand_digits=8)
        self._no_logs = no_logs
        if log_folder is not None:
            self._log_folder = Path(log_folder).absolute()
        else:
            self._log_folder = Path(LOG_BASE)
            if name is not None:
                for n in name.split("/"):
                    self._log_folder /= n
            else:
                self._log_folder /= self.name
        if event_file_name:
            self._event_logger = EventLogger(
                self._log_folder / event_file_name, types=event_types
            )
            self.register_listener(None, self._event_logger)
        else:
            self._event_logger = None
        if log_file_name is None:
            log_file_name = "log.txt"
        if len(log_file_name) == 0:
            log_to_file = False
        if (
            log_folder
            or log_negotiations
            or log_stats_every
            or log_to_file
            or log_ufuns
        ):
            self._log_folder.mkdir(parents=True, exist_ok=True)
            self._agent_log_folder = self._log_folder / "_agent_logs"
            self._agent_log_folder.mkdir(parents=True, exist_ok=True)
        self._agent_loggers: dict[str, logging.Logger] = {}
        self.log_file_name = (
            str(self._log_folder / log_file_name) if log_to_file else None
        )
        self.log_file_level = log_file_level
        self.log_screen_level = log_screen_level
        self.log_to_screen = log_to_screen
        self.log_negotiations = log_negotiations
        self.logger = (
            create_loggers(
                file_name=self.log_file_name,
                module_name=None,
                screen_level=log_screen_level if log_to_screen else None,
                file_level=log_file_level,
                app_wide_log_file=True,
            )
            if not no_logs
            else None
        )
        self.ignore_contract_execution_exceptions = ignore_contract_execution_exceptions
        self.ignore_agent_exception = ignore_agent_exceptions
        self.times: dict[str, float] = defaultdict(float)
        self.simulation_exceptions: dict[int, list[str]] = defaultdict(list)
        self.mechanism_exceptions: dict[int, list[str]] = defaultdict(list)
        self.contract_exceptions: dict[int, list[str]] = defaultdict(list)
        self.agent_exceptions: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.negotiator_exceptions: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self._negotiations: dict[str, NegotiationInfo] = {}
        self.unsigned_contracts: dict[int, set[Contract]] = defaultdict(set)
        self.breach_processing = breach_processing
        self.n_steps = n_steps
        self.save_signed_contracts = save_signed_contracts
        self.save_cancelled_contracts = save_cancelled_contracts
        self.save_negotiations = save_negotiations
        self.save_resolved_breaches = save_resolved_breaches
        self.save_unresolved_breaches = save_unresolved_breaches
        self.construct_graphs = construct_graphs
        self.operations = operations
        self._current_step = 0
        self.negotiation_speed = negotiation_speed
        self.default_signing_delay = default_signing_delay
        self.time_limit = time_limit if time_limit is not None else float("inf")
        self.neg_n_steps = neg_n_steps
        self.neg_time_limit = neg_time_limit
        self.neg_step_time_limit = neg_step_time_limit
        self.frozen_time = 0.0
        self._entities: dict[int, set[Entity]] = defaultdict(set)
        self._negotiations: dict[str, NegotiationInfo] = {}
        self.force_signing = force_signing
        self.neg_quota_step = negotiation_quota_per_step
        self.neg_quota_simulation = negotiation_quota_per_simulation
        self._start_time = None
        self._log_ufuns = log_ufuns
        self._log_negs = log_negotiations
        self.safe_stats_monitoring = safe_stats_monitoring
        self.shuffle_negotiations = shuffle_negotiations
        self.info = info if info is not None else dict()

        if isinstance(mechanisms, Collection) and not isinstance(mechanisms, dict):
            mechanisms = dict(zip(mechanisms, [dict()] * len(mechanisms)))
        self.mechanisms: dict[str, dict[str, Any]] | None = mechanisms
        self.awi_type = get_class(awi_type, scope=globals())

        self._log_folder = str(self._log_folder)
        self._stats: dict[str, list[Any]] = defaultdict(list)
        self.__stepped_mechanisms: set[str] = set()
        self.__n_negotiations = 0
        self.__n_contracts_signed = 0
        self.__n_contracts_concluded = 0
        self.__n_contracts_cancelled = 0
        self.__n_contracts_dropped = 0
        self.__stats_stage = 0
        self.__stage = 0
        self.__n_new_contract_executions = 0
        self.__n_new_breaches = 0
        self.__n_new_contract_errors = 0
        self.__n_new_contract_nullifications = 0
        self.__activity_level = 0
        self.__blevel = 0.0
        self._saved_contracts: dict[str, dict[str, Any]] = {}
        self._saved_negotiations: dict[str, dict[str, Any]] = {}
        self._saved_breaches: dict[str, dict[str, Any]] = {}
        self._started = False
        self.batch_signing = batch_signing
        self.agents: dict[str, Agent] = {}
        self.immediate_negotiations = start_negotiations_immediately
        self.stats_monitors: set[StatsMonitor] = set()
        self.world_monitors: set[WorldMonitor] = set()
        self._edges_negotiation_requests_accepted: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_negotiation_requests_rejected: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_negotiations_started: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_negotiations_rejected: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_negotiations_succeeded: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_negotiations_failed: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_concluded: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_signed: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_cancelled: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_nullified: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_erred: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_executed: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self._edges_contracts_breached: dict[
            int, dict[tuple[Agent, Agent], list[dict[str, Any]]]
        ] = defaultdict(deflistdict)
        self.neg_requests_sent: dict[str, int] = defaultdict(int)
        self.neg_requests_received: dict[str, int] = defaultdict(int)
        self.negs_registered: dict[str, int] = defaultdict(int)
        self.negs_succeeded: dict[str, int] = defaultdict(int)
        self.negs_failed: dict[str, int] = defaultdict(int)
        self.negs_timedout: dict[str, int] = defaultdict(int)
        self.negs_initiated: dict[str, int] = defaultdict(int)
        self.contracts_concluded: dict[str, int] = defaultdict(int)
        self.contracts_signed: dict[str, int] = defaultdict(int)
        self.neg_requests_rejected: dict[str, int] = defaultdict(int)
        self.contracts_dropped: dict[str, int] = defaultdict(int)
        self.breaches_received: dict[str, int] = defaultdict(int)
        self.breaches_committed: dict[str, int] = defaultdict(int)
        self.contracts_erred: dict[str, int] = defaultdict(int)
        self.contracts_nullified: dict[str, int] = defaultdict(int)
        self.contracts_executed: dict[str, int] = defaultdict(int)
        self.contracts_breached: dict[str, int] = defaultdict(int)
        self.attribs: dict[str, dict[str, Any]] = {}
        self._sim_start: float = 0
        self._step_start: float = 0
        if log_stats_every is None or log_stats_every < 1:
            self._stats_file_name = None
            self._stats_dir_name = None
        else:
            stats_file_name = _path(str(Path(self._log_folder) / "stats.csv"))
            self._stats_file_name = stats_file_name.name
            self._stats_dir_name = stats_file_name.parent

        self.bulletin_board: BulletinBoard
        self.set_bulletin_board(bulletin_board=bulletin_board)
        stats_calls = [_ for _ in self.operations if _ == Operations.StatsUpdate]
        self._single_stats_call = len(stats_calls) == 1
        self._two_stats_calls = len(stats_calls) == 2
        self._n_negs_per_agent_per_step: dict[str, int] = defaultdict(int)
        self._n_negs_per_agent: dict[str, int] = defaultdict(int)

        self.genius_port = (
            genius_port
            if genius_port > 0
            else ANY_JAVA_PORT
            if genius_port == ANY_JAVA_PORT
            else get_free_tcp_port()
        )
        self.params = dict(
            negotiation_speed=negotiation_speed,
            negotiation_can_cross_step_boundaries=not (
                self.negotiation_speed is None
                or (
                    self.neg_n_steps is not None
                    and self.negotiation_speed is not None
                    and self.neg_n_steps < self.negotiation_speed
                )
                or (
                    self.neg_n_steps is None
                    and (
                        self.neg_time_limit is not None
                        and not math.isinf(self.neg_time_limit)
                    )
                )
            ),
            default_signing_delay=default_signing_delay,
            batch_signing=batch_signing,
            breach_processing=breach_processing,
            mechanisms=mechanisms,
            start_negotiations_immediately=start_negotiations_immediately,
            ignore_agent_exceptions=ignore_agent_exceptions,
            ignore_negotiation_exceptions=ignore_negotiation_exceptions,
            ignore_contract_execution_exceptions=ignore_contract_execution_exceptions,
            ignore_simulation_exceptions=ignore_simulation_exceptions,
            operations=operations,
            genius_port=self.genius_port,
        )
        self.loginfo(f"{self.name}: World Created")

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats

    @property
    def breach_fraction(self) -> float:
        """Fraction of signed contracts that led to breaches"""
        n_breaches = sum(self.stats["n_breaches"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return n_breaches / n_signed_contracts if n_signed_contracts != 0 else 0.0

    breach_rate = breach_fraction

    def n_saved_contracts(self, ignore_no_issue: bool = True) -> int:
        """
        Number of saved contracts

        Args:
            ignore_no_issue: If true, only contracts resulting from negotiation (has some issues) will be counted
        """
        if ignore_no_issue:
            return len([_ for _ in self._saved_contracts.values() if _["issues"]])
        return len(self._saved_contracts)

    @property
    def agreement_fraction(self) -> float:
        """Fraction of negotiations ending in agreement and leading to signed contracts"""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = self.n_saved_contracts(True)
        return n_contracts / n_negs if n_negs != 0 else np.nan

    agreement_rate = agreement_fraction

    @property
    def cancellation_fraction(self) -> float:
        """Fraction of contracts concluded (through negotiation or otherwise)
        that were cancelled."""
        n_negs = sum(self.stats["n_negotiations"])
        n_contracts = self.n_saved_contracts(False)
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return (1.0 - n_signed_contracts / n_contracts) if n_contracts != 0 else np.nan

    cancellation_rate = cancellation_fraction

    def loginfo(self, s: str, event: Event | None = None) -> None:
        """logs info-level information

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        self.logger.info(f"{self._log_header()}: " + s.strip())

    def set_bulletin_board(self, bulletin_board):
        self.bulletin_board = (
            bulletin_board if bulletin_board is not None else BulletinBoard()
        )
        self.bulletin_board.add_section("breaches")
        self.bulletin_board.add_section("stats")
        self.bulletin_board.add_section("settings")
        self.bulletin_board.record("settings", self.n_steps, "n_steps")
        self.bulletin_board.record("settings", self.time_limit, "time_limit")
        self.bulletin_board.record(
            "settings", self.negotiation_speed, "negotiation_speed"
        )
        self.bulletin_board.record("settings", self.neg_n_steps, "neg_n_steps")
        self.bulletin_board.record("settings", self.neg_time_limit, "neg_time_limit")
        self.bulletin_board.record(
            "settings", self.neg_step_time_limit, "neg_step_time_limit"
        )
        self.bulletin_board.record(
            "settings", self.default_signing_delay, "default_signing_delay"
        )
        self.bulletin_board.record("settings", self.force_signing, "force_signing")
        self.bulletin_board.record("settings", self.batch_signing, "batch_signing")
        self.bulletin_board.record(
            "settings", self.breach_processing, "breach_processing"
        )
        self.bulletin_board.record(
            "settings",
            list(self.mechanisms.keys()) if self.mechanisms is not None else [],
            "mechanism_names",
        )
        self.bulletin_board.record(
            "settings",
            self.mechanisms if self.mechanisms is not None else dict(),
            "mechanisms",
        )
        self.bulletin_board.record(
            "settings", self.immediate_negotiations, "start_negotiations_immediately"
        )

    @classmethod
    def is_basic_stat(self, s: str) -> bool:
        """Checks whether a given statistic is agent specific."""
        return (
            s in ["activity_level", "breach_level", "n_bankrupt", "n_breaches"]
            or s.startswith("n_contracts")
            or s.startswith("n_negotiation")
            or s.startswith("n_registered_negotiations")
        )

    @property
    def current_step(self):
        return self._current_step

    def _agent_logger(self, aid: str) -> logging.Logger:
        """Returns the logger associated with a given agent"""
        if aid not in self._agent_loggers.keys():
            self._agent_loggers[aid] = (
                create_loggers(
                    file_name=self._agent_log_folder / f"{aid}.txt",
                    module_name=None,
                    file_level=self.log_file_level,
                    app_wide_log_file=False,
                    module_wide_log_file=False,
                )
                if not self._no_logs
                else None
            )
        return self._agent_loggers[aid]

    def logdebug_agent(self, aid: str, s: str, event: Event | None = None) -> None:
        """logs debug to the agent individual log

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs:
            return
        logger = self._agent_logger(aid)
        logger.debug(f"{self._log_header()}: " + s.strip())

    def on_event(self, event: Event, sender: EventSource):
        """Received when an event is raised"""
        if event.type == "negotiator_exception":
            negotiator = event.data.get("negotiator")
            if not negotiator:
                return
            agent = negotiator.owner
            if not agent:
                return
            self.logdebug_agent(
                agent.id,
                f"Negotiator {negotiator.name} raised: "
                + str(event.data.get("exception", "Unknown exception")),
            )

    @property
    def log_folder(self):
        return self._log_folder

    def loginfo_agent(self, aid: str, s: str, event: Event | None = None) -> None:
        """logs information to the agent individual log

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        logger = self._agent_logger(aid)
        logger.info(f"{self._log_header()}: " + s.strip())

    def logwarning_agent(self, aid: str, s: str, event: Event | None = None) -> None:
        """logs warning to the agent individual log

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        logger = self._agent_logger(aid)
        logger.warning(f"{self._log_header()}: " + s.strip())

    def logerror_agent(self, aid: str, s: str, event: Event | None = None) -> None:
        """logs information to the agent individual log

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        logger = self._agent_logger(aid)
        logger.error(f"{self._log_header()}: " + s.strip())

    def logdebug(self, s: str, event: Event | None = None) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        self.logger.debug(f"{self._log_header()}: " + s.strip())

    def logwarning(self, s: str, event: Event | None = None) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        self.logger.warning(f"{self._log_header()}: " + s.strip())

    def logerror(self, s: str, event: Event | None = None) -> None:
        """logs error-level information

        Args:
            s (str): The string to log
            event (Event): The event to announce after logging

        """
        if event:
            self.announce(event)
        if self._no_logs or not self.logger:
            return
        self.logger.error(f"{self._log_header()}: " + s.strip())

    @property
    def time(self) -> float:
        """Elapsed time since world started in seconds. 0.0 if the world did not start running"""
        if self._start_time is None:
            return 0.0
        if (
            self.n_steps is not None
            and self.current_step >= self.n_steps
            and self.frozen_time > 0.0
        ):
            return self.frozen_time
        return time.perf_counter() - self._start_time

    @property
    def remaining_time(self) -> float | None:
        """Returns remaining time in seconds. None if no time limit is given."""
        if not self._start_time:
            return self.time_limit
        limit = self.time_limit - (time.perf_counter() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def relative_time(self) -> float:
        """Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps."""

        if self.time_limit == float("inf") and self.n_steps is None:
            return 0.0

        relative_step = (
            self.current_step / self.n_steps if self.n_steps is not None else np.nan
        )
        relative_time = self.time / self.time_limit
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> int | None:
        """Returns the remaining number of steps until the end of the mechanism run. None if unlimited"""
        if self.n_steps is None:
            return None

        return self.n_steps - self.current_step

    @abstractmethod
    def breach_record(self, breach: Breach) -> dict[str, Any]:
        """Converts a breach to a record suitable for storage during the simulation"""

    def _register_breach(self, breach: Breach) -> None:
        # we do not report breachs with no victims
        if breach.victims is None or len(breach.victims) < 1:
            return
        for v in breach.victims:
            self.breaches_received[v] += 1
        self.breaches_committed[breach.perpetrator] += 1
        self.bulletin_board.record(
            section="breaches", key=breach.id, value=self.breach_record(breach)
        )

    @property
    def saved_negotiations(self) -> list[dict[str, Any]]:
        return list(self._saved_negotiations.values())

    def on_exception(self, entity: Entity, e: Exception) -> None:
        """
        Called when an exception happens.

        Args:
            entity: The entity that caused the exception
            e: The exception
        """

    def call(self, agent: Agent, method: Callable, *args, **kwargs) -> Any:
        """
        Calls a method on an agent updating exeption count

        Args:
            agent: The agent on which the method is to be called
            method: The bound method (bound to the agent)
            *args: position arguments
            **kwargs: keyword arguments

        Returns:
            whatever method returns
        """
        old_stdout = sys.stdout  # backup current stdout
        if self.disable_agent_printing:
            sys.stdout = open(os.devnull, "w")
        _strt = time.perf_counter()
        try:
            result = method(*args, **kwargs)
            _end = time.perf_counter()
            self.times[agent.id] = _end - _strt
            return result
        except Exception as e:
            _end = time.perf_counter()
            self.times[agent.id] = _end - _strt
            self.agent_exceptions[agent.id].append(
                (self._current_step, exception2str())
            )
            self.on_exception(agent, e)
            if not self.ignore_agent_exception:
                raise e
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logerror(
                f"Entity exception @{agent.id}: "
                f"{traceback.format_tb(exc_traceback)}",
                Event("entity-exception", dict(exception=e)),
            )
        finally:
            if self.disable_agent_printing:
                sys.stdout.close()
                sys.stdout = old_stdout  # reset old stdout

    def _add_edges(
        self,
        src: Agent | str,
        dst: list[Agent | str],
        target: dict[int, dict[tuple[Agent, Agent], list[dict[str, Any]]]],
        bi=False,
        issues: list[Issue] | None = None,
        agreement: dict[str, Any] | None = None,
    ):
        """Registers an edge"""
        if not self.construct_graphs:
            return
        attr = None
        if issues is not None:
            attr = {i.name: i.values for i in issues}
        if agreement is not None:
            attr = agreement
        for p in dst:
            if p == src:
                continue
            src_id = src.id if isinstance(src, Agent) else src
            p_id = p.id if isinstance(p, Agent) else p
            target[self.current_step][(src_id, p_id)].append(attr)
            if bi:
                target[self.current_step][(p_id, src_id)].append(attr)

    def is_valid_contract(self, contract: Contract) -> bool:
        """
        Confirms that the agreement is valid given the world rules.

        Args:
            contract: The contract being tested

        Return:
            Returns True for valid contracts and False for invalid contracts

        Remarks:

            - This test will be conducted after agents are asked to sign the contract
              and only for signed contracts.
            - If False is returned, the contract will considered unsigned and will be
              recorded as a concluded but not signed contract with no rejectors
        """
        return True

    def _sign_contract(self, contract: Contract) -> list[str] | None:
        """Called to sign a contract and returns whether or not it was signed"""
        # if self._contract_finalization_time(contract) >= self.n_steps or \
        #     self._contract_execution_time(contract) < self.current_step:
        #     return None
        partners = [self.agents[_] for _ in contract.partners]

        def _do_sign(c, p):
            s_ = c.signatures.get(p, None)
            if s_ is not None:
                return s_

            try:
                result = self.call(p, p.sign_all_contracts, [c])[0]
                if self.time >= self.time_limit:
                    result = None
                return result
            except Exception as e:
                self.agent_exceptions[p.id].append((self._current_step, str(e)))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                self.logerror(
                    f"Signature exception @ {p.name}: {traceback.format_tb(exc_traceback)}",
                    Event("agent-exception", dict(method="sign_contract", exception=e)),
                )
                return None

        if self.force_signing:
            signatures = [(partner, partner.id) for partner in partners]
            rejectors = []
        else:
            signatures = list(
                zip(partners, (_do_sign(contract, partner) for partner in partners))
            )
            rejectors = [
                partner for partner, signature in signatures if signature is None
            ]
        if len(rejectors) == 0:
            contract.signatures = {a.id: s for a, s in signatures}
            contract.signed_at = self.current_step
            for partner in partners:
                self.contracts_signed[partner.id] += 1
                self.call(partner, partner.on_contract_signed_, contract=contract)
                if self.time >= self.time_limit:
                    break
        else:
            for partner in partners:
                self.call(
                    partner,
                    partner.on_contract_cancelled_,
                    contract=contract,
                    rejectors=[_.id for _ in rejectors],
                )
                if self.time >= self.time_limit:
                    break
        return [_.id for _ in rejectors]

    def on_contract_processed(self, contract):
        """
        Called whenever a contract finished processing to be removed from unsigned contracts

        Args:
            contract: Contract

        Remarks:

            - called by on_contract_cancelled and on_contract_signed

        """
        unsigned = self.unsigned_contracts.get(self.current_step, None)
        if unsigned is None:
            return
        try:
            unsigned.remove(contract)
        except KeyError:
            pass

    @abstractmethod
    def contract_record(self, contract: Contract) -> dict[str, Any]:
        """Converts a contract to a record suitable for permanent storage"""

    def _contract_record(self, contract: Contract) -> dict[str, Any]:
        """Converts a contract to a record suitable for permanent storage"""
        record = self.contract_record(contract)
        record.update({"negotiation_id": contract.mechanism_id})
        return record

    def on_contract_signed(self, contract: Contract) -> bool:
        """Called to add a contract to the existing set of contract after it is signed

        Args:

            contract: The contract to add

        Returns:

            True if everything went OK and False otherwise

        Remarks:

            - By default this function just adds the contract to the set of contracts maintaned by the world.
            - You should ALWAYS call this function when overriding it.

        """
        if not self.is_valid_contract(contract):
            # TODO check adding an edge of type dropped
            record = self._contract_record(contract)
            record["signed_at"] = self.current_step
            record["executed_at"] = -1
            record["breaches"] = ""
            record["nullified_at"] = -1
            record["dropped_at"] = self.current_step
            record["erred_at"] = -1
            self._saved_contracts[contract.id] = record
            self.__n_contracts_dropped += 1
            for p in contract.partners:
                self.contracts_dropped[p] += 1
            self.on_contract_processed(contract)
            return False

        self._add_edges(
            contract.partners[0],
            contract.partners,
            self._edges_contracts_signed,
            bi=True,
        )
        self.__n_contracts_signed += 1
        for p in contract.partners:
            self.contracts_signed[p] += 1
        try:
            self.unsigned_contracts[self.current_step].remove(contract)
        except KeyError:
            pass
        record = self._contract_record(contract)

        if self.save_signed_contracts:
            record["signed_at"] = self.current_step
            record["executed_at"] = -1
            record["breaches"] = ""
            record["nullified_at"] = -1
            record["erred_at"] = -1
            record["dropped_at"] = -1
            self._saved_contracts[contract.id] = record
        else:
            self._saved_contracts.pop(contract.id, None)
        return True

    def on_contract_cancelled(self, contract):
        """Called whenever a concluded contract is not signed (cancelled)

        Args:

            contract: The contract to add

        Remarks:

            - By default this function just adds the contract to the set of contracts maintaned by the world.
            - You should ALWAYS call this function when overriding it.

        """
        self._add_edges(
            contract.partners[0],
            contract.partners,
            self._edges_contracts_cancelled,
            bi=True,
        )
        record = self._contract_record(contract)
        record["signed_at"] = -1
        record["executed_at"] = -1
        record["breaches"] = ""
        record["nullified_at"] = -1
        record["dropped_at"] = -1
        record["erred_at"] = -1

        self._saved_contracts[contract.id] = record
        self.__n_contracts_cancelled += 1
        self.on_contract_processed(contract)

    def _process_unsigned(self):
        """Processes all concluded but unsigned contracts"""
        unsigned = self.unsigned_contracts.get(self.current_step, None)
        signed = []
        cancelled = []
        if unsigned:
            if self.batch_signing:
                agent_contracts = defaultdict(list)
                agent_signed = defaultdict(list)
                agent_cancelled = defaultdict(list)
                contract_signatures = defaultdict(int)
                contract_rejectors = defaultdict(list)
                for contract in unsigned:
                    for p in contract.partners:
                        if contract.signatures.get(p, None) is None:
                            agent_contracts[p].append(contract)
                        else:
                            contract_signatures[contract.id] += 1
                for agent_id, contracts in agent_contracts.items():
                    slist = self.call(
                        self.agents[agent_id],
                        self.agents[agent_id].sign_all_contracts,
                        contracts,
                    )
                    if self.time >= self.time_limit:
                        break
                    if slist is None:
                        slist = [False] * len(contracts)
                    elif isinstance(slist, str):
                        slist = [slist] * len(contracts)
                    elif isinstance(slist, dict):
                        slist = [slist.get(c.id, None) for c in contracts]
                    elif isinstance(slist, Iterable):
                        slist = list(slist)
                        missing = len(contracts) - len(slist)
                        if missing > 0:
                            slist += [None] * missing
                        elif missing < 0:
                            slist = slist[: len(contracts)]
                    for contract, signature in zip(contracts, slist):
                        if signature is not None:
                            contract_signatures[contract.id] += 1
                        else:
                            contract_rejectors[contract.id].append(agent_id)
                for contract in unsigned:
                    if contract_signatures[contract.id] == len(contract.partners):
                        contract.signatures = dict(
                            zip(contract.partners, contract.partners)
                        )
                        contract.signed_at = self.current_step
                        for partner in contract.partners:
                            agent_signed[partner].append(contract)
                        signed.append(contract)
                    else:
                        rejectors = contract_rejectors.get(contract.id, [])
                        for partner in contract.partners:
                            agent_cancelled[partner].append((contract, rejectors))
                        cancelled.append(contract)
                everyone = set(agent_signed.keys()).union(set(agent_cancelled.keys()))
                for agent_id in everyone:
                    cinfo = agent_cancelled[agent_id]
                    rejectors = [_[1] for _ in cinfo]
                    clist = [_[0] for _ in cinfo]
                    self.call(
                        self.agents[agent_id],
                        self.agents[agent_id].on_contracts_finalized,
                        agent_signed[agent_id],
                        clist,
                        rejectors,
                    )
                    if self.time >= self.time_limit:
                        break
            else:
                for contract in unsigned:
                    rejectors = self._sign_contract(contract)
                    if rejectors is not None and len(rejectors) == 0:
                        signed.append(contract)
                    else:
                        cancelled.append(contract)
            for contract in signed:
                self.on_contract_signed(contract)
            for contract in cancelled:
                self.on_contract_cancelled(contract)

    def _make_negotiation_record(self, negotiation: NegotiationInfo) -> dict[str, Any]:
        """Creates a record of the negotiation to be saved"""
        if negotiation is None:
            return {}
        mechanism = negotiation.mechanism
        if mechanism is None:
            return {}
        running, agreement = mechanism.state.running, mechanism.state.agreement
        record = {
            "id": mechanism.id,
            "partner_ids": [_.id for _ in negotiation.partners],
            "partners": [_.name for _ in negotiation.partners],
            "partner_types": [_.type_name for _ in negotiation.partners],
            "requested_at": negotiation.requested_at,
            "ended_at": self.current_step,
            "mechanism_type": mechanism.__class__.__name__,
            "issues": [str(issue) for issue in negotiation.issues],
            "final_status": "running"
            if running
            else "succeeded"
            if agreement is not None
            else "failed",
            "failed": agreement is None,
            "agreement": str(agreement),
            "group": negotiation.group,
            "caller": negotiation.caller,
        }
        if negotiation.annotation:
            record.update(to_flat_dict(negotiation.annotation))
        dd = mechanism.state.asdict()
        dd = {(k if k not in record.keys() else f"{k}_neg"): v for k, v in dd.items()}
        dd["history"] = [_.asdict() for _ in mechanism.history]
        if hasattr(mechanism, "negotiator_offers"):
            dd["offers"] = {
                n.owner.id
                if n.owner
                else n.name: [_ for _ in mechanism.negotiator_offers(n.id)]
                for n in mechanism.negotiators
            }
        record.update(dd)
        return record

    def _log_negotiation(self, negotiation: NegotiationInfo) -> None:
        if not self._log_negs:
            return
        mechanism = negotiation.mechanism
        if not mechanism:
            return
        negs_folder = str(Path(self._log_folder) / "negotiations")
        os.makedirs(negs_folder, exist_ok=True)
        record = self._make_negotiation_record(negotiation)
        if len(record) < 1:
            return
        add_records(str(Path(self._log_folder) / "negotiations.csv"), [record])
        data = pd.DataFrame([to_flat_dict(_) for _ in mechanism.history])
        data.to_csv(os.path.join(negs_folder, f"{mechanism.id}.csv"), index=False)

    @property
    def n_simulation_exceptions(self) -> dict[int, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions it and its negotiators have raised
        """
        result = defaultdict(int)
        for k, v in self.simulation_exceptions.items():
            result[k] += len(v)
        return result

    @property
    def n_contract_exceptions(self) -> dict[int, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions it and its negotiators have raised
        """
        result = defaultdict(int)
        for k, v in self.contract_exceptions.items():
            result[k] += len(v)
        return result

    @property
    def n_mechanism_exceptions(self) -> dict[int, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions it and its negotiators have raised
        """
        result = defaultdict(int)
        for k, v in self.mechanism_exceptions.items():
            result[k] += len(v)
        return result

    @property
    def n_total_simulation_exceptions(self) -> dict[int, int]:
        """
        Returns the total number of exceptions per step that are not directly raised by agents or their negotiators.

        Remarks:
            - This property sums the totals of `n_simulation_exceptions`, `n_contract_exceptions`, and `n_mechanism_exceptions`
        """
        result = defaultdict(int)
        for d in (
            self.n_mechanism_exceptions,
            self.n_contract_exceptions,
            self.n_simulation_exceptions,
        ):
            for k, v in d.items():
                result[k] += v
        return result

    @property
    def n_agent_exceptions(self) -> dict[str, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions it and its negotiators have raised
        """
        result = dict()
        for k, v in self.agent_exceptions.items():
            result[k] = len(v)
        return result

    @property
    def n_total_agent_exceptions(self) -> dict[str, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions it and its negotiators have raised
        """
        result: dict[str, int] = defaultdict(int)
        for k, v in self.agent_exceptions.items():
            result[k] += len(v)
        for k, v in self.negotiator_exceptions.items():
            result[k] += len(v)
        return result

    @property
    def n_negotiator_exceptions(self) -> dict[str, int]:
        """
        Returns a mapping from agent ID to the total number of exceptions its negotiators have raised
        """
        result = dict()
        for k, v in self.negotiator_exceptions.items():
            result[k] = len(v)
        return result

    def is_valid_agreement(
        self, negotiation: NegotiationInfo, agreement: Outcome, mechanism: Mechanism
    ) -> bool:
        """
        Confirms that the agreement is valid given the world rules.

        Args:
            negotiation: The `NegotiationInfo` that led to the agreement

            agreement: The agreement
            mechanism: The mechanism that led to the agreement

        Return:

            Returns True for valid agreements and False for invalid agreements

        Remarks:

            - This test is conducted before the agents are asked to sign the corresponding contract
            - Invalid agreements will be treated as never happened and agents will not be asked to sign it
        """
        return True

    def on_contract_concluded(self, contract: Contract, to_be_signed_at: int) -> None:
        """Called to add a contract to the existing set of unsigned contract after it is concluded

        Args:

            contract: The contract to add
            to_be_signed_at: The timestep at which the contract is to be signed

        Remarks:

            - By default this function just adds the contract to the set of contracts maintaned by the world.
            - You should ALWAYS call this function when overriding it.

        """
        self.__n_contracts_concluded += 1
        for p in contract.partners:
            self.contracts_concluded[p] += 1
        self._add_edges(
            contract.partners[0],
            contract.partners,
            self._edges_contracts_concluded,
            agreement=contract.agreement,
            bi=True,
        )
        self.unsigned_contracts[to_be_signed_at].add(contract)

    def _register_contract(
        self, mechanism, negotiation, to_be_signed_at
    ) -> Contract | None:
        partners = negotiation.partners
        if self.save_negotiations:
            _stats = self._make_negotiation_record(negotiation)
            self._saved_negotiations[mechanism.id] = _stats
        if mechanism.state.agreement is None or negotiation is None:
            return None
        for partner in partners:
            self.negs_succeeded[partner.id] += 1
        if not self.is_valid_agreement(
            negotiation, mechanism.state.agreement, mechanism
        ):
            return None
        agreement = mechanism.state.agreement
        agreement = outcome2dict(agreement, issues=[_.name for _ in mechanism.issues])
        signed_at = -1
        contract = Contract(
            partners=list(_.id for _ in partners),
            annotation=negotiation.annotation,
            issues=negotiation.issues,
            agreement=agreement,
            concluded_at=self.current_step,
            to_be_signed_at=to_be_signed_at,
            signed_at=signed_at,
            mechanism_state=mechanism.state,
            mechanism_id=mechanism.id,
        )
        self.on_contract_concluded(contract, to_be_signed_at)
        for partner in partners:
            self.call(
                partner,
                partner.on_negotiation_success_,
                contract=contract,
                mechanism=mechanism,
            )
            if self.time >= self.time_limit:
                break
        if self.batch_signing:
            if to_be_signed_at != self.current_step:
                sign_status = f"to be signed at {contract.to_be_signed_at}"
            else:
                sign_status = ""
        else:
            if to_be_signed_at == self.current_step:
                rejectors = self._sign_contract(contract)
                signed = rejectors is not None and len(rejectors) == 0
                if signed:
                    signed = self.on_contract_signed(contract)
                sign_status = (
                    "signed"
                    if signed
                    else f"cancelled by {rejectors if rejectors else 'being invalid!!'}"
                )
            else:
                sign_status = f"to be signed at {contract.to_be_signed_at}"
            # self.on_contract_processed(contract=contract)
        if negotiation.annotation is not None:
            annot_ = dict(
                zip(
                    negotiation.annotation.keys(),
                    (str(_) for _ in negotiation.annotation.values()),
                )
            )
        else:
            annot_ = ""
        self.logdebug(
            f"Contract [{sign_status}]: "
            f"{[_.name for _ in partners]}"
            f" > {str(mechanism.state.agreement)} on annotation {annot_}",
            Event(
                "negotiation-success",
                dict(mechanism=mechanism, contract=contract, partners=partners),
            ),
        )
        return contract

    def _register_failed_negotiation(self, mechanism, negotiation) -> None:
        partners = negotiation.partners
        mechanism_state = mechanism.state
        annotation = negotiation.annotation
        self._add_edges(
            partners[0],
            partners,
            self._edges_negotiations_failed,
            issues=mechanism.issues,
            bi=True,
        )
        for partner in partners:
            self.negs_failed[partner.id] += 1
            if mechanism_state.timedout:
                self.negs_timedout[partner.id] += 1
        if self.save_negotiations:
            _stats = self._make_negotiation_record(negotiation)
            self._saved_negotiations[mechanism.id] = _stats
        for partner in partners:
            self.call(
                partner,
                partner.on_negotiation_failure_,
                partners=[_.id for _ in partners],
                annotation=annotation,
                mechanism=mechanism,
                state=mechanism_state,
            )
            if self.time >= self.time_limit:
                break

        self.logdebug(
            f"Negotiation failure between {[_.name for _ in partners]}"
            f" on annotation {negotiation.annotation} ",
            Event("negotiation-failure", dict(mechanism=mechanism, partners=partners)),
        )

    def _tobe_signed_at(self, agreement: Outcome, force_immediate_signing=False) -> int:
        return (
            self.current_step
            if force_immediate_signing
            else self.current_step + self.default_signing_delay
        )

    def _step_a_mechanism(
        self,
        mechanism,
        force_immediate_signing,
        action: dict[str, Action | None] | None = None,
    ) -> tuple[Contract | None, bool]:
        """Steps a mechanism one step.

        Returns:

            The agreement or None and whether the negotiation is still running
        """
        contract = None
        try:
            result = mechanism.step(action)
        except Exception as e:
            result = mechanism.abort()
            if not self.ignore_negotiation_exceptions:
                raise e
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logerror(
                f"Mechanism exception: " f"{traceback.format_tb(exc_traceback)}",
                Event("entity-exception", dict(exception=e)),
            )
        finally:
            namap = dict()
            for neg in mechanism.negotiators:
                namap[neg.id] = neg.owner

            if mechanism.stats["times"]:
                for source, t in mechanism.stats["times"].items():
                    self.times[namap[source].id if namap[source] else "Unknown"] += t

            if mechanism.stats["exceptions"]:
                for source, exceptions in mechanism.stats["exceptions"].items():
                    self.negotiator_exceptions[
                        namap[source].id if namap[source] else "Unknown"
                    ].append(
                        list(zip(itertools.repeat(self._current_step), exceptions))
                    )

        agreement, is_running = result.agreement, not result.ended
        if agreement is not None or not is_running:
            negotiation = self._negotiations.get(mechanism.id, None)
            if self._debug:
                assert (
                    negotiation is not None
                ), f"{mecahanism.id} just finished but it is not in the set of running negotiations!!"
            if agreement is None:
                self._register_failed_negotiation(mechanism.nmi, negotiation)
            else:
                contract = self._register_contract(
                    mechanism.nmi,
                    negotiation,
                    self._tobe_signed_at(agreement, force_immediate_signing),
                )
            self._log_negotiation(negotiation)
            self._negotiations.pop(mechanism.id, None)
        return contract, is_running

    def _step_negotiations(
        self,
        mechanisms: list[Mechanism],
        n_steps: int | float | None,
        force_immediate_signing: bool,
        partners: list[list[Agent]],
        action: dict[str, dict[str, Action | None]] | None = None,
    ) -> tuple[list[Contract | None], list[bool], int, int, int, int]:
        """
        Runs all bending negotiations.

        Args:
            mechanisms: The mechanisms to step forward
            n_steps: The maximum number of steps to step each mechanism.
            force_immediate_signing: If true, all agreements are signed as contracts immediately upon agreement.
            partners: List of partners for each mechanism.
            action: Mapping of negotiator IDs to corresponding negotiation action (e.g. offer in SAO) for every mechanism.
                         Negotiators will be called upon to act only if no action is passed here.

        Remarks:
            - The actual number of steps executed is
              min(n_steps, self.negotiation_speed, mechanism.n_remaining_steps)
              with any None substituted with float('inf')

        """
        running = [_ is not None for _ in mechanisms]
        contracts: list[Contract | None] = [None] * len(mechanisms)
        indices = list(range(len(mechanisms)))
        n_steps_broken_, n_steps_success_ = 0, 0
        n_broken_, n_success_ = 0, 0
        current_step = 0
        if n_steps is None:
            n_steps = float("inf")
        if self.negotiation_speed is not None:
            n_steps = min(n_steps, self.negotiation_speed)

        while any(running):
            if self.shuffle_negotiations:
                random.shuffle(indices)
            for i in indices:
                if not running[i]:
                    continue
                if self.time >= self.time_limit:
                    break
                mechanism = mechanisms[i]
                contract, r = self._step_a_mechanism(
                    mechanism,
                    force_immediate_signing,
                    action=action.get(mechanism.id, None) if action else None,
                )
                contracts[i] = contract
                running[i] = r
                if not running[i]:
                    if contract is None:
                        n_broken_ += 1
                        n_steps_broken_ += mechanism.state.step + 1
                    else:
                        n_success_ += 1
                        n_steps_success_ += mechanism.state.step + 1
                    for _p in partners:
                        self._add_edges(
                            _p[0],
                            _p,
                            self._edges_negotiations_succeeded
                            if contract is not None
                            else self._edges_negotiations_failed,
                            issues=mechanism.issues,
                            bi=True,
                        )
            current_step += 1
            if current_step >= n_steps:
                break
            if self.time >= self.time_limit:
                break
        return (
            contracts,
            running,
            n_steps_broken_,
            n_steps_success_,
            n_broken_,
            n_success_,
        )

    def append_stats(self):
        if self._stats_file_name is not None:
            save_stats(
                self,
                log_dir=self._stats_dir_name,
                stats_file_name=self._stats_file_name,
            )

    def step(
        self,
        n_neg_steps: int | None = None,
        n_mechanisms: int | None = None,
        actions: dict[str, Any] | None = None,
        neg_actions: dict[str, dict[str, Action | None]] | None = None,
    ) -> bool:
        """
        A single simulation step.

        Args:
            n_mechanisms: Number of mechanisms to step (None for all)
            n_neg_steps: Number of steps for every mechanism (None to complete one simulation step)
            actions: Mapping of agent IDs to their actions. The agent will be asked to act only if this is not given
            neg_actions: Mapping of mechanism IDs to a negotiator action.
                         Negotiators will be called upon to act only if no action is passed here.
                         This is a dict with keys corresponding to mechanism IDs and values corresponding to a dict mapping
                         a negotiator (key) to its action (value)

        Remarks:

            - We have two modes of operation depending on `n_neg_steps`

              1. `n_neg_steps is None` will run a single complete simulation step every call
                  including all negotiations and everything before and after them.
              2. `n_neg_steps is an integer` will step the simulation so that the given number
                  of simulation steps are executed every call. The simulator will run operations
                  before and after negotiations appropriately.

            - We have two modes of operation depending on `n_mechanisms`

              1. `n_mechanisms is None` will step all negotiations according to `n_neg_steps`
              2. `n_mechanisms` is an integer and `n_neg_steps` will step this number of
                  mechanisms in parallel every call to step.

            - We have a total of four modes:

              1. `n_neg_steps` and `n_mechanisms` are both None: Each call to `step` corresponds
                  to one simulation step from start to end.
              2. `n_neg_steps` and `n_mechanisms` are both integers: Each call to `step` steps
                 `n_mechanisms` mechanisms by `n_neg_steps` steps.
              3. `n_neg_steps` is None and `n_mechanisms` is an integer: Each call to `step` runs
                 `n_mechanisms` according to `negotiation_speed`
              4. `n_neg_steps` is an integer and `n_mechanisms` is None: Each call to `step`
                 steps all mechanisms `n_neg_steps` steps.

            - Never mix calls with `n_neg_steps` equaling `None` and an integer.
            - Never call this method again on a world if it ever returned `False` on that world.
            - TODO Implement actions. Currently they are just ignored
        """
        if self.time >= self.time_limit:
            return False
        if self.current_step >= self.n_steps:
            return False
        cross_step_boundary = n_neg_steps is not None
        if self._debug:
            existing = {
                _.mechanism.id for _ in self._negotiations.values() if _ is not None
            }
            passed = set(neg_actions.keys()) if neg_actions else set()
            missing = passed.difference(existing)
            assert (
                not missing
            ), f"Mechanisms not found:\n{existing=} ({len(existing)})\n{passed=} ({len(passed)})\n{missing=} ({len(missing)})"

        #
        _n_registered_negotiations_before = len(self._negotiations)
        n_steps_broken, n_steps_success = 0, 0
        n_broken, n_success = 0, 0

        def _negotiate(n_steps_to_run: int | None = n_neg_steps) -> bool:
            """Runs all bending negotiations. Returns True if all negotiations are done"""
            if n_steps_to_run is not None and n_steps_to_run == 0:
                mechanisms = list(
                    (_.mechanism, _.partners)
                    for _ in self._negotiations.values()
                    if _ is not None
                )
                running = [
                    _[0] for _ in mechanisms if _ is not None and not _[0].state.ended
                ]
                return not running

            mechanisms = list(
                (_.mechanism, _.partners)
                for _ in self._negotiations.values()
                if _ is not None and _.mechanism.id not in self.__stepped_mechanisms
            )
            if n_mechanisms is not None and len(mechanisms) > n_mechanisms:
                mechanisms = mechanisms[:n_mechanisms]
            if not mechanisms:
                self.__stepped_mechanisms = set()
                mechanisms = list(
                    (_.mechanism, _.partners)
                    for _ in self._negotiations.values()
                    if _ is not None
                )
                if n_mechanisms is not None and len(mechanisms) > n_mechanisms:
                    mechanisms = mechanisms[:n_mechanisms]
            (
                _,
                _,
                n_steps_broken_,
                n_steps_success_,
                n_broken_,
                n_success_,
            ) = self._step_negotiations(
                [_[0] for _ in mechanisms],
                n_steps_to_run,
                False,
                [_[1] for _ in mechanisms],
                action=neg_actions,
            )
            self.__stepped_mechanisms = self.__stepped_mechanisms.union(
                {_[0].id for _ in mechanisms}
            )
            running = [
                _.mechanism.id
                for _ in self._negotiations.values()
                if _ is not None
                and _.mechanism is not None
                and not _.mechanism.state.ended
            ]

            # self._stats["n_registered_negotiations_before"].append(
            #     _n_registered_negotiations_before
            # )
            # self._stats["n_negotiation_rounds_successful"].append(n_steps_success)
            # self._stats["n_negotiation_rounds_failed"].append(n_steps_broken)
            # self._stats["n_negotiation_successful"].append(n_success)
            # self._stats["n_negotiation_failed"].append(n_broken)
            # self._stats["n_registered_negotiations_after"].append(
            #     len(self._negotiations)
            # )
            return not running

        if cross_step_boundary:
            if self.operations[self.__next_operation_index] != Operations.Negotiations:
                self.__stepped_mechanisms = set()
                if not self._step_to_negotiations(cross_step_boundary):
                    return False

            if _negotiate(n_neg_steps):
                self.__next_operation_index += 1
                # TODO correct this. Curently we just store whatever happens in the last negotiation step not for all negotiations.
                #
                n_steps_broken_ = 0
                n_steps_success_ = 0
                n_broken_ = 0
                n_success_ = 0
                if self.time < self.time_limit:
                    n_total_broken = n_broken + n_broken_
                    if n_total_broken > 0:
                        n_steps_broken = (
                            n_steps_broken * n_broken + n_steps_broken_ * n_broken_
                        ) / n_total_broken
                        n_broken = n_total_broken
                    n_total_success = n_success + n_success_
                    if n_total_success > 0:
                        n_steps_success = (
                            n_steps_success * n_success + n_steps_success_ * n_success_
                        ) / n_total_success
                        n_success = n_total_success
                self._stats["n_registered_negotiations_before"].append(
                    _n_registered_negotiations_before
                )
                self._stats["n_negotiation_rounds_successful"].append(n_steps_success)
                self._stats["n_negotiation_rounds_failed"].append(n_steps_broken)
                self._stats["n_negotiation_successful"].append(n_success)
                self._stats["n_negotiation_failed"].append(n_broken)
                self._stats["n_registered_negotiations_after"].append(
                    len(self._negotiations)
                )
                if self.__next_operation_index >= len(self.operations):
                    self.__next_operation_index = 0
                if not self._step_to_negotiations(cross_step_boundary):
                    return False
            return True
        if self._debug:
            assert self.__next_operation_index == 0
        if not self._step_to_negotiations(cross_step_boundary):
            return False
        self.__stepped_mechanisms = set()
        if self._debug:
            assert self.__next_operation_index != 0
        while self.__next_operation_index != 0:
            if not _negotiate(n_neg_steps):
                pass
                # print(
                #     "Some negotiations are still running but all should be completed by now"
                # )
            self.__next_operation_index += 1
            if self.__next_operation_index >= len(self.operations):
                self.__next_operation_index = 0
            if not self._step_to_negotiations(cross_step_boundary):
                return False
        return True

    def _pre_step(self) -> bool:
        self.__stepped_mechanisms = set()
        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        if self.time >= self.time_limit:
            return False
        if self.current_step >= self.n_steps:
            return False

        self.__stats_stage = 0
        self.__stage = 0
        self.__n_new_contract_executions = 0
        self.__n_new_breaches = 0
        self.__n_new_contract_errors = 0
        self.__n_new_contract_nullifications = 0
        self.__activity_level = 0
        self.__blevel = 0.0
        self._n_negs_per_agent_per_step = defaultdict(int)
        self._started = True
        if self.current_step == 0:
            self._sim_start = time.perf_counter()
            self._step_start = self._sim_start
            for priority in sorted(self._entities.keys()):
                for agent in self._entities[priority]:
                    self.call(agent, agent.init_)
                    if self.time >= self.time_limit:
                        return False
            # update monitors
            for monitor in self.stats_monitors:
                if self.safe_stats_monitoring:
                    __stats = copy.deepcopy(self.stats)
                else:
                    __stats = self.stats
                monitor.init(__stats, world_name=self.name)
            for monitor in self.world_monitors:
                monitor.init(self)
        else:
            self._step_start = time.perf_counter()
        # do checkpoint processing
        self.checkpoint_on_step_started()

        for agent in self.agents.values():
            self.call(agent, agent.on_simulation_step_started)
            if self.time >= self.time_limit:
                return False

        self.loginfo(
            f"{len(self._negotiations)} Negotiations/{len(self.agents)} Agents"
        )
        return True

    def _step_to_negotiations(self, cross_step_boundary: bool = True) -> bool:
        """Runs the operations before/after negotiations but not the negotiations themselves"""
        if self.__next_operation_index == 0:
            if not self._pre_step():
                return False

        # initialize stats
        # ----------------

        def _step_agents():
            # Step all entities in the world once:
            # ------------------------------------
            # note that entities are simulated in the partial-order specified by their priority value
            tasks: list[Entity] = []
            for priority in sorted(self._entities.keys()):
                tasks += [_ for _ in self._entities[priority]]

            for task in tasks:
                self.call(task, task.step_)
                if self.time >= self.time_limit:
                    break

        def _sign_contracts():
            self._process_unsigned()

        def _simulation_step():
            try:
                self.simulation_step(self.__stage)
                if self.time >= self.time_limit:
                    return
            except Exception as e:
                self.simulation_exceptions[self._current_step].append(exception2str())
                if not self.ignore_simulation_exceptions:
                    raise (e)
            self.__stage += 1

        def _execute_contracts():
            # execute contracts that are executable at this step
            # --------------------------------------------------
            n_new_breaches = self.__n_new_breaches
            n_new_contract_executions = self.__n_new_contract_executions
            n_new_contract_errors = self.__n_new_contract_errors
            n_new_contract_nullifications = self.__n_new_contract_nullifications
            activity_level = self.__activity_level
            blevel = self.__blevel
            current_contracts = [
                _ for _ in self.executable_contracts() if _.nullified_at < 0
            ]
            if len(current_contracts) > 0:
                # remove expired contracts
                executed = set()
                current_contracts = self.order_contracts_for_execution(
                    current_contracts
                )

                for contract in current_contracts:
                    if self.time >= self.time_limit:
                        break
                    if contract.signed_at < 0:
                        continue
                    try:
                        contract_breaches = self.start_contract_execution(contract)
                    except Exception as e:
                        for p in contract.partners:
                            self.contracts_erred[p] += 1
                        self.contract_exceptions[self._current_step].append(
                            exception2str()
                        )
                        contract.executed_at = self.current_step
                        self._saved_contracts[contract.id]["breaches"] = ""
                        self._saved_contracts[contract.id]["executed_at"] = -1
                        self._saved_contracts[contract.id]["dropped_at"] = -1
                        self._saved_contracts[contract.id]["nullified_at"] = -1
                        self._saved_contracts[contract.id][
                            "erred_at"
                        ] = self._current_step
                        self._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self._edges_contracts_erred,
                            bi=True,
                        )
                        n_new_contract_errors += 1
                        if not self.ignore_contract_execution_exceptions:
                            raise e
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        self.logerror(
                            f"Contract exception @{str(contract)}: "
                            f"{traceback.format_tb(exc_traceback)}",
                            Event(
                                "contract-exception",
                                dict(contract=contract, exception=e),
                            ),
                        )
                        continue
                    if contract_breaches is None:
                        for p in contract.partners:
                            self.contracts_nullified[p] += 1
                        self._saved_contracts[contract.id]["breaches"] = ""
                        self._saved_contracts[contract.id]["executed_at"] = -1
                        self._saved_contracts[contract.id]["dropped_at"] = -1
                        self._saved_contracts[contract.id][
                            "nullified_at"
                        ] = self._current_step
                        self._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self._edges_contracts_nullified,
                            bi=True,
                        )
                        self._saved_contracts[contract.id]["erred_at"] = -1
                        n_new_contract_nullifications += 1
                        self.loginfo(
                            f"Contract nullified: {str(contract)}",
                            Event("contract-nullified", dict(contract=contract)),
                        )
                    elif len(contract_breaches) < 1:
                        for p in contract.partners:
                            self.contracts_executed[p] += 1
                        self._saved_contracts[contract.id]["breaches"] = ""
                        self._saved_contracts[contract.id]["dropped_at"] = -1
                        self._saved_contracts[contract.id][
                            "executed_at"
                        ] = self._current_step
                        self._add_edges(
                            contract.partners[0],
                            contract.partners,
                            self._edges_contracts_executed,
                            bi=True,
                        )
                        self._saved_contracts[contract.id]["nullified_at"] = -1
                        self._saved_contracts[contract.id]["erred_at"] = -1
                        executed.add(contract)
                        n_new_contract_executions += 1
                        _size = self.contract_size(contract)
                        if _size is not None:
                            activity_level += _size
                        for partner in contract.partners:
                            self.call(
                                self.agents[partner],
                                self.agents[partner].on_contract_executed,
                                contract,
                            )
                            if self.time >= self.time_limit:
                                break
                    else:
                        for p in contract.partners:
                            self.contracts_breached[p] += 1
                        self._saved_contracts[contract.id]["executed_at"] = -1
                        self._saved_contracts[contract.id]["nullified_at"] = -1
                        self._saved_contracts[contract.id]["dropped_at"] = -1
                        self._saved_contracts[contract.id]["erred_at"] = -1
                        self._saved_contracts[contract.id]["breaches"] = "; ".join(
                            f"{_.perpetrator}:{_.type}({_.level})"
                            for _ in contract_breaches
                        )
                        breachers = {
                            (_.perpetrator, tuple(_.victims)) for _ in contract_breaches
                        }
                        for breacher, victims in breachers:
                            if isinstance(victims, str) or isinstance(victims, Agent):
                                victims = [victims]
                            self._add_edges(
                                breacher,
                                victims,
                                self._edges_contracts_breached,
                                bi=False,
                            )
                        for b in contract_breaches:
                            self._saved_breaches[b.id] = b.as_dict()
                            self.loginfo(
                                f"Breach of {str(contract)}: {str(b)} ",
                                Event(
                                    "contract-breached",
                                    dict(contract=contract, breach=b),
                                ),
                            )
                        resolution = self._process_breach(
                            contract, list(contract_breaches)
                        )
                        if resolution is None:
                            n_new_breaches += 1
                            blevel += sum(_.level for _ in contract_breaches)
                        else:
                            n_new_contract_executions += 1
                            self.loginfo(
                                f"Breach resolution cor {str(contract)}: {str(resolution)} ",
                                Event(
                                    "breach-resolved",
                                    dict(
                                        contract=contract,
                                        breaches=list(contract_breaches),
                                        resolution=resolution,
                                    ),
                                ),
                            )
                        self.complete_contract_execution(
                            contract, list(contract_breaches), resolution
                        )
                        self.loginfo(
                            f"Executed {str(contract)}",
                            Event("contract-executed", dict(contract=contract)),
                        )
                        for partner in contract.partners:
                            self.call(
                                self.agents[partner],
                                self.agents[partner].on_contract_breached,
                                contract,
                                list(contract_breaches),
                                resolution,
                            )
                            if self.time >= self.time_limit:
                                break
                    contract.executed_at = self.current_step
            dropped = self.get_dropped_contracts()
            self.delete_executed_contracts()  # note that all contracts even breached ones are to be deleted
            for c in dropped:
                self.loginfo(
                    f"Dropped {str(c)}",
                    Event("dropped-contract", dict(contract=c)),
                )
                self._saved_contracts[c.id]["dropped_at"] = self._current_step
                for p in c.partners:
                    self.contracts_dropped[p] += 1
            self.__n_contracts_dropped += len(dropped)

        def _stats_update():
            self.update_stats(self.__stats_stage)
            self.__stats_stage += 1

        operation_map = {
            Operations.AgentSteps: _step_agents,
            Operations.ContractExecution: _execute_contracts,
            Operations.ContractSigning: _sign_contracts,
            Operations.Negotiations: None,
            Operations.SimulationStep: _simulation_step,
            Operations.StatsUpdate: _stats_update,
        }

        for i, operation in enumerate(self.operations):
            if i < self.__next_operation_index:
                continue
            if operation == Operations.Negotiations:
                self.__next_operation_index = i
                break
            operation_map[operation]()
            if self.time >= self.time_limit:
                self.__next_operation_index = i
                return False
        else:
            self.__next_operation_index = 0
            # remove all negotiations that are completed
            # ------------------------------------------
            completed = list(
                k
                for k, _ in self._negotiations.items()
                if _ is not None and _.mechanism.ended
            )
            for key in completed:
                self._negotiations.pop(key, None)

            # update stats
            # ------------
            self._stats["n_contracts_executed"].append(self.__n_new_contract_executions)
            self._stats["n_contracts_erred"].append(self.__n_new_contract_errors)
            self._stats["n_contracts_nullified"].append(
                self.__n_new_contract_nullifications
            )
            self._stats["n_contracts_cancelled"].append(self.__n_contracts_cancelled)
            self._stats["n_contracts_dropped"].append(self.__n_contracts_dropped)
            self._stats["n_breaches"].append(self.__n_new_breaches)
            self._stats["breach_level"].append(self.__blevel)
            self._stats["n_contracts_signed"].append(self.__n_contracts_signed)
            self._stats["n_contracts_concluded"].append(self.__n_contracts_concluded)
            self._stats["n_negotiations"].append(self.__n_negotiations)
            self._stats["activity_level"].append(self.__activity_level)
            current_time = time.perf_counter() - self._step_start
            self._stats["step_time"].append(current_time)
            total = self._stats.get("total_time", [0.0])[-1]
            self._stats["total_time"].append(total + current_time)
            self.__n_negotiations = 0
            self.__n_contracts_signed = 0
            self.__n_contracts_concluded = 0
            self.__n_contracts_cancelled = 0
            self.__n_contracts_dropped = 0

            self.__n_new_contract_executions = 0
            self.__n_new_breaches = 0
            self.__n_new_contract_errors = 0
            self.__n_new_contract_nullifications = 0
            self.__activity_level = 0
            self.__blevel = 0.0

            self.append_stats()
            for agent in self.agents.values():
                self.call(agent, agent.on_simulation_step_ended)
                if self.time >= self.time_limit:
                    return False

            for monitor in self.stats_monitors:
                if self.safe_stats_monitoring:
                    __stats = copy.deepcopy(self.stats)
                else:
                    __stats = self.stats
                monitor.step(__stats, world_name=self.name)
            for monitor in self.world_monitors:
                monitor.step(self)

            self._current_step += 1
            self.frozen_time = self.time
            if cross_step_boundary:
                return self._step_to_negotiations()
        # always indicate that the simulation is to continue
        return True

    @property
    def total_time(self):
        """Returns total simulation time (till now) in mx"""
        return self._stats.get("total_time", [0.0])[-1]

    @property
    def saved_breaches(self) -> list[dict[str, Any]]:
        return list(self._saved_breaches.values())

    @property
    def resolved_breaches(self) -> list[dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if _["resolved"])

    @property
    def unresolved_breaches(self) -> list[dict[str, Any]]:
        return list(_ for _ in self._saved_breaches.values() if not _["resolved"])

    def run(self):
        """Runs the simulation until it ends"""
        self._start_time = time.perf_counter()
        for _ in range(self.n_steps):
            if self.time >= self.time_limit:
                break
            if not self.step():
                break

    def run_with_progress(self, callback: Callable[[int], None] | None = None) -> None:
        """Runs the simulation showing progress, with optional callback"""
        from rich.progress import track

        self._start_time = time.perf_counter()
        for _ in track(range(self.n_steps)):
            if self.time >= self.time_limit:
                break
            if not self.step():
                break

    def register(self, x: Entity, simulation_priority: int = 0):
        """
        Registers an entity in the world so it can be looked up by name. Should not be called directly

        Args:
            x: The entity to be registered
            simulation_priority: The simulation periority. Entities with lower periorities will be stepped first during

        Returns:

        """
        # super().register(x) # If we inherit from session, we can do that but it is not needed as we do not do string
        # based resolution now
        if hasattr(x, "_world"):
            x._world = self
        if hasattr(x, "step_"):
            self._entities[simulation_priority].add(x)

    def register_stats_monitor(self, m: StatsMonitor):
        self.stats_monitors.add(m)

    def unregister_stats_monitor(self, m: StatsMonitor):
        self.stats_monitors.remove(m)

    def register_world_monitor(self, m: WorldMonitor):
        self.world_monitors.add(m)

    def unregister_world_monitor(self, m: WorldMonitor):
        self.world_monitors.remove(m)

    def join(self, x: Agent, simulation_priority: int = 0, **kwargs):
        """Add an agent to the world.

        Args:
            x: The agent to be registered
            simulation_priority: The simulation priority. Entities with lower pprioritieswill be stepped first during
            kwargs: Any key-value pairs specifying attributes of the agent. NegMAS internally uses the attribute 'color'
                    when drawing the agent in `draw`

        Returns:

        """
        self.register(x, simulation_priority=simulation_priority)
        self.agents[x.id] = x
        self.attribs[x.id] = kwargs
        x.awi = self.awi_type(self, x)
        if self._started and not x.initialized:
            self.call(x, x.init_)
        self.loginfo(f"{x.name} joined", Event("agent-joined", dict(agent=x)))

    def _combine_edges(
        self,
        beg: int,
        end: int,
        target: dict[int, dict[tuple[Agent, Agent], list[dict[str, Any]]]],
    ):
        """Combines edges for the given steps [beg, end)"""
        result = deflistdict()

        def add_dicts(d1, d2):
            d3 = deflistdict()
            for k, v in d1.items():
                d3[k] = d1[k] + d2[k]
            for k, v in d2.items():
                if k not in d3.keys():
                    d3[k] = d2[k]
            return d3

        for i in range(beg, end):
            result = add_dicts(result, target[i])
        return result

    def _get_edges(
        self,
        target: dict[int, dict[tuple[Agent, Agent], list[dict[str, Any]]]],
        step: int,
    ) -> list[tuple[Agent, Agent, int]]:
        """Get the edges for the given step"""
        return [(*k, {"weight": len(v)}) for k, v in target[step].items() if len(v) > 0]

    def _register_negotiation(
        self,
        mechanism_name,
        mechanism_params,
        roles,
        caller,
        partners,
        annotation,
        issues,
        req_id,
        run_to_completion=False,
        may_run_immediately=True,
        group: str | None = None,
    ) -> tuple[NegotiationInfo | None, Contract | None, Mechanism | None]:
        """Registers a negotiation and returns the negotiation info"""
        if self._n_negs_per_agent_per_step[caller.id] >= self.neg_quota_step:
            return None, None, None
        if self._n_negs_per_agent[caller.id] >= self.neg_quota_simulation:
            return None, None, None
        self.neg_requests_sent[caller.id] += 1
        for partner in partners:
            self.neg_requests_received[partner.id] += 1
        n_outcomes_ = CartesianOutcomeSpace(issues).cardinality
        if n_outcomes_ < 1:
            self.logwarning(
                f"A negotiation with no outcomes is requested by {caller.name}",
                event=Event(
                    "zero-outcomes-negotiation",
                    dict(caller=caller, partners=partners, annotation=annotation),
                ),
            )
            return None, None, None
        factory = MechanismFactory(
            world=self,
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            issues=issues,
            req_id=req_id,
            caller=caller,
            partners=partners,
            roles=roles,
            annotation=annotation,
            group=group,
            neg_n_steps=self.neg_n_steps,
            neg_time_limit=self.neg_time_limit,
            neg_step_time_limit=self.neg_step_time_limit,
            log_ufuns_file=str(Path(self._log_folder) / "ufuns.csv")
            if self._log_ufuns
            else None,
        )
        neg = factory.init()
        if neg is None:
            self._add_edges(
                caller, partners, self._edges_negotiations_rejected, issues=issues
            )
            return None, None, None
        if neg.mechanism is None:
            self._add_edges(
                caller, partners, self._edges_negotiations_rejected, issues=issues
            )
            return neg, None, None
        self.__n_negotiations += 1
        self._n_negs_per_agent_per_step[caller.id] += 1
        self._n_negs_per_agent[caller.id] += 1
        self._add_edges(
            caller, partners, self._edges_negotiations_started, issues=issues
        )
        # if not run_to_completion:
        self._negotiations[neg.mechanism.id] = neg
        self.negs_initiated[caller.id] += 1
        for partner in partners:
            self.negs_registered[partner.id] += 1
        if run_to_completion:
            running, contract = True, None
            while running:
                contract, running = self._step_a_mechanism(neg.mechanism, True)

            self._add_edges(
                caller,
                partners,
                self._edges_negotiations_succeeded
                if contract is not None
                else self._edges_negotiations_failed,
                issues=issues,
            )
            return None, contract, neg.mechanism
        if may_run_immediately and self.immediate_negotiations:
            running = True
            for _ in range(self.negotiation_speed):
                contract, running = self._step_a_mechanism(neg.mechanism, False)
                if not running:
                    self._add_edges(
                        caller,
                        partners,
                        self._edges_negotiations_succeeded
                        if contract is not None
                        else self._edges_negotiations_failed,
                        issues=issues,
                    )
                    return None, contract, neg.mechanism
        # self.loginfo(
        #    f'{caller.id} request was accepted')
        return neg, None, None

    def _unregister_negotiation(self, neg: MechanismFactory) -> None:
        if neg is None or neg.mechanism is None:
            return
        del self._negotiations[neg.mechanism.id]

    def request_negotiation_about(
        self,
        req_id: str,
        caller: Agent,
        issues: list[Issue],
        partners: list[Agent | str],
        roles: list[str] | None = None,
        annotation: dict[str, Any] | None = None,
        mechanism_name: str | None = None,
        mechanism_params: dict[str, Any] | None = None,
        group: str | None = None,
    ) -> NegotiationInfo:
        """
        Requests to start a negotiation with some other agents

        Args:
            req_id: An ID For the request that is unique to the caller
            caller: The agent requesting the negotiation
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            group: An identifier for the group to which the negotiation belongs. This is not not used by the system.

        Returns:

            None. The caller will be informed by a callback function `on_neg_request_accepted` or
            `on_neg_request_rejected` about the status of the negotiation.

        """
        if roles is None:
            if isinstance(partners, str) or isinstance(partners, Agent):
                partners = [partners]
            if (
                len(partners) == 1
                and isinstance(partners[0], str)
                and partners[0] != caller.id
            ):
                partners = [caller.id, partners[0]]
            if (
                len(partners) == 1
                and isinstance(partners[0], Agent)
                and partners[0] != caller
            ):
                partners = [caller, partners[0]]
        self.loginfo(
            f"{caller.name} requested negotiation "
            + (
                f"using {mechanism_name}[{mechanism_params}] "
                if mechanism_name is not None or mechanism_params is not None
                else ""
            )
            + f"with {[_.name for _ in partners]} (ID {req_id})",
            Event(
                "negotiation-request",
                dict(
                    caller=caller,
                    partners=partners,
                    issues=issues,
                    mechanism_name=mechanism_name,
                    annotation=annotation,
                    req_id=req_id,
                ),
            ),
        )
        neg, *_ = self._register_negotiation(
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            roles=roles,
            caller=caller,
            partners=partners,
            annotation=annotation,
            group=group,
            issues=issues,
            req_id=req_id,
            run_to_completion=False,
        )
        success = neg is not None and neg.mechanism is not None
        self._add_edges(
            caller,
            partners,
            self._edges_negotiation_requests_accepted
            if success
            else self._edges_negotiation_requests_rejected,
            issues=issues,
        )

        return neg

    def run_negotiation(
        self,
        caller: Agent,
        issues: list[Issue],
        partners: list[str | Agent],
        negotiator: Negotiator,
        preferences: Preferences | None = None,
        caller_role: str | None = None,
        roles: list[str] | None = None,
        annotation: dict[str, Any] | None = None,
        mechanism_name: str | None = None,
        mechanism_params: dict[str, Any] | None = None,
    ) -> tuple[Contract | None, NegotiatorMechanismInterface | None]:
        """
        Runs a negotiation until completion

        Args:
            caller: The agent requesting the negotiation
            partners: A list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            negotiator: The negotiator to be used in the negotiation
            preferences: The utility function. Only needed if the negotiator does not already know it
            caller_role: The role of the caller in the negotiation
            issues: Negotiation issues
            annotation: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_name: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object

        Returns:

            A Tuple of a contract and the nmi of the mechanism used to get it in case of success. None otherwise

        """
        if roles is None:
            if isinstance(partners, str) or isinstance(partners, Agent):
                partners = [partners]
            if (
                len(partners) == 1
                and isinstance(partners[0], str)
                and partners[0] != caller.id
            ):
                partners = [caller.id, partners[0]]
            if (
                len(partners) == 1
                and isinstance(partners[0], Agent)
                and partners[0] != caller
            ):
                partners = [caller, partners[0]]
        partners = [self.agents[_] if isinstance(_, str) else _ for _ in partners]
        self.loginfo(
            f"{caller.name} requested immediate negotiation "
            f"{mechanism_name}[{mechanism_params}] with {[_.name for _ in partners]}",
            Event(
                "negotiation-request-immediate",
                dict(
                    caller=caller,
                    partners=partners,
                    issues=issues,
                    mechanism_name=mechanism_name,
                    annotation=annotation,
                ),
            ),
        )
        req_id = caller.create_negotiation_request(
            issues=issues,
            partners=partners,
            annotation=annotation,
            negotiator=negotiator,
            extra={},
        )
        neg, contract, mechanism = self._register_negotiation(
            mechanism_name=mechanism_name,
            mechanism_params=mechanism_params,
            roles=roles,
            caller=caller,
            partners=partners,
            annotation=annotation,
            issues=issues,
            req_id=req_id,
            run_to_completion=True,
        )
        if contract is not None:
            return contract, mechanism.nmi
        if neg and neg.mechanism:
            mechanism = neg.mechanism
            if negotiator is not None:
                mechanism.add(negotiator, preferences=preferences, role=caller_role)
            mechanism.run()
            if mechanism.agreement is None:
                contract = None
                self._register_failed_negotiation(
                    mechanism=mechanism.nmi, negotiation=neg
                )
            else:
                contract = self._register_contract(
                    mechanism.nmi, neg, self._tobe_signed_at(mechanism.agreement, True)
                )
            return contract, mechanism.nmi
        return None, None

    def run_negotiations(
        self,
        caller: Agent,
        issues: list[Issue] | list[list[Issue]],
        partners: list[list[str | Agent]],
        negotiators: list[Negotiator],
        preferences: list[Preferences] | None = None,
        caller_roles: list[str] | None = None,
        roles: list[list[str] | None] | None = None,
        annotations: list[dict[str, Any] | None] | None = None,
        mechanism_names: str | list[str] | None = None,
        mechanism_params: dict[str, Any] | list[dict[str, Any]] | None = None,
        all_or_none: bool = False,
    ) -> list[tuple[Contract, NegotiatorMechanismInterface]]:
        """
        Requests to run a set of negotiations simultaneously. Returns after all negotiations are run to completion

        Args:
            caller: The agent requesting the negotiation
            partners: A list of list of partners to participate in the negotiation.
                      Note that the caller itself may not be in this list which
                      makes it possible for an agent to request a negotaition
                      that it does not participate in. If that is not to be
                      allowed in some world, override this method and explicitly
                      check for these kinds of negotiations and return False.
                      If partners[i] is passed as a single string/`Agent` or as a list
                      containing a single string/`Agent`, then he caller will be added
                      at the beginning of the list. This will only be done if
                      `roles` was passed as None.
            issues: Negotiation issues
            negotiators: The negotiator to be used in the negotiation
            ufuns: The utility function. Only needed if the negotiator does not already know it
            caller_roles: The role of the caller in the negotiation
            annotations: Extra information to be passed to the `partners` when asking them to join the negotiation
            partners: A list of partners to participate in the negotiation
            roles: The roles of different partners. If None then each role for each partner will be None
            mechanism_names: Name of the mechanism to use. It must be one of the mechanism_names that are supported by the
            `World` or None which means that the `World` should select the mechanism. If None, then `roles` and `my_role`
            must also be None
            mechanism_params: A dict of parameters used to initialize the mechanism object
            all_of_none: If True, ALL partners must agree to negotiate to go through.

        Returns:

             A list of tuples each with two values: contract (None for failure) and nmi (The mechanism info [None if
             the partner refused the negotiation])

        """
        group = unique_name(base="NG")
        partners = [
            [self.agents[_] if isinstance(_, str) else _ for _ in p] for p in partners
        ]
        n_negs = len(partners)
        if isinstance(issues[0], Issue):
            issues = [issues] * n_negs
        if roles is None or not (
            isinstance(roles, list) and isinstance(roles[0], list)
        ):
            roles = [roles] * n_negs
        if annotations is None or isinstance(annotations, dict):
            annotations = [annotations] * n_negs
        if mechanism_names is None or isinstance(mechanism_names, str):
            mechanism_names = [mechanism_names] * n_negs
        if mechanism_params is None or isinstance(mechanism_params, dict):
            mechanism_params = [mechanism_params] * n_negs
        if caller_roles is None or isinstance(caller_roles, str):
            caller_roles = [caller_roles] * n_negs
        if negotiators is None or isinstance(negotiators, Negotiator):
            raise ValueError(f"Must pass all negotiators for run_negotiations")
        if preferences is None or isinstance(preferences, Preferences):
            preferences = [preferences] * n_negs

        self.loginfo(
            f"{caller.name} requested {n_negs} immediate negotiation "
            f"{mechanism_names}[{mechanism_params}] between {[[_.name for _ in p] for p in partners]}",
            Event(
                "negotiation-request",
                dict(
                    caller=caller,
                    partners=partners,
                    issues=issues,
                    mechanism_name=mechanism_names,
                    all_or_none=all_or_none,
                    annotations=annotations,
                ),
            ),
        )
        negs = []
        for (
            issue,
            partner,
            role,
            annotation,
            mech_name,
            mech_param,
            negotiator_,
        ) in zip(
            issues,
            partners,
            roles,
            annotations,
            mechanism_names,
            mechanism_params,
            negotiators,
        ):
            if role is None:
                if isinstance(partner, str) or isinstance(partner, Agent):
                    partner = [partner]
            if (
                len(partner) == 1
                and isinstance(partner[0], str)
                and partner[0] != caller.id
            ):
                partner = [caller.id, partners[0]]
            if (
                len(partner) == 1
                and isinstance(partner[0], Agent)
                and partner[0] != caller
            ):
                partner = [caller, partner[0]]
            req_id = caller.create_negotiation_request(
                issues=issue,
                partners=partner,
                annotation=annotation,
                negotiator=negotiator_,
                extra={},
            )
            neg, *_ = self._register_negotiation(
                mechanism_name=mech_name,
                mechanism_params=mech_param,
                roles=role,
                caller=caller,
                partners=partner,
                group=group,
                annotation=annotation,
                issues=issue,
                req_id=req_id,
                run_to_completion=False,
                may_run_immediately=False,
            )
            # neg.partners.append(caller)
            if neg is None and all_or_none:
                for _n in negs:
                    self._unregister_negotiation(_n)
                return []
            negs.append(neg)
        if all(_ is None for _ in negs):
            return []
        completed = [False] * n_negs
        contracts = [None] * n_negs
        amis = [None] * n_negs
        for i, (neg, crole, ufun, negotiator) in enumerate(
            zip(negs, caller_roles, preferences, negotiators)
        ):
            completed[i] = neg is None or (neg.mechanism is None) or negotiator is None
            if completed[i]:
                continue
            mechanism = neg.mechanism
            mechanism.add(negotiator, ufun=ufun, role=crole)

        locs = [i for i in range(n_negs) if not completed[i]]
        cs, rs, _, _, _, _ = self._step_negotiations(
            [negs[i].mechanism for i in locs],
            float("inf"),
            True,
            [negs[i].partners for i in locs],
        )
        for i, loc in enumerate(locs):
            contracts[loc] = cs[i]
            completed[loc] = not rs[i]
            amis[i] = negs[i].mechanism.nmi
        return list(zip(contracts, amis))

    def _log_header(self):
        if self.time is None:
            return f"{self.name} (not started)"
        return f"{self.current_step}/{self.n_steps} [{self.relative_time:0.2%}]"

    def ignore_contract(self, contract, as_dropped=False):
        """
        Ignores the contract as if it was never agreed upon or as if was dropped

        Args:

            contract: The contract to ignore
            as_dropped: If true, the contract is treated as a dropped invalid
                        contract, otherwise it is treated as if it never
                        happened.

        """
        if as_dropped:
            if contract.agreement is not None:
                self.__n_contracts_dropped += 1
                for p in contract.partners:
                    self.contracts_dropped[p] += 1
            if contract.id in self._saved_contracts.keys():
                self._saved_contracts[contract.id]["dropped_at"] = self.current_step
        else:
            if contract.agreement is not None:
                self.__n_contracts_concluded -= 1
            if contract.id in self._saved_contracts.keys():
                if self._saved_contracts[contract.id]["signed_at"] >= 0:
                    self.__n_contracts_signed -= 1
                    for p in contract.partners:
                        self.contracts_signed[p] -= 1
                del self._saved_contracts[contract.id]
        for p in contract.partners:
            self.contracts_dropped[p] += 1
        self.on_contract_processed(contract)

    @property
    def saved_contracts(self) -> list[dict[str, Any]]:
        return list(self._saved_contracts.values())

    @property
    def executed_contracts(self) -> list[dict[str, Any]]:
        return list(
            _ for _ in self._saved_contracts.values() if _.get("executed_at", -1) >= 0
        )

    @property
    def signed_contracts(self) -> list[dict[str, Any]]:
        return list(
            _ for _ in self._saved_contracts.values() if _.get("signed_at", -1) >= 0
        )

    @property
    def nullified_contracts(self) -> list[dict[str, Any]]:
        return list(
            _ for _ in self._saved_contracts.values() if _.get("nullified_at", -1) >= 0
        )

    @property
    def erred_contracts(self) -> list[dict[str, Any]]:
        return list(
            _ for _ in self._saved_contracts.values() if _.get("erred_at", -1) >= 0
        )

    @property
    def cancelled_contracts(self) -> list[dict[str, Any]]:
        return list(
            _ for _ in self._saved_contracts.values() if not _.get("signed_at", -1) < 0
        )

    def save_config(self, file_name: str):
        """
        Saves the config of the world as a yaml file

        Args:
            file_name: Name of file to save the config to

        Returns:

        """
        with open(file_name, "w") as file:
            yaml.safe_dump(self.__dict__, file)

    def _process_breach(
        self, contract: Contract, breaches: list[Breach], force_immediate_signing=True
    ) -> Contract | None:
        new_contract = None
        # calculate total breach level
        total_breach_levels = defaultdict(int)
        for breach in breaches:
            total_breach_levels[breach.perpetrator] += breach.level

        if self.breach_processing == BreachProcessing.VICTIM_THEN_PERPETRATOR:
            # give agents the chance to set renegotiation agenda in ascending order of their total breach levels
            for agent_name, _ in sorted(
                zip(total_breach_levels.keys(), total_breach_levels.values()),
                key=lambda x: x[1],
            ):
                agent = self.agents[agent_name]
                agenda = agent.set_renegotiation_agenda(
                    contract=contract, breaches=breaches
                )
                if agenda is None:
                    continue
                negotiators = []
                for partner in contract.partners:
                    negotiator = self.call(
                        self.agents[partner],
                        self.agents[partner].respond_to_renegotiation_request,
                        contract=contract,
                        breaches=breaches,
                        agenda=agenda,
                    )
                    if self.time >= self.time_limit:
                        negotiator = None
                    if negotiator is None:
                        break
                    negotiators.append(negotiator)
                else:
                    # everyone accepted this renegotiation
                    results = self.run_negotiation(
                        caller=agent,
                        issues=agenda.issues,
                        partners=[self.agents[_] for _ in contract.partners],
                    )
                    if results is not None:
                        new_contract, mechanism = results
                        self._register_contract(
                            mechanism=mechanism,
                            negotiation=None,
                            to_be_signed_at=self._tobe_signed_at(
                                mechanism.agreement, force_immediate_signing
                            ),
                        )
                        break
        elif self.breach_processing == BreachProcessing.META_NEGOTIATION:
            raise NotImplementedError(
                "Meta negotiation is not yet implemented. Agents should negotiate about the "
                "agend then a negotiation should be conducted as usual"
            )

        if new_contract is not None:
            for breach in breaches:
                if self.save_resolved_breaches:
                    self._saved_breaches[breach.id]["resolved"] = True
                else:
                    self._saved_breaches.pop(breach.id, None)
            return new_contract
        for breach in breaches:
            if self.save_unresolved_breaches:
                self._saved_breaches[breach.id]["resolved"] = False
            else:
                self._saved_breaches.pop(breach.id, None)
            self._register_breach(breach)
        return None
        # todo add _get_signing_delay(contract) and implement it in SCML2019

    if nx:

        def graph(
            self,
            steps: tuple[int, int] | int | None = None,
            what: Collection[str] = EDGE_TYPES,
            who: Callable[[Agent], bool] | None = None,
            together: bool = True,
        ) -> nx.Graph | list[nx.Graph]:
            """
            Generates a graph showing some aspect of the simulation

            Args:
                steps: The step/steps to generate the graphs for. If a tuple is given all edges within the given range
                       (inclusive beginning, exclusive end) will be accumulated
                what: The edges to have on the graph. Options are: negotiations, concluded, signed, executed
                who: Either a callable that receives an agent and returns True if it is to be shown or None for all
                together: IF specified all edge types are put in the same graph.

            Returns:
                A networkx graph representing the world if together==True else a list of graphs one for each item in what

            """
            if steps is None:
                steps = self.current_step
            if isinstance(steps, int):
                steps = [steps, steps + 1]
            steps = tuple(min(self.n_steps, max(0, _)) for _ in steps)
            if who is None:
                who = lambda x: True
            agents = [_.id for _ in self.agents.values() if who(_)]
            if together:
                g = nx.MultiDiGraph()
                g.add_nodes_from(agents)
                graphs = [g] * len(what)
            else:
                graphs = [nx.DiGraph() for _ in what]
                for g in graphs:
                    g.add_nodes_from(agents)
            max_step = max(steps) - 1
            for g, edge_type in zip(graphs, what):
                edge_info = getattr(self, f"_edges_{edge_type.replace('-', '_')}")
                edge_info = {max_step: self._combine_edges(*steps, edge_info)}
                color = EDGE_COLORS[edge_type]
                edgelist = self._get_edges(edge_info, max_step)
                for e in edgelist:
                    e[2]["color"] = color
                g.add_edges_from(edgelist)
            return graphs[0] if together else graphs

        def draw(
            self,
            steps: tuple[int, int] | int | None = None,
            what: Collection[str] = DEFAULT_EDGE_TYPES,
            who: Callable[[Agent], bool] | None = None,
            where: Callable[[Agent], int | tuple[float, float]] | None = None,
            together: bool = True,
            axs: Collection[Axes] | None = None,
            ncols: int = 4,
            figsize: tuple[int, int] = (15, 15),
            show_node_labels=True,
            show_edge_labels=True,
            **kwargs,
        ) -> tuple[Axes, nx.Graph] | tuple[Axes, list[nx.Graph]]:
            """
            Generates a graph showing some aspect of the simulation

            Args:
                steps: The step/steps to generate the graphs for. If a tuple is given all edges within the given range
                       (inclusive beginning, exclusive end) will be accomulated
                what: The edges to have on the graph. Options are: negotiations, concluded, signed, executed
                who: Either a callable that receives an agent and returns True if it is to be shown or None for all
                where: A callable that returns for each agent the position it showed by drawn at either as an integer
                       specifying the column in which to draw the column or a tuple of two floats specifying the position
                       within the drawing area of the agent. If None, the default Networkx layout will be used.
                together: IF specified all edge types are put in the same graph.
                axs: The axes used for drawing. If together is true, it should be a single `Axes` object otherwise it should
                     be a list of `Axes` objects with the same length as what.
                show_node_labels: show node labels!
                show_edge_labels: show edge labels!
                kwargs: passed to networx.draw_networkx

            Returns:
                A networkx graph representing the world if together==True else a list of graphs one for each item in what

            """

            import matplotlib.pyplot as plt

            if not self.construct_graphs:
                self.logwarning(
                    "Asked to draw a world simulation without enabling `construct_graphs`. Will be ignored"
                )
                return [None, None]
            if steps is None:
                steps = self.current_step
            if isinstance(steps, int):
                steps = [steps, steps + 1]
            steps = tuple(min(self.n_steps, max(0, _)) for _ in steps)
            if who is None:
                who = lambda x: True
            if together:
                titles = [""]
            else:
                titles = what
            if axs is None:
                if together:
                    fig, axs = plt.subplots()
                else:
                    nrows = int(math.ceil(len(what) / ncols))
                    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
                    axs = axs.flatten().tolist()
            if together:
                axs = [axs]
            graphs = self.graph(steps, what, who, together)
            graph = graphs[0] if not together else graphs
            graphs = [graphs] if together else graphs
            if where is None:
                pos = nx.spring_layout(graph, iterations=200)
            else:
                pos = [where(a) for a in graph.nodes]
                if not isinstance(pos[0], tuple):
                    deltax = 5
                    deltay = 5
                    cols = defaultdict(list)
                    for agent, p in zip(graph.nodes, pos):
                        cols[p].append(agent)
                    pos = dict()
                    for c, ros in cols.items():
                        for r, agent in enumerate(ros):
                            pos[agent] = ((1 + c) * deltay, r * deltax)
                else:
                    pos = dict(zip(graph.nodes, pos))
            if together:
                g = graph
                nx.draw_networkx_nodes(g, pos, ax=axs[0])
                edges = [_ for _ in g.edges]
                if len(edges) > 0:
                    info = [_ for _ in g.edges.data("color")]
                    colors = [_[2] for _ in info]
                    edges = [(_[0], _[1]) for _ in info]
                    clist = list(set(colors))
                    edgelists = [list() for _ in range(len(clist))]
                    for c, lst in zip(clist, edgelists):
                        for i, clr in enumerate(colors):
                            if clr == c:
                                lst.append(edges[i])
                    for lst, clr in zip(edgelists, clist):
                        nx.draw_networkx_edges(
                            g, pos, edgelist=g.edges, edge_color=clr, ax=axs[0]
                        )
                    if show_edge_labels:
                        info = [_ for _ in g.edges.data("weight")]
                        weights = [str(_[2]) for _ in info if _[2] > 1]
                        edges = [(_[0], _[1]) for _ in info if _[2] > 1]
                        nx.draw_networkx_edge_labels(
                            g, pos, dict(zip(edges, weights)), ax=axs[0]
                        )
                if show_node_labels:
                    nx.draw_networkx_labels(
                        g, pos, dict(zip(g.nodes, g.nodes)), ax=axs[0]
                    )
            else:
                for g, ax, title in zip(graphs, axs, titles):
                    nx.draw_networkx_nodes(g, pos, ax=ax)
                    nx.draw_networkx_edges(
                        g, pos, edgelist=g.edges, edge_color=EDGE_COLORS[title], ax=ax
                    )
                    if show_edge_labels:
                        info = [_ for _ in g.edges.data("weight")]
                        weights = [str(_[2]) for _ in info if _[2] > 1]
                        edges = [(_[0], _[1]) for _ in info if _[2] > 1]
                        nx.draw_networkx_edge_labels(
                            g, pos, dict(zip(edges, weights)), ax=ax
                        )
                    if show_node_labels:
                        nx.draw_networkx_labels(
                            g, pos, dict(zip(g.nodes, g.nodes)), ax=ax
                        )
                    ax.set_ylabel(title)
            total_time = time.perf_counter() - self._sim_start
            step = max(steps)
            remaining = (self.n_steps - step - 1) * total_time / (step + 1)
            title = (
                f"Step: {step + 1}/{self.n_steps} [{humanize_time(total_time)} rem "
                f"{humanize_time(remaining)}] {total_time / (remaining + total_time):04.2%}"
            )
            if together:
                axs[0].set_title(title)
            else:
                f = plt.gcf()
                f.suptitle(title)
            return (axs[0], graph) if together else (axs, graphs)

    def save_gif(
        self,
        path: str | Path | None = None,
        what: Collection[str] = EDGE_TYPES,
        who: Callable[[Agent], bool] | None = None,
        together: bool = True,
        draw_every: int = 1,
        fps: int = 5,
    ) -> None:
        try:
            import gif

            if path is None and self.log_folder is not None:
                path = Path(self.log_folder) / (self.name + ".gif")

            # define the animation function. Simply draw the world
            @gif.frame
            def plot_frame(s):
                self.draw(
                    steps=(s - draw_every, s),
                    what=what,
                    who=who,
                    together=together,
                    ncols=3,
                    figsize=(20, 20),
                )

            # create frames
            frames = []
            for s in range(draw_every, self.n_steps):
                if s % draw_every != 0:
                    continue
                frames.append(plot_frame(s))
            if path is not None:
                path.unlink(missing_ok=True)
                gif.save(frames, str(path), duration=1000 // fps)
            return frames
        except Exception as e:
            self.logwarning(f"GIF generation failed with exception {str(e)}")
            warn(
                "GIF generation failed. Make suer you have gif installed\n\nyou can install it using >> pip install gif",
                NegmasImportWarning,
            )
            return []

    @property
    def business_size(self) -> float:
        """The total business size defined as the total money transferred within the system"""
        return sum(self.stats["activity_level"])

    @property
    def n_negotiation_rounds_successful(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_contracts_concluded"])
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_successful"]) / n_negs

    @property
    def n_negotiation_rounds_failed(self) -> float:
        """Average number of rounds in a successful negotiation"""
        n_negs = sum(self.stats["n_negotiations"]) - self.n_saved_contracts(True)
        if n_negs == 0:
            return np.nan
        return sum(self.stats["n_negotiation_rounds_failed"]) / n_negs

    @property
    def contract_execution_fraction(self) -> float:
        """Fraction of signed contracts successfully executed with no breaches, or errors"""
        n_executed = sum(self.stats["n_contracts_executed"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return n_executed / n_signed_contracts if n_signed_contracts > 0 else 0.0

    @property
    def contract_dropping_fraction(self) -> float:
        """Fraction of signed contracts that were never executed because they were signed to late to be executable"""
        n_dropped = sum(self.stats["n_contracts_dropped"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return n_dropped / n_signed_contracts if n_signed_contracts > 0 else 0.0

    @property
    def contract_err_fraction(self) -> float:
        """Fraction of signed contracts that caused exception during their execution"""
        n_erred = sum(self.stats["n_contracts_erred"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return n_erred / n_signed_contracts if n_signed_contracts > 0 else 0.0

    @property
    def contract_nullification_fraction(self) -> float:
        """Fraction of signed contracts were nullified by the system (e.g. due to bankruptcy)"""
        n_nullified = sum(self.stats["n_contracts_nullified"])
        n_signed_contracts = len(
            [_ for _ in self._saved_contracts.values() if _["signed_at"] >= 0]
        )
        return n_nullified / n_signed_contracts if n_signed_contracts > 0 else 0.0

    @property
    def breach_level(self) -> float:
        """The average breach level per contract"""
        blevel = np.nansum(self.stats["breach_level"])
        n_contracts = sum(self.stats["n_contracts_executed"]) + sum(
            self.stats["n_breaches"]
        )
        return blevel / n_contracts if n_contracts > 0 else 0.0

    @abstractmethod
    def delete_executed_contracts(self) -> None:
        """Called after processing executable contracts at every simulation step to delete processed contracts"""

    @abstractmethod
    def executable_contracts(self) -> Collection[Contract]:
        """Called at every time-step to get the contracts that are `executable` at this point of the simulation"""

    def get_dropped_contracts(self) -> Collection[Contract]:
        """Called at the end of every time-step to get a list of the contracts that are signed but will never be
        executed"""
        return []

    def post_step_stats(self):
        """Called at the end of the simulation step to update all stats

        Kept for backward compatibility and will be dropped. Override `update_stats` ins
        """

    def pre_step_stats(self):
        """Called at the beginning of the simulation step to prepare stats or update them

        Kept for backward compatibility and will be dropped. Override `update_stats` instead
        """

    def update_stats(self, stage: int):
        """
        Called to update any custom stats that the world designer wants to keep

        Args:
            stage: How many times was this method called during this stage

        Remarks:

            - Default behavior is:
              - If `Operations` . `StatsUpdate` appears once in operations, it calls post_step_stats once
              - Otherwise: it calls pre_step_stats for stage 0,  and post_step_stats for any other stage.
        """
        if self._single_stats_call:
            self.post_step_stats()
            return
        if stage == 0:
            self.pre_step_stats()
            return
        self.post_step_stats()
        return

    @abstractmethod
    def order_contracts_for_execution(
        self, contracts: Collection[Contract]
    ) -> Collection[Contract]:
        """Orders the contracts in a specific time-step that are about to be executed"""
        return contracts

    @abstractmethod
    def start_contract_execution(self, contract: Contract) -> set[Breach] | None:
        """
        Tries to execute the contract

        Args:
            contract:

        Returns:
            Set[Breach]: The set of breaches committed if any. If there are no breaches return an empty set

        Remarks:

            - You must call super() implementation of this method before doing anything
            - It is possible to return None which indicates that the contract was nullified (i.e. not executed due to a
              reason other than an execution exeception).

        """
        self.loginfo(
            f"Executing {str(contract)}",
            Event("executing-contract", dict(contract=contract)),
        )
        return set()

    @abstractmethod
    def complete_contract_execution(
        self, contract: Contract, breaches: list[Breach], resolution: Contract
    ) -> None:
        """
        Called after breach resolution is completed for contracts for which some potential breaches occurred.

        Args:
            contract: The contract considered.
            breaches: The list of potential breaches that was generated by `_execute_contract`.
            resolution: The agreed upon resolution

        Returns:

        """

    @abstractmethod
    def execute_action(
        self, action: Action, agent: Agent, callback: Callable | None = None
    ) -> bool:
        """Executes the given action by the given agent"""

    @abstractmethod
    def get_private_state(self, agent: Agent) -> dict:
        """Reads the private state of the given agent"""

    @abstractmethod
    def simulation_step(self, stage: int = 0):
        """A single step of the simulation.

        Args:
            stage: How many times so far was this method called within the current simulation step

        Remarks:

            - Using the stage parameter, it is possible to have `Operations` . `SimulationStep` several times with
              the list of operations while differentiating between these calls.

        """

    @abstractmethod
    def contract_size(self, contract: Contract) -> float:
        """
        Returns an estimation of the **activity level** associated with this contract. Higher is better
        Args:
            contract:

        Returns:

        """

    def __getstate__(self):
        state = self.__dict__.copy()
        if "logger" in state.keys():
            state.pop("logger", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.logger = create_loggers(
            file_name=self.log_file_name,
            module_name=None,
            screen_level=self.log_screen_level if self.log_to_screen else None,
            file_level=self.log_file_level,
            app_wide_log_file=True,
        )
