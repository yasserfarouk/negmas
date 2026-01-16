"""Provides interfaces for defining negotiation mechanisms."""

from __future__ import annotations
import copy
from pathlib import Path
import math
import pprint
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Sequence,
    cast,
    runtime_checkable,
    Protocol,
    TypeVar,
    Generic,
)
from warnings import warn

from attrs import define

from negmas import warnings
from negmas.checkpoints import CheckpointMixin
from negmas.common import (
    DEFAULT_JAVA_PORT,
    MechanismAction,
    MechanismState,
    NegotiatorInfo,
    NegotiatorMechanismInterface,
    TRACE_ELEMENT_MEMBERS,
    TraceElement,
)
from negmas.events import Event, EventSource
from negmas.helpers import snake_case
from negmas.helpers.inout import DEFAULT_TABLE_STORAGE_FORMAT, TableStorageFormat
from negmas.helpers.misc import get_free_tcp_port
from negmas.helpers.strings import humanize_time
from negmas.negotiators import Negotiator
from negmas.outcomes import Outcome
from negmas.outcomes.common import check_one_and_only, ensure_os
from negmas.outcomes.protocols import OutcomeSpace
from negmas.preferences import (
    kalai_points,
    nash_points,
    pareto_frontier,
    pareto_frontier_bf,
)
from negmas.preferences.crisp_ufun import UtilityFunction
from negmas.preferences.ops import (
    OutcomeOptimality,
    max_relative_welfare_points,
    max_welfare_points,
)
from negmas.types import NamedObject

if TYPE_CHECKING:
    from negmas.outcomes.base_issue import Issue
    from negmas.outcomes.protocols import DiscreteOutcomeSpace
    from negmas.preferences import Preferences
    from negmas.preferences.base_ufun import BaseUtilityFunction
    from negmas.inout import Scenario

__all__ = ["Mechanism", "MechanismStepResult", "Traceable", "CompletedRun"]

TState = TypeVar("TState", bound=MechanismState)
TAction = TypeVar("TAction", bound=MechanismAction)
TNMI = TypeVar("TNMI", bound=NegotiatorMechanismInterface)
TNegotiator = TypeVar("TNegotiator", bound=Negotiator)


@define(frozen=True)
class MechanismStepResult(Generic[TState]):
    """
    Represents the results of a negotiation step.

    This is what `round()` should return.
    """

    state: TState
    """The returned state."""
    completed: bool = True
    """Whether the current round is completed or not."""
    broken: bool = False
    """True only if END_NEGOTIATION was selected by one negotiator."""
    timedout: bool = False
    """True if a timeout occurred."""
    agreement: Outcome | None = None
    """The agreement if any. Allows for a single outcome or a collection of outcomes."""
    error: bool = False
    """True if an error occurred in the mechanism."""
    error_details: str = ""
    """Detailed description of any error that occurred during the step."""
    waiting: bool = False
    """Whether to consider that the round is still running and call the round
    method again without increasing the step number."""
    exceptions: dict[str, list[str]] | None = None
    """A mapping from negotiator ID to a list of exceptions raised by that
    negotiator in this round."""
    times: dict[str, float] | None = None
    """A mapping from negotiator ID to the time it consumed during this round."""


@define
class CompletedRun(Generic[TState]):
    """Represents a completed negotiation run with all its data.

    This class encapsulates the results of a negotiation including the history,
    scenario, agreement, and various statistics. It can be saved to and loaded
    from disk in various formats.

    Attributes:
        history: The negotiation history/trace data.
        history_type: Type of history stored ("history", "full_trace", "trace", "extended_trace").
        scenario: The negotiation scenario with outcome space and utility functions.
        agreement: The final agreement reached, or None if no agreement.
        agreement_stats: Optimality statistics for the agreement.
        outcome_stats: Basic outcome statistics (agreement, broken, timedout, utilities).
        config: Configuration parameters used for the negotiation.
        metadata: Arbitrary metadata associated with the run.
    """

    history: list[TState] | list[TraceElement] | list[tuple]
    history_type: str
    scenario: Scenario | None
    agreement: Outcome | None
    agreement_stats: OutcomeOptimality | None
    outcome_stats: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def save(
        self,
        parent: Path | str,
        name: str,
        single_file: bool = False,
        per_negotiator: bool = False,
        save_scenario: bool = True,
        save_scenario_stats: bool = False,
        save_agreement_stats: bool = True,
        save_config: bool = True,
        overwrite: bool = True,
        warn_if_existing: bool = True,
        storage_format: TableStorageFormat | None = DEFAULT_TABLE_STORAGE_FORMAT,
    ) -> Path:
        """Saves the completed run to disk.

        Args:
            parent: Parent directory where to save the run.
            name: Name for the saved run (directory name or file name without extension).
            single_file: If True, save only the trace as a single file.
            per_negotiator: If True, save traces per negotiator in a subdirectory.
                Files are named {type}@{index}_{name}.{ext}. Only works with full_trace history type.
            save_scenario: If True, save the scenario information.
            save_scenario_stats: If True, save scenario statistics.
            save_agreement_stats: If True, save agreement statistics.
            save_config: If True, save the configuration.
            overwrite: If True, overwrite existing files/directories.
            warn_if_existing: If True, warn when overwriting.
            storage_format: Format for table storage ("csv", "gzip", "parquet").

        Returns:
            Path to the saved file or directory.
        """
        from negmas.helpers.inout import dump, save_table

        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)

        # Determine column names based on history type
        if self.history_type == "full_trace":
            trace_columns = TRACE_ELEMENT_MEMBERS
        elif self.history_type == "trace":
            trace_columns = ["negotiator", "offer"]
        elif self.history_type == "extended_trace":
            trace_columns = ["step", "negotiator", "offer"]
        else:  # "history" or unknown
            trace_columns = None

        # Determine file extension based on storage format
        ext_map = {"csv": ".csv", "gzip": ".csv.gz", "parquet": ".parquet"}
        ext = ext_map.get(storage_format, ".csv") if storage_format else ".csv"

        if single_file:
            # Save only the trace as a single file
            file_path = parent / f"{name}{ext}"
            if file_path.exists():
                if warn_if_existing:
                    warn(f"File {file_path} already exists")
                if not overwrite:
                    return file_path

            if trace_columns:
                # history is list[tuple] when trace_columns is set (TraceElement is a namedtuple)
                save_table(
                    cast(list[tuple], self.history),
                    file_path,
                    columns=trace_columns,
                    storage_format=storage_format,
                )
            else:
                # History data (list of state dicts)
                # MechanismState is an attrs class with asdict(), namedtuples have _asdict()
                history_dicts: list[dict[str, Any]] = []
                for s in self.history:
                    if hasattr(s, "asdict"):
                        history_dicts.append(s.asdict())  # type: ignore[union-attr]
                    elif hasattr(s, "_asdict"):
                        history_dicts.append(s._asdict())  # type: ignore[union-attr]
                    else:
                        history_dicts.append(dict(s))  # type: ignore[arg-type]
                save_table(history_dicts, file_path, storage_format=storage_format)
            return file_path

        # Multi-file mode: create directory structure
        save_dir = parent / name
        if save_dir.exists():
            if warn_if_existing:
                warn(f"Directory {save_dir} already exists")
            if not overwrite:
                return save_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save trace/history
        trace_file = save_dir / f"trace{ext}"
        if trace_columns:
            # history is list[tuple] when trace_columns is set (TraceElement is a namedtuple)
            save_table(
                cast(list[tuple], self.history),
                trace_file,
                columns=trace_columns,
                storage_format=storage_format,
            )
        else:
            # MechanismState is an attrs class with asdict(), namedtuples have _asdict()
            history_dicts: list[dict[str, Any]] = []
            for s in self.history:
                if hasattr(s, "asdict"):
                    history_dicts.append(s.asdict())  # type: ignore[union-attr]
                elif hasattr(s, "_asdict"):
                    history_dicts.append(s._asdict())  # type: ignore[union-attr]
                else:
                    history_dicts.append(dict(s))  # type: ignore[arg-type]
            save_table(history_dicts, trace_file, storage_format=storage_format)

        # Save per-negotiator traces if requested
        if per_negotiator and self.history_type == "full_trace":
            negotiator_dir = save_dir / "negotiator_behavior"
            negotiator_dir.mkdir(parents=True, exist_ok=True)

            # Get negotiator info from config
            negotiator_names = self.config.get("negotiator_names", [])
            negotiator_types = self.config.get("negotiator_types", [])
            negotiator_ids = self.config.get("negotiator_ids", [])

            # Build a mapping from negotiator_id to (index, name, type)
            # If we don't have IDs, we'll use names as IDs
            if not negotiator_ids:
                negotiator_ids = negotiator_names

            id_to_info: dict[str, tuple[int, str, str]] = {}
            for i, (nid, nname, ntype) in enumerate(
                zip(negotiator_ids, negotiator_names, negotiator_types)
            ):
                id_to_info[nid] = (i, nname, ntype)

            # Group history by negotiator
            # In full_trace mode, entries are TraceElement namedtuples
            negotiator_traces: dict[str, list[dict[str, Any]]] = {}
            for entry in self.history:
                # entry is a TraceElement (namedtuple) with 'negotiator' field
                entry_dict: dict[str, Any] = (
                    entry._asdict()  # type: ignore[union-attr]
                    if hasattr(entry, "_asdict")
                    else {}
                )
                nid = entry_dict.get("negotiator")
                if nid is None:
                    continue
                if nid not in negotiator_traces:
                    negotiator_traces[nid] = []
                # Extract per-negotiator trace: time, relative_time, step, offer, response, text, data
                negotiator_traces[nid].append(
                    {
                        "time": entry_dict.get("time"),
                        "relative_time": entry_dict.get("relative_time"),
                        "step": entry_dict.get("step"),
                        "offer": entry_dict.get("offer"),
                        "response": entry_dict.get("state"),
                        "text": entry_dict.get("text"),
                        "data": entry_dict.get("data"),
                    }
                )

            # Save each negotiator's trace
            for nid, trace in negotiator_traces.items():
                if nid in id_to_info:
                    idx, nname, ntype = id_to_info[nid]
                else:
                    # Fallback: use negotiator_id directly
                    idx = list(negotiator_traces.keys()).index(nid)
                    nname = nid
                    ntype = "Unknown"
                # Naming: {type}@{index}_{name}.{ext}
                filename = f"{ntype}@{idx}_{nname}{ext}"
                neg_file = negotiator_dir / filename
                save_table(trace, neg_file, storage_format=storage_format)

        # Save run info (history_type, agreement, basic stats)
        run_info: dict[str, Any] = {
            "history_type": self.history_type,
            "agreement": self.agreement,
            "storage_format": storage_format,
        }
        dump(run_info, save_dir / "run_info.yaml")

        # Save scenario
        if save_scenario and self.scenario is not None:
            scenario_dir = save_dir / "scenario"
            try:
                if save_scenario_stats:
                    self.scenario.calc_stats()
                self.scenario.dumpas(
                    scenario_dir,
                    type="yml",
                    save_stats=save_scenario_stats,
                    save_info=True,
                )
            except Exception:
                # If scenario saving fails, save basic info
                scenario_dir.mkdir(parents=True, exist_ok=True)
                dump(
                    {"outcome_space": str(self.scenario.outcome_space)},
                    scenario_dir / "outcome_space.yaml",
                )

        # Save outcome stats (always saved in directory mode)
        if self.outcome_stats:
            dump(self.outcome_stats, save_dir / "outcome_stats.yaml")

        # Save agreement stats (optimality stats)
        if save_agreement_stats and self.agreement_stats is not None:
            stats_dict = {
                "pareto_optimality": self.agreement_stats.pareto_optimality,
                "nash_optimality": self.agreement_stats.nash_optimality,
                "kalai_optimality": self.agreement_stats.kalai_optimality,
                "modified_kalai_optimality": self.agreement_stats.modified_kalai_optimality,
                "max_welfare_optimality": self.agreement_stats.max_welfare_optimality,
                "ks_optimality": self.agreement_stats.ks_optimality,
                "modified_ks_optimality": self.agreement_stats.modified_ks_optimality,
            }
            dump(stats_dict, save_dir / "agreement_stats.yaml")

        # Save config
        if save_config and self.config:
            dump(self.config, save_dir / "config.yaml")

        # Save metadata
        if self.metadata:
            dump(self.metadata, save_dir / "metadata.yaml")

        return save_dir

    @classmethod
    def load(
        cls,
        path: Path | str,
        load_scenario: bool = True,
        load_scenario_stats: bool = False,
        load_agreement_stats: bool = True,
        load_config: bool = True,
    ) -> "CompletedRun[TState]":
        """Loads a completed run from the given path.

        Args:
            path: Path to a file (single-file mode) or directory (multi-file mode).
            load_scenario: If True, load the scenario from the scenario directory.
            load_scenario_stats: If True, load scenario statistics when loading the scenario.
            load_agreement_stats: If True, load agreement optimality statistics.
            load_config: If True, load the configuration from config.yaml.

        Returns:
            A CompletedRun instance with the loaded data.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path format is not recognized.
        """
        from negmas.helpers.inout import load, load_table
        from negmas.inout import Scenario

        path = Path(path).expanduser().absolute()

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            # Single-file mode: load just the trace
            history = load_table(path, as_dataframe=False)

            # Detect history type from columns
            history_type = "history"  # default
            if len(history) > 0:
                first_row = history[0]
                if isinstance(first_row, dict):
                    keys = set(first_row.keys())
                    # Core full_trace columns (without optional text/data)
                    full_trace_core_cols = {
                        "time",
                        "relative_time",
                        "step",
                        "negotiator",
                        "offer",
                        "responses",
                        "state",
                    }
                    trace_cols = {"negotiator", "offer"}
                    extended_trace_cols = {"step", "negotiator", "offer"}

                    # Detect full_trace if core columns are present (text/data are optional)
                    if full_trace_core_cols.issubset(keys):
                        history_type = "full_trace"
                    elif keys == extended_trace_cols:
                        history_type = "extended_trace"
                    elif keys == trace_cols:
                        history_type = "trace"
                    # else: remains "history" (state dicts)

            return cls(
                history=history,  # type: ignore
                history_type=history_type,
                scenario=None,
                agreement=None,
                agreement_stats=None,
                outcome_stats={},
                config={},
                metadata={},
            )

        # Multi-file mode: load from directory
        if not path.is_dir():
            raise ValueError(f"Path is neither a file nor directory: {path}")

        # Load run info
        run_info_path = path / "run_info.yaml"
        if run_info_path.exists():
            run_info = load(run_info_path)
            history_type = run_info.get("history_type", "history")
            agreement = run_info.get("agreement")
            storage_format = run_info.get("storage_format", "csv")
        else:
            history_type = "history"
            agreement = None
            storage_format = "csv"

        # Determine trace file extension
        ext_map = {"csv": ".csv", "gzip": ".csv.gz", "parquet": ".parquet"}
        ext = ext_map.get(storage_format, ".csv")

        # Try to find and load trace file
        trace_file = path / f"trace{ext}"
        if not trace_file.exists():
            # Try other extensions
            for try_ext in [".csv", ".csv.gz", ".parquet"]:
                trace_file = path / f"trace{try_ext}"
                if trace_file.exists():
                    break
            else:
                raise FileNotFoundError(f"No trace file found in {path}")

        history = load_table(trace_file, as_dataframe=False)

        # Load scenario if present and requested
        scenario = None
        if load_scenario:
            scenario_dir = path / "scenario"
            if scenario_dir.exists():
                try:
                    scenario = Scenario.load(
                        scenario_dir, load_stats=load_scenario_stats
                    )
                except Exception:
                    scenario = None

        # Load agreement stats if present and requested
        agreement_stats = None
        if load_agreement_stats:
            stats_path = path / "agreement_stats.yaml"
            if stats_path.exists():
                stats_dict = load(stats_path)
                if stats_dict:
                    agreement_stats = OutcomeOptimality(
                        pareto_optimality=stats_dict.get(
                            "pareto_optimality", float("nan")
                        ),
                        nash_optimality=stats_dict.get("nash_optimality", float("nan")),
                        kalai_optimality=stats_dict.get(
                            "kalai_optimality", float("nan")
                        ),
                        modified_kalai_optimality=stats_dict.get(
                            "modified_kalai_optimality", float("nan")
                        ),
                        max_welfare_optimality=stats_dict.get(
                            "max_welfare_optimality", float("nan")
                        ),
                        ks_optimality=stats_dict.get("ks_optimality", float("nan")),
                        modified_ks_optimality=stats_dict.get(
                            "modified_ks_optimality", float("nan")
                        ),
                    )

        # Load config if present and requested
        config: dict[str, Any] = {}
        if load_config:
            config_path = path / "config.yaml"
            config = load(config_path) if config_path.exists() else {}

        # Load metadata if present
        metadata_path = path / "metadata.yaml"
        metadata = load(metadata_path) if metadata_path.exists() else {}

        # Load outcome stats if present
        outcome_stats_path = path / "outcome_stats.yaml"
        outcome_stats = load(outcome_stats_path) if outcome_stats_path.exists() else {}

        return cls(
            history=history,  # type: ignore
            history_type=history_type,
            scenario=scenario,
            agreement=agreement,
            agreement_stats=agreement_stats,
            outcome_stats=outcome_stats if outcome_stats else {},
            config=config if config else {},
            metadata=metadata if metadata else {},
        )


class Mechanism(
    NamedObject,
    EventSource,
    CheckpointMixin,
    Generic[TNMI, TState, TAction, TNegotiator],
    ABC,
):
    """Base class for all negotiation Mechanisms.

    Override the `round` function of this class to implement a round of your mechanism

    Args:
        outcome_space: The negotiation agenda
        outcomes: list of outcomes (optional as you can pass `issues`). If an int then it is the number of outcomes
        n_steps: Number of rounds allowed (None means infinity)
        time_limit: Number of real seconds allowed (None means infinity)
        pend: Probability of ending the negotiation at any step
        pend_per_second: Probability of ending the negotiation every second
        hidden_time_limit: Number of real seconds allowed but not visilbe to the negotiators
        max_n_negotiators:  Maximum allowed number of negotiators.
        dynamic_entry: Allow negotiators to enter/leave negotiations between rounds
        cache_outcomes: If true, a list of all possible outcomes will be cached
        max_cardinality: The maximum allowed number of outcomes in the cached set
        annotation: Arbitrary annotation
        checkpoint_every: The number of steps to checkpoint after. Set to <= 0 to disable
        checkpoint_folder: The folder to save checkpoints into. Set to None to disable
        checkpoint_filename: The base filename to use for checkpoints (multiple checkpoints will be prefixed with
                             step number).
        single_checkpoint: If true, only the most recent checkpoint will be saved.
        extra_checkpoint_info: Any extra information to save with the checkpoint in the corresponding json file as
                               a dictionary with string keys
        exist_ok: IF true, checkpoints override existing checkpoints with the same filename.
        name: Name of the mechanism session. Should be unique. If not given, it will be generated.
        genius_port: the port used to connect to Genius for all negotiators in this mechanism (0 means any).
        id: An optional system-wide unique identifier. You should not change
            the default value except in special circumstances like during
            serialization and should always guarantee system-wide uniquness
            if you set this value explicitly
    """

    def __init__(
        self,
        initial_state: TState | None = None,
        outcome_space: OutcomeSpace | None = None,
        issues: Sequence[Issue] | None = None,
        outcomes: Sequence[Outcome] | int | None = None,
        n_steps: int | float | None = None,
        time_limit: float | None = None,
        pend: float = 0,
        pend_per_second: float = 0,
        step_time_limit: float | None = None,
        negotiator_time_limit: float | None = None,
        hidden_time_limit: float = float("inf"),
        max_n_negotiators: int | None = None,
        dynamic_entry=False,
        annotation: dict[str, Any] | None = None,
        nmi_factory: type[TNMI] = NegotiatorMechanismInterface,
        extra_callbacks=False,
        checkpoint_every: int = 1,
        checkpoint_folder: PathLike | None = None,
        checkpoint_filename: str | None = None,
        extra_checkpoint_info: dict[str, Any] | None = None,
        single_checkpoint: bool = True,
        exist_ok: bool = True,
        name=None,
        genius_port: int = DEFAULT_JAVA_PORT,
        id: str | None = None,
        type_name: str | None = None,
        verbosity: int = 0,
        ignore_negotiator_exceptions=False,
    ):
        """Initialize the instance.

        Args:
            initial_state: Initial state.
            outcome_space: Outcome space.
            issues: Issues.
            outcomes: Outcomes.
            n_steps: N steps.
            time_limit: Time limit.
            pend: Pend.
            pend_per_second: Pend per second.
            step_time_limit: Step time limit.
            negotiator_time_limit: Negotiator time limit.
            hidden_time_limit: Hidden time limit.
            max_n_negotiators: Max n negotiators.
            dynamic_entry: Dynamic entry.
            annotation: Annotation.
            nmi_factory: Nmi factory.
            extra_callbacks: Extra callbacks.
            checkpoint_every: Checkpoint every.
            checkpoint_folder: Checkpoint folder.
            checkpoint_filename: Checkpoint filename.
            extra_checkpoint_info: Extra checkpoint info.
            single_checkpoint: Single checkpoint.
            exist_ok: Exist ok.
            name: Name.
            genius_port: Genius port.
            id: Id.
            type_name: Type name.
            verbosity: Verbosity.
            ignore_negotiator_exceptions: Ignore negotiator exceptions.
        """
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        self.__verbosity = verbosity
        self._negotiator_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        super().__init__(name=name, id=id, type_name=type_name)

        self.ignore_negotiator_exceptions = ignore_negotiator_exceptions
        self._negotiator_times = defaultdict(float)
        CheckpointMixin.checkpoint_init(
            self,
            step_attrib="_step",
            every=checkpoint_every,
            folder=checkpoint_folder,
            filename=checkpoint_filename,
            info=extra_checkpoint_info,
            exist_ok=exist_ok,
            single=single_checkpoint,
        )
        self.__last_second_tried = 0
        self._hidden_time_limit = (
            hidden_time_limit if hidden_time_limit is not None else float("inf")
        )
        time_limit = time_limit if time_limit is not None else float("inf")
        step_time_limit = (
            step_time_limit if step_time_limit is not None else float("inf")
        )
        negotiator_time_limit = (
            negotiator_time_limit if negotiator_time_limit is not None else float("inf")
        )

        # parameters fixed for all runs

        self.set_id(str(uuid.uuid4()))
        if n_steps == float("inf"):
            n_steps = None
        if isinstance(n_steps, float):
            n_steps = int(n_steps)
        if pend is None:
            pend = 0
        if pend_per_second is None:
            pend_per_second = 0
        self.nmi = nmi_factory(
            id=self.id,
            n_outcomes=outcome_space.cardinality,
            outcome_space=outcome_space,
            time_limit=time_limit,
            pend=pend,
            pend_per_second=pend_per_second,
            n_steps=n_steps,
            step_time_limit=step_time_limit,
            negotiator_time_limit=negotiator_time_limit,
            dynamic_entry=dynamic_entry,
            max_n_negotiators=max_n_negotiators,
            annotation=annotation if annotation is not None else dict(),
            _mechanism=self,
        )

        self._current_state = initial_state if initial_state else MechanismState()  # type: ignore This is a shortcut to allow users to create mechanisms without passing any initial_state
        self._current_state: TState

        self._history: list[TState] = []
        self._stats: dict[str, Any] = dict()
        self._stats["round_times"] = list()
        self._stats["times"] = defaultdict(float)
        self._stats["exceptions"] = defaultdict(list)
        # if self.nmi.issues is not None:
        #     self.nmi.issues = tuple(self.nmi.issues)
        # if self.nmi.outcomes is not None:
        #     self.nmi.outcomes = tuple(self.nmi.outcomes)

        self._requirements = {}
        self._negotiators: list[TNegotiator] = []
        self._negotiator_map: dict[str, TNegotiator] = dict()
        self._negotiator_index: dict[str, int] = dict()
        self._roles = []
        self._start_time = None
        self.__discrete_os = None
        self.__discrete_outcomes = None
        self._extra_callbacks = extra_callbacks

        self.negotiators_of_role = defaultdict(list)
        self.role_of_negotiator = {}
        # mechanisms do not differentiate between RANDOM_JAVA_PORT and ANY_JAVA_PORT.
        # if either is given as the genius_port, it will fix a port and all negotiators
        # that are not explicitly assigned to a port (by passing port>0 to them) will just
        # use that port.
        self.genius_port = genius_port if genius_port > 0 else get_free_tcp_port()

        self.params: dict[str, Any] = dict(
            dynamic_entry=dynamic_entry, genius_port=genius_port, annotation=annotation
        )

    def log(self, nid: str, data: dict[str, Any], level: str) -> None:
        """Saves a log for a negotiator"""
        d = data | dict(
            step=self.current_step,
            relative_time=self.relative_time,
            time=self.time,
            level=level,
        )
        self._negotiator_logs[nid].append(d)

    def log_info(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at info level"""
        self.log(nid, level="info", data=data)

    def log_debug(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at debug level"""
        self.log(nid, level="debug", data=data)

    def log_warning(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at warning level"""
        self.log(nid, level="warning", data=data)

    def log_error(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at error level"""
        self.log(nid, level="error", data=data)

    def log_critical(self, nid: str, data: dict[str, Any]) -> None:
        """Logs at critical level"""
        self.log(nid, level="critical", data=data)

    @property
    def negotiator_times(self) -> dict[str, float]:
        """The total time consumed by every negotiator.

        Each mechanism class is responsible of updating this for any activities of the negotiator it controls.
        """
        return self._negotiator_times

    @property
    def negotiators(self) -> list[TNegotiator]:
        """All negotiators participating in this negotiation mechanism.

        Returns:
            list[TNegotiator]: List of negotiator objects currently registered
        """
        return self._negotiators

    @property
    def participants(self) -> list[NegotiatorInfo]:
        """Returns a list of all participant names."""
        return [
            NegotiatorInfo(name=_.name, id=_.id, type=snake_case(_.__class__.__name__))
            for _ in self.negotiators
        ]

    def is_valid(self, outcome: Outcome):
        """Checks whether the outcome is valid given the issues."""
        return outcome in self.nmi.outcome_space

    @property
    def outcome_space(self) -> OutcomeSpace:
        """The space of all possible negotiation outcomes.

        Returns:
            OutcomeSpace: Defines valid outcomes including issues, values, and constraints
        """
        return self.nmi.outcome_space

    def discrete_outcome_space(
        self, levels: int = 5, max_cardinality: int = 10_000_000_000
    ) -> DiscreteOutcomeSpace:
        """Returns a stable discrete version of the given outcome-space."""
        if self.__discrete_os:
            return self.__discrete_os
        self.__discrete_os = self.outcome_space.to_discrete(
            levels=levels,
            max_cardinality=max_cardinality
            if max_cardinality is not None
            else float("inf"),
        )
        return self.__discrete_os

    @property
    def outcomes(self):
        """All possible outcomes for discrete spaces, or None for continuous spaces."""
        return self.nmi.outcomes

    def discrete_outcomes(
        self, levels: int = 5, max_cardinality: int | float = float("inf")
    ) -> list[Outcome]:
        """A discrete set of outcomes that spans the outcome space.

        Args:
            max_cardinality: The maximum number of outcomes to return. If None, all outcomes will be returned for discrete issues
            and *10_000* if any of the issues was continuous

        Returns:

            list[Outcome]: list of `n` or less outcomes
        """
        if self.outcomes is not None:
            return list(self.outcomes)
        if self.__discrete_outcomes:
            return self.__discrete_outcomes
        self.__discrete_os = self.outcome_space.to_discrete(
            levels=levels,
            max_cardinality=max_cardinality
            if max_cardinality is not None
            else float("inf"),
        )
        self.__discrete_outcomes = list(self.__discrete_os.enumerate_or_sample())
        return self.__discrete_outcomes

    def random_outcomes(
        self, n: int = 1, with_replacement: bool = False
    ) -> list[Outcome]:
        """Returns random outcomes.

        Args:
              n: Number of outcomes to generate
              with_replacement: If true, outcomes may be repeated

        Returns:
              A list of outcomes of at most n outcomes.

        Remarks:

                - If the number of outcomes `n` cannot be satisfied, a smaller number will be returned
                - Sampling is done without replacement (i.e. returned outcomes are unique).
        """
        return list(
            self.outcome_space.sample(
                n, with_replacement=with_replacement, fail_if_not_enough=False
            )
        )

    def random_outcome(self) -> Outcome:
        """Returns a single random offer"""
        return self.outcome_space.random_outcome()

    @property
    def time(self) -> float:
        """Elapsed time since mechanism started in seconds.

        0.0 if the mechanism did not start running
        """
        if self._start_time is None:
            return 0.0

        return time.perf_counter() - self._start_time

    @property
    def remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds.

        None if no time limit is given.
        """
        if self.nmi.time_limit == float("+inf"):
            return None
        if not self._start_time:
            return self.nmi.time_limit

        limit = self.nmi.time_limit - (time.perf_counter() - self._start_time)
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def expected_remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds (expectation).

        None if no time limit or pend_per_second is given.
        """
        rem = self.remaining_time
        pend = self.nmi.pend_per_second
        if pend <= 0:
            return rem
        return min(rem, (1 / pend)) if rem is not None else (1 / pend)

    @property
    def relative_time(self) -> float:
        """
        Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps.

        Remarks:
            - If pend or pend_per_second are defined in the `NegotiatorMechanismInterface`,
              and time_limit/n_steps are not given, this becomes an expectation that is limited above by one.
        """
        n_steps = self.nmi.n_steps
        time_limit = self.nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if self.nmi.pend <= 0 and self.nmi.pend_per_second <= 0:
                return 0.0
            if self.nmi.pend > 0:
                n_steps = int(math.ceil(1 / self.nmi.pend))
            if self.nmi.pend_per_second > 0:
                time_limit = 1 / self.nmi.pend_per_second

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit is not None else -1.0
        return min(1.0, max([relative_step, relative_time]))

    @property
    def expected_relative_time(self) -> float:
        """
        Returns a positive number indicating elapsed relative time or steps.

        Remarks:
            - This is relative to the expected time/step at which the negotiation ends given all timing
              conditions (time_limit, n_step, pend, pend_per_second).
        """
        n_steps = self.nmi.n_steps
        time_limit = self.nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if self.nmi.pend <= 0 and self.nmi.pend_per_second <= 0:
                return 0.0
        if n_steps is None:
            # set the expected number of steps to the reciprocal of the probability of ending at every step
            n_steps = (
                int(math.ceil(1 / self.nmi.pend)) if self.nmi.pend > 0 else n_steps
            )
        else:
            n_steps = min(int(math.ceil(1 / self.nmi.pend)), n_steps)
        if time_limit == float("inf"):
            # set the expected number of seconds to the reciprocal of the probability of ending every second
            time_limit = (
                int(math.ceil(1 / self.nmi.pend_per_second))
                if self.nmi.pend_per_second > 0
                else time_limit
            )
        else:
            time_limit = (
                min(time_limit, int(math.ceil(1 / self.nmi.pend_per_second)))
                if self.nmi.pend_per_second > 0
                else time_limit
            )

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit is not None else -1.0
        return max([relative_step, relative_time])

    @property
    def remaining_steps(self) -> int | None:
        """
        Returns the remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        if self.nmi.n_steps is None:
            return None

        return self.nmi.n_steps - self._current_state.step

    @property
    def expected_remaining_steps(self) -> int | None:
        """
        Returns the expected remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        rem = self.remaining_steps
        pend = self.nmi.pend
        if pend <= 0:
            return rem

        return (
            min(rem, int(math.ceil(1 / pend)))
            if rem is not None
            else int(math.ceil(1 / pend))
        )

    @property
    def requirements(self):
        """A dictionary specifying the requirements that must be in the
        capabilities of any negotiator to join the mechanism."""
        return self._requirements

    @requirements.setter
    def requirements(
        self,
        requirements: dict[
            str,
            (
                tuple[int | float | str, int | float | str]
                | list
                | set
                | int
                | float
                | str
            ),
        ],
    ):
        """Set negotiation requirements that negotiators must satisfy.

        Args:
            requirements: Dict mapping requirement names to acceptable values (single value, tuple range, or list/set of options)
        """
        self._requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in requirements.items()
        }

    def is_satisfying(self, capabilities: dict) -> bool:
        """Checks if the  given capabilities are satisfying mechanism
        requirements.

        Args:
            capabilities: capabilities to check

        Returns:
            bool are the requirements satisfied by the capabilities.

        Remarks:

            - Requirements are also a dict with the following meanings:

                - tuple: Min and max acceptable values
                - list/set: Any value in the iterable is acceptable
                - Single value: The capability must match this value

            - Capabilities can also have the same three possibilities.
        """
        requirements = self.requirements
        for r, v in requirements.items():
            if v is None:
                if r not in capabilities.keys():
                    return False

                else:
                    continue

            if r not in capabilities.keys():
                return False

            if capabilities[r] is None:
                continue

            c = capabilities[r]
            if isinstance(c, tuple):
                # c is range
                if isinstance(v, tuple):
                    # both ranges
                    match = v[0] <= c[0] <= v[1] or v[0] <= c[1] <= v[1]
                else:
                    # c is range and cutoff_utility is not a range
                    match = (
                        any(c[0] <= _ <= c[1] for _ in v)
                        if isinstance(v, set)
                        else c[0] <= v <= c[1]
                    )
            elif isinstance(c, list) or isinstance(c, set):
                # c is list
                if isinstance(v, tuple):
                    # c is a list and cutoff_utility is a range
                    match = any(v[0] <= _ <= v[1] for _ in c)
                else:
                    # c is a list and cutoff_utility is not a range
                    match = any(_ in v for _ in c) if isinstance(v, set) else v in c
            else:
                # c is a single value
                if isinstance(v, tuple):
                    # c is a singlton and cutoff_utility is a range
                    match = v[0] <= c <= v[1]
                else:
                    # c is a singlton and cutoff_utility is not a range
                    match = c in v if isinstance(v, set) else c == v
            if not match:
                return False

        return True

    def can_participate(self, negotiator: TNegotiator) -> bool:
        """Checks if the negotiator can participate in this type of negotiation in
        general.

        Returns:
            bool: True if the negotiator  can participate

        Remarks:
            The only reason this may return `False` is if the mechanism requires some requirements
            that are not within the capabilities of the negotiator.

            When evaluating compatibility, the negotiator is considered incapable of participation if any
            of the following conditions hold:
            * A mechanism requirement is not in the capabilities of the negotiator
            * A mechanism requirement is in the capabilities of the negotiator by the values required for it
              is not in the values announced by the negotiator.

            An negotiator that lists a `None` value for a capability is announcing that it can work with all its
            values. On the other hand, a mechanism that lists a requirement as None announces that it accepts
            any value for this requirement as long as it exist in the negotiator
        """
        return self.is_satisfying(negotiator.capabilities)

    def can_accept_more_negotiators(self) -> bool:
        """Whether the mechanism can **currently** accept more negotiators."""
        return (
            True
            if self.nmi.max_n_negotiators is None or self._negotiators is None
            else len(self._negotiators) < self.nmi.max_n_negotiators
        )

    def can_enter(self, negotiator: TNegotiator) -> bool:
        """Whether the negotiator can enter the negotiation now."""
        return self.can_accept_more_negotiators() and self.can_participate(negotiator)

    # def extra_state(self) -> dict[str, Any] | None:
    #     """Returns any extra state information to be kept in the `state` and `history` properties"""
    #     return dict()

    @property
    def state(self) -> TState:
        """Returns the current state.

        Override `extra_state` if you want to keep extra state
        """
        return self._current_state

    def _get_nmi(self, negotiator: TNegotiator) -> TNMI:
        _ = negotiator
        return self.nmi

    def _get_ami(self, negotiator: TNegotiator) -> TNMI:
        _ = negotiator
        warnings.deprecated("_get_ami is depricated. Use `get_nmi` instead of it")
        return self.nmi

    def add(
        self,
        negotiator: TNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
    ) -> bool | None:
        """Add an negotiator to the negotiation.

        Args:

            negotiator: The negotiator to be added.
            preferences: The utility function to use. If None, then the negotiator must already have a stored
                  utility function otherwise it will fail to enter the negotiation.
            ufun: [depricated] same as preferences but must be a `UFun` object.
            role: The role the negotiator plays in the negotiation mechanism. It is expected that mechanisms inheriting from
                  this class will check this parameter to ensure that the role is a valid role and is still possible for
                  negotiators to join on that role. Roles may include things like moderator, representative etc based
                  on the mechanism


        Returns:

            * True if the negotiator was added.
            * False if the negotiator was already in the negotiation.
            * None if the negotiator cannot be added. This can happen in the following cases:

              1. The capabilities of the negotiator do not match the requirements of the negotiation
              2. The outcome-space of the negotiator's preferences do not contain the outcome-space of the negotiation
              3. The negotiator refuses to join (by returning False from its `join` method) see `Negotiator.join` for possible reasons of that
        """

        from negmas.preferences import (
            BaseUtilityFunction,
            MappingUtilityFunction,
            Preferences,
        )

        if ufun is not None:
            if not isinstance(ufun, BaseUtilityFunction):
                ufun = MappingUtilityFunction(ufun, outcome_space=self.outcome_space)
            preferences = ufun
        if (
            preferences is not None
            and not isinstance(preferences, Preferences)
            and isinstance(preferences, Callable)
        ):
            preferences = MappingUtilityFunction(
                preferences, outcome_space=self.outcome_space
            )
        if (
            preferences
            and preferences.outcome_space
            and self.outcome_space not in preferences.outcome_space
        ):
            return None
        if not self.can_enter(negotiator):
            return None

        if negotiator in self._negotiators:
            return False

        if role is None:
            role = "negotiator"

        if negotiator.join(
            nmi=self._get_nmi(negotiator),
            state=self.state,
            preferences=preferences,
            role=role,
        ):
            self._negotiators.append(negotiator)
            self._current_state.n_negotiators += 1
            self._negotiator_map[negotiator.id] = negotiator
            self._negotiator_index[negotiator.id] = len(self._negotiators) - 1
            self._roles.append(role)
            self.role_of_negotiator[negotiator.uuid] = role
            self.negotiators_of_role[role].append(negotiator)
            return True
        return None

    def get_negotiator(self, source: str) -> Negotiator | None:
        """Returns the negotiator with the given ID if present in the
        negotiation."""
        return self._negotiator_map.get(source, None)

    def get_negotiator_raise(self, source: str) -> Negotiator:
        """Returns the negotiator with the given ID if present in the
        negotiation otherwise it raises an exception."""
        return self._negotiator_map[source]

    def can_leave(self, negotiator: Negotiator) -> bool:
        """Can the negotiator leave now?"""
        return (
            True
            if self.nmi.dynamic_entry
            else not self.nmi.state.running and negotiator in self._negotiators
        )

    def _call(self, negotiator: TNegotiator, callback: Callable, *args, **kwargs):
        result = None
        try:
            result = callback(*args, **kwargs)
        except Exception as e:
            if self.ignore_negotiator_exceptions:
                pass
            else:
                self.state.has_error = True
                self.state.error_details = str(e)
                self.state.erred_negotiator = negotiator.id if negotiator else ""
                a = negotiator.owner if negotiator else None
                self.state.erred_agent = a.id if a else ""
        return result

    def remove(self, negotiator: TNegotiator) -> bool | None:
        """Remove the negotiator from the negotiation.

        Returns:
            * True if the negotiator was removed.
            * False if the negotiator was not in the negotiation already.
            * None if the negotiator cannot be removed.
        """
        if not self.can_leave(negotiator):
            return False
        n = self._negotiator_map.get(negotiator.id, None)
        if n is None:
            return False
        indx = self._negotiator_index.pop(negotiator.id, None)
        if indx is not None:
            self._negotiators = self._negotiators[:indx] + self._negotiators[indx + 1 :]
        self._negotiator_map.pop(negotiator.id)
        if self._extra_callbacks:
            strt = time.perf_counter()
            self._call(negotiator, negotiator.on_leave, self.nmi.state)
            self._negotiator_times[negotiator.id] += time.perf_counter() - strt
        return True

    def add_requirements(self, requirements: dict) -> None:
        """Merges new requirements into the existing mechanism requirements.

        Args:
            requirements: Dict mapping requirement names to acceptable values
        """
        requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in requirements.items()
        }
        if hasattr(self, "_requirements"):
            self._requirements.update(requirements)
        else:
            self._requirements = requirements

    def remove_requirements(self, requirements: Iterable) -> None:
        """Removes specified requirements from the mechanism.

        Args:
            requirements: Iterable of requirement names to remove
        """
        for r in requirements:
            if r in self._requirements.keys():
                self._requirements.pop(r, None)

    def negotiator_index(self, source: str) -> int | None:
        """Gets the negotiator index.

        Args:
            source (str): source

        Returns:
            int | None:
        """

        return self._negotiator_index.get(source, None)

    @property
    def negotiator_ids(self) -> list[str]:
        """Negotiator ids.

        Returns:
            list[str]: The result.
        """
        return [_.id for _ in self._negotiators]

    @property
    def genius_negotiator_ids(self) -> list[str]:
        """Genius negotiator ids.

        Returns:
            list[str]: The result.
        """
        return [
            getattr(_, "java_uuid") if hasattr(_, "java_uuid") else _.id
            for _ in self._negotiators
        ]

    def genius_id(self, id: str | None) -> str | None:
        """Gets the Genius ID corresponding to the given negotiator if known otherwise its normal ID"""
        if id is None:
            return None
        negotiator = self._negotiator_map.get(id, None)
        if not negotiator:
            return id

        return (
            getattr(negotiator, "java_uuid")
            if hasattr(negotiator, "java_uuid")
            else negotiator.id
        )

    @property
    def agent_ids(self) -> list[str | None]:
        """Agent ids.

        Returns:
            list[str | None]: The result.
        """
        return [_.owner.id if _.owner else None for _ in self._negotiators]

    @property
    def agent_names(self) -> list[str | None]:
        """Agent names.

        Returns:
            list[str | None]: The result.
        """
        return [_.owner.name if _.owner else None for _ in self._negotiators]

    @property
    def negotiator_names(self) -> list[str]:
        """Negotiator names.

        Returns:
            list[str]: The result.
        """
        return [_.name for _ in self._negotiators]

    @property
    def agreement(self):
        """Agreement."""
        return self._current_state.agreement

    @property
    def n_outcomes(self):
        """Returns the total number of possible outcomes in the outcome space."""
        return self.nmi.n_outcomes

    @property
    def issues(self) -> list[Issue]:
        """Returns the issues of the outcome space (if defined).

        Will raise an exception if the outcome space has no defined
        issues
        """
        return getattr(self.nmi.outcome_space, "issues")

    @property
    def completed(self):
        """Ended without timing out (either with agreement or broken by a negotiator)"""
        return self.agreement is not None or self._current_state.broken

    @property
    def ended(self):
        """Ended in any way"""
        return self._current_state.ended

    @property
    def n_steps(self):
        """Returns the maximum number of negotiation steps allowed, or None if unlimited."""
        return self.nmi.n_steps

    @property
    def time_limit(self):
        """Returns the maximum negotiation time in seconds, or infinity if unlimited."""
        return self.nmi.time_limit

    @property
    def running(self):
        """Running."""
        return self._current_state.running

    @property
    def dynamic_entry(self):
        """Returns whether negotiators can join/leave during negotiation."""
        return self.nmi.dynamic_entry

    @property
    def max_n_negotiators(self):
        """Max n negotiators."""
        return self.nmi.max_n_negotiators

    @property
    def state4history(self) -> Any:
        """Returns the state as it should be stored in the history."""
        return copy.deepcopy(self._current_state)

    def _add_to_history(self, state4history):
        if len(self._history) == 0:
            self._history.append(state4history)
            return
        last = self._history[-1]
        if last["step"] == state4history:
            self._history[-1] = state4history
            return
        self._history.append(state4history)

    def on_mechanism_error(self) -> None:
        """Called when there is a mechanism error.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        if self._extra_callbacks:
            for a in self.negotiators:
                strt = time.perf_counter()
                self._call(a, a.on_mechanism_error, state=state)
                self._negotiator_times[a.id] += time.perf_counter() - strt

    def on_negotiation_end(self) -> None:
        """Called at the end of each negotiation.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        for a in self.negotiators:
            strt = time.perf_counter()
            self._call(a, a._on_negotiation_end, state=state)
            self._negotiator_times[a.id] += time.perf_counter() - strt
        self.announce(
            Event(
                type="negotiation_end",
                data={
                    "agreement": self.agreement,
                    "state": state,
                    "annotation": self.nmi.annotation,
                },
            )
        )
        self.checkpoint_final_step()

    @property
    def verbosity(self) -> int:
        """
        Verbosity level.

        - Children of this class should only print if verbosity > 1
        """
        return self.__verbosity

    def on_negotiation_start(self) -> bool:
        """Called before starting the negotiation.

        If it returns False then negotiation will end immediately
        """
        return True

    @property
    def atomic_steps(self) -> bool:
        """Is every step corresponding to a single action by a single negotiator"""
        return True

    @abstractmethod
    def __call__(
        self, state: TState, action: dict[str, TAction] | None = None
    ) -> MechanismStepResult[TState]:
        """
        Implements a single step of the mechanism. Override this!

        Args:
            state: The mechanism state. When overriding, set the type of this
                   to the specific `MechanismState` descendent for your mechanism.
            action: An optional action (value) of the next negotiator (key). If given, the call
                   should just execute the action without calling the next negotiator.

        Returns:
            `MechanismStepResult` showing the result of the negotiation step
        """
        ...

    def step(self, action: dict[str, TAction] | None = None) -> TState:
        """Runs a single step of the mechanism.

        Returns:
            MechanismState: The state of the negotiation *after* the round is conducted
            action: An optional action (value) for the next negotiator (key). If given, the call
                   should just execute the action without calling the next negotiator.

        Remarks:

            - Every call yields the results of one round (see `round()`)
            - If the mechanism was yet to start, it will start it and runs one round
            - There is another function (`run()`) that runs the whole mechanism in blocking mode
        """

        if self._start_time is None or self._start_time < 0:
            self._start_time = time.perf_counter()
        if self.__verbosity >= 1:
            if self.current_step == 0:
                print(
                    f"{self.name}: Step {self.current_step} starting after {datetime.now()}",
                    flush=True,
                )
            else:
                _elapsed = time.perf_counter() - self._start_time
                remaining = self.expected_remaining_steps
                etatime = self.expected_remaining_time
                etatime = etatime if etatime is not None else float("inf")
                if remaining is not None:
                    tt = humanize_time(
                        min((_elapsed * remaining) / self.current_step, etatime)
                    )
                    if tt is not None:
                        _eta = tt + f" {remaining} steps"
                    else:
                        _eta = "--"
                else:
                    _eta = "--"
                print(
                    f"{self.name}: Step {self.current_step} starting after {humanize_time(_elapsed, show_ms=True)} [ETA {_eta}]",
                    flush=True,
                    end="\r" if self.verbosity == 1 else "\n",
                )
        self.checkpoint_on_step_started()
        state = self.state
        state4history = self.state4history
        rs, rt = random.random(), 2

        # end with a timeout if condition is met
        current_time = self.time
        if self.__last_second_tried < int(current_time):
            rt, self.__last_second_tried = random.random(), int(current_time)

        if (
            (current_time > self.time_limit)
            or (self.nmi.n_steps and self._current_state.step >= self.nmi.n_steps)
            or current_time > self._hidden_time_limit
            or rs < self.nmi.pend - 1e-8
            or rt < self.nmi.pend_per_second - 1e-8
        ):
            (
                self._current_state.running,
                self._current_state.broken,
                self._current_state.timedout,
            ) = (False, False, True)
            self.on_negotiation_end()
            return self.state

        # if there is a single negotiator and no other negotiators can be added,
        # end without starting
        if len(self._negotiators) < 2:
            if self.nmi.dynamic_entry:
                return self.state
            else:
                (
                    self._current_state.running,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (False, False, False)
                self.on_negotiation_end()
                return self.state

        # if the mechanism states that it is broken, timedout or ended with
        # agreement, report that
        if (
            self._current_state.broken
            or self._current_state.timedout
            or self._current_state.agreement is not None
        ):
            self._current_state.running = False
            self.on_negotiation_end()
            return self.state

        if not self._current_state.running:
            # if we did not start, just start
            self._current_state.running = True
            self._current_state.step = 0
            self._current_state.relative_time = self.relative_time
            self._start_time = time.perf_counter()
            self._current_state.started = True
            state = self.state
            # if the mechanism indicates that it cannot start, keep trying
            if self.on_negotiation_start() is False:
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, False)
                return self.state
            for a in self.negotiators:
                strt = time.perf_counter()
                self._call(a, a._on_negotiation_start, state=state)
                self._negotiator_times[a.id] += time.perf_counter() - strt
            self.announce(Event(type="negotiation_start", data=None))
        else:
            # if no steps are remaining, end with a timeout
            remaining_steps, remaining_time = self.remaining_steps, self.remaining_time
            if (remaining_steps is not None and remaining_steps <= 0) or (
                remaining_time is not None and remaining_time <= 0.0
            ):
                self._current_state.running = False
                (
                    self._current_state.agreement,
                    self._current_state.broken,
                    self._current_state.timedout,
                ) = (None, False, True)
                self.on_negotiation_end()
                return self.state

        # send round start only if the mechanism is not waiting for anyone
        # TODO check this.
        if not self._current_state.waiting and self._extra_callbacks:
            for negotiator in self._negotiators:
                strt = time.perf_counter()
                self._call(negotiator, negotiator.on_round_start, state=state)
                self._negotiator_times[negotiator.id] += time.perf_counter() - strt

        # run a round of the mechanism and get the new state
        step_start = (
            time.perf_counter() if not self._current_state.waiting else self._last_start
        )
        self._last_start = step_start
        self._current_state.waiting = False
        try:
            result = self(self._current_state, action=action)
        except TypeError:
            result = self(self._current_state)
        self._current_state = result.state
        step_time = time.perf_counter() - step_start
        self._stats["round_times"].append(step_time)

        # if negotaitor times are reported, save them
        if result.times:
            for k, v in result.times.items():
                if v is not None:
                    self._stats["times"][k] += v
        # if negotaitor exceptions are reported, save them
        if result.exceptions:
            for k, v in result.exceptions.items():
                if v:
                    self._stats["exceptions"][k] += v

        # update current state variables from the result of the round just run
        (
            self._current_state.has_error,
            self._current_state.error_details,
            self._current_state.waiting,
        ) = (result.state.has_error, result.state.error_details, result.state.waiting)
        if self._current_state.has_error:
            self.on_mechanism_error()
        if (
            self.nmi.step_time_limit is not None
            and step_time > self.nmi.step_time_limit
        ):
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (False, True, None)
        else:
            (
                self._current_state.broken,
                self._current_state.timedout,
                self._current_state.agreement,
            ) = (result.state.broken, result.state.timedout, result.state.agreement)
        if (
            (self._current_state.agreement is not None)
            or self._current_state.broken
            or self._current_state.timedout
        ):
            self._current_state.running = False

        # now switch to the new state
        state = self.state
        if not self._current_state.waiting and result.completed:
            state4history = self.state4history
            if self._extra_callbacks:
                for negotiator in self._negotiators:
                    strt = time.perf_counter()
                    self._call(negotiator, negotiator.on_round_end, state=state)
                    self._negotiator_times[negotiator.id] += time.perf_counter() - strt
            self._add_to_history(state4history)
            # we only indicate a new step if no one is waiting
            self._current_state.step += 1
            self._current_state.time = self.time
            self._current_state.relative_time = self.relative_time

        if not self._current_state.running:
            self.on_negotiation_end()
        return self.state

    def __next__(self) -> TState:
        """next  .

        Returns:
            TState: The result.
        """
        result = self.step()
        if not self._current_state.running:
            raise StopIteration

        return result

    def abort(self) -> TState:
        """Aborts the negotiation."""
        (
            self._current_state.has_error,
            self._current_state.error_details,
            self._current_state.waiting,
        ) = (True, "Uncaught Exception", False)
        self.on_mechanism_error()
        (
            self._current_state.broken,
            self._current_state.timedout,
            self._current_state.agreement,
        ) = (True, False, None)
        state = self.state
        state4history = self.state4history
        self._current_state.running = False
        if self._extra_callbacks:
            for negotiator in self._negotiators:
                strt = time.perf_counter()
                self._call(negotiator, negotiator.on_round_end, state=state)
                self._negotiator_times[negotiator.id] += time.perf_counter() - strt
        self._add_to_history(state4history)
        self._current_state.step += 1
        self.on_negotiation_end()
        return state

    @classmethod
    def runall(
        cls,
        mechanisms: list[Mechanism] | tuple[Mechanism, ...],
        keep_order=True,
        method: str = "ordered",
        ordering: tuple[int, ...] | None = None,
        ordering_fun: Callable[[int, list[TState | None]], int] | None = None,
        completion_callback: Callable[[int, Mechanism], None] | None = None,
        ignore_mechanism_exceptions: bool = False,
    ) -> list[TState | None]:
        """Runs all mechanisms.

        Args:
            mechanisms: list of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step. This is only allowed if the method is ordered
            method: the method to use for running all the sessions.
                    Acceptable options are: sequential, ordered, threads, processes
            ordering: Controls the order of advancing the negotiations with the "ordered" method.
            ordering_fun: A function to implement dynamic ordering for the "ordered" method.
                 This function receives a list of states and returns the index of the next mechanism to step.
                 Note that a state may be None if the corresponding mechanism was None and it should never be stepped
            completion_callback: A callback to be called at the moment each mechanism is ended
            ignore_mechanism_exceptions: If given, mechanisms with exceptions will be treated as ending and running will continue

        Returns:
            - list of states of all mechanisms after completion
            - None for any such states indicates disagreements

        Remarks:
            - sequential means running each mechanism until completion before going to the next
            - ordered means stepping mechanisms in some order which can be controlled by `ordering`. If no ordering is given, the ordering is just round-robin
        """
        if method == "serial":
            warn(
                "`serial`  method is deprecated. Please use 'ordered' instead.",
                DeprecationWarning,
            )
            method = "ordered"
        if method == "sequential":
            if not keep_order:
                mechanisms = [_ for _ in mechanisms]
                random.shuffle(mechanisms)
            for i, mechanism in enumerate(mechanisms):
                try:
                    mechanism.run()
                except Exception as e:
                    if not ignore_mechanism_exceptions:
                        raise e
                    mechanism.state.has_error = True
                    mechanism.state.error_details = str(e)

                if completion_callback:
                    completion_callback(i, mechanism)
            states = [_.state for _ in mechanisms]
        elif method == "ordered":
            completed = [_ is None for _ in mechanisms]
            states: list[TState | None] = [None] * len(mechanisms)
            allindices = list(range(len(list(mechanisms))))
            indices = allindices if not ordering else list(ordering)
            notmentioned = set(allindices).difference(indices)
            assert (
                len(notmentioned) == 0
            ), f"Mechanisms {notmentioned} are never mentioned in the ordering."
            if ordering_fun:
                j = 0
                while not all(completed):
                    states = [_.state for _ in mechanisms]
                    i = ordering_fun(j, states)
                    j += 1
                    mechanism = mechanisms[i]
                    if completed[i]:
                        continue
                    try:
                        result = mechanism.step()
                    except Exception as e:
                        if not ignore_mechanism_exceptions:
                            raise e
                        mechanism.state.has_error = True
                        mechanism.state.error_details = str(e)
                        completed[i] = True
                        if completion_callback:
                            completion_callback(i, mechanism)
                        result = mechanism.state
                    if result.running:
                        continue
                    completed[i] = True
                    if completion_callback:
                        completion_callback(i, mechanism)
            else:
                while not all(completed):
                    if not keep_order:
                        random.shuffle(indices)
                    for i in indices:
                        mechanism = mechanisms[i]
                        if completed[i]:
                            continue
                        try:
                            result = mechanism.step()
                        except Exception as e:
                            if not ignore_mechanism_exceptions:
                                raise e
                            mechanism.state.has_error = True
                            mechanism.state.error_details = str(e)
                            completed[i] = True
                            if completion_callback:
                                completion_callback(i, mechanism)
                            result = mechanism.state
                        if result.running:
                            continue
                        completed[i] = True
                        if completion_callback:
                            completion_callback(i, mechanism)
                        states[i] = mechanism.state
                        if all(completed):
                            break
        elif method == "threads":
            raise NotImplementedError()
        elif method == "processes":
            raise NotImplementedError()
        else:
            raise ValueError(
                f"method {method} is unknown. Acceptable options are ordered, sequential, threads, processes"
            )
        return states

    @classmethod
    def stepall(
        cls, mechanisms: list[Mechanism] | tuple[Mechanism, ...], keep_order=True
    ) -> list[TState]:
        """Step all mechanisms.

        Args:
            mechanisms: list of mechanisms
            keep_order: if True, the mechanisms will be run in order every step otherwise the order will be randomized
                        at every step

        Returns:
            - list of states of all mechanisms after completion
        """
        indices = list(range(len(list(mechanisms))))
        if not keep_order:
            random.shuffle(indices)

        completed = [_ is None for _ in mechanisms]
        for i in indices:
            done, mechanism = completed[i], mechanisms[i]
            if done:
                continue
            result = mechanism.step()
            if result.running:
                continue
            completed[i] = True
        return [_.state for _ in mechanisms]

    def run_with_progress(self, timeout=None) -> TState:
        """Run with progress.

        Args:
            timeout: Timeout.

        Returns:
            TState: The result.
        """
        from rich.progress import Progress

        if timeout is None:
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    progress.update(task, completed=int(self.relative_time * 100))
        else:
            start_time = time.perf_counter()
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    progress.update(task, completed=int(self.relative_time * 100))
                    if time.perf_counter() - start_time > timeout:
                        (
                            self._current_state.running,
                            self._current_state.timedout,
                            self._current_state.broken,
                        ) = (False, True, False)
                        self.on_negotiation_end()
                        break
        return self.state

    def run(self, timeout=None) -> TState:
        """Execute the negotiation mechanism until completion or timeout.

        Args:
            timeout: Maximum time in seconds to run, or None for no limit

        Returns:
            TState: Final negotiation state after completion
        """
        if timeout is None:
            for _ in self:
                pass
        else:
            start_time = time.perf_counter()
            for _ in self:
                if time.perf_counter() - start_time > timeout:
                    (
                        self._current_state.running,
                        self._current_state.timedout,
                        self._current_state.broken,
                    ) = (False, True, False)
                    self.on_negotiation_end()
                    break
        return self.state

    @property
    def history(self) -> list[TState]:
        """Complete history of mechanism states throughout the negotiation.

        Returns:
            list[TState]: Chronological list of all negotiation states from start to current
        """
        return self._history

    @property
    def stats(self):
        """Mechanism statistics collected during negotiation (e.g., step counts, agreement metrics)."""
        return self._stats

    @property
    def current_step(self):
        """Returns the current negotiation step number (0-indexed)."""
        return self._current_state.step

    def _get_preferences(self) -> list[UtilityFunction]:
        preferences = []
        for a in self.negotiators:
            preferences.append(a.preferences)
        return preferences

    def _pareto_frontier(
        self,
        method: Callable,
        max_cardinality: int | float = float("inf"),
        sort_by_welfare=True,
    ) -> tuple[list[tuple[float, ...]], list[Outcome]]:
        ufuns = tuple(self._get_preferences())
        if any(_ is None for _ in ufuns):
            raise ValueError(
                "Some negotiators have no ufuns. Cannot calcualate the pareto frontier"
            )
        outcomes = self.discrete_outcomes(max_cardinality=max_cardinality)
        points = [tuple(ufun(outcome) for ufun in ufuns) for outcome in outcomes]
        reservs = tuple(
            _.reserved_value if _ is not None else float("-inf") for _ in ufuns
        )
        rational_indices = [
            i for i, _ in enumerate(points) if all(a >= b for a, b in zip(_, reservs))
        ]
        points = [points[_] for _ in rational_indices]
        indices = list(method(points, sort_by_welfare=sort_by_welfare))
        frontier = [points[_] for _ in indices]
        if frontier is None:
            raise ValueError("Could not find the pareto-frontier")
        return frontier, [outcomes[rational_indices[_]] for _ in indices]

    def pareto_frontier_bf(
        self, max_cardinality: float = float("inf"), sort_by_welfare=True
    ) -> tuple[list[tuple[float, ...]], list[Outcome]]:
        """Pareto frontier bf.

        Args:
            max_cardinality: Max cardinality.
            sort_by_welfare: Sort by welfare.

        Returns:
            tuple[list[tuple[float, ...]], list[Outcome]]: The result.
        """
        return self._pareto_frontier(
            pareto_frontier_bf, max_cardinality, sort_by_welfare
        )

    def pareto_frontier(
        self, max_cardinality: float = float("inf"), sort_by_welfare=True
    ) -> tuple[tuple[tuple[float, ...], ...], list[Outcome]]:
        """Pareto frontier.

        Args:
            max_cardinality: Max cardinality.
            sort_by_welfare: Sort by welfare.

        Returns:
            tuple[tuple[tuple[float, ...], ...], list[Outcome]]: The result.
        """
        ufuns = tuple(self._get_preferences())
        if any(_ is None for _ in ufuns):
            raise ValueError(
                "Some negotiators have no ufuns. Cannot calculate the pareto frontier"
            )
        outcomes = self.discrete_outcomes(max_cardinality=max_cardinality)
        results = pareto_frontier(
            ufuns,
            outcomes=outcomes,
            n_discretization=None,
            max_cardinality=float("inf"),
            sort_by_welfare=sort_by_welfare,
        )
        return results[0], [outcomes[_] for _ in results[1]]

    def max_welfare_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...], ...] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        """Max welfare points.

        Args:
            max_cardinality: Max cardinality.
            frontier: Frontier.
            frontier_outcomes: Frontier outcomes.

        Returns:
            tuple[tuple[tuple[float, ...], Outcome], ...]: The result.
        """
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = max_welfare_points(
            ufuns, frontier, outcome_space=self.outcome_space
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def max_relative_welfare_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...], ...] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        """Max relative welfare points.

        Args:
            max_cardinality: Max cardinality.
            frontier: Frontier.
            frontier_outcomes: Frontier outcomes.

        Returns:
            tuple[tuple[tuple[float, ...], Outcome], ...]: The result.
        """
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = max_relative_welfare_points(
            ufuns, frontier, outcome_space=self.outcome_space
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def modified_kalai_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...], ...] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        """Modified kalai points.

        Args:
            max_cardinality: Max cardinality.
            frontier: Frontier.
            frontier_outcomes: Frontier outcomes.

        Returns:
            tuple[tuple[tuple[float, ...], Outcome], ...]: The result.
        """
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = kalai_points(
            ufuns,
            frontier,
            outcome_space=self.outcome_space,
            subtract_reserved_value=False,
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def kalai_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...], ...] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        """Kalai points.

        Args:
            max_cardinality: Max cardinality.
            frontier: Frontier.
            frontier_outcomes: Frontier outcomes.

        Returns:
            tuple[tuple[tuple[float, ...], Outcome], ...]: The result.
        """
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        kalai_pts = kalai_points(
            ufuns,
            frontier,
            outcome_space=self.outcome_space,
            subtract_reserved_value=True,
        )
        return tuple(
            (kalai_utils, frontier_outcomes[indx]) for kalai_utils, indx in kalai_pts
        )

    def nash_points(
        self,
        max_cardinality: float = float("inf"),
        frontier: tuple[tuple[float, ...], ...] | None = None,
        frontier_outcomes: list[Outcome] | None = None,
    ) -> tuple[tuple[tuple[float, ...], Outcome], ...]:
        """Nash points.

        Args:
            max_cardinality: Max cardinality.
            frontier: Frontier.
            frontier_outcomes: Frontier outcomes.

        Returns:
            tuple[tuple[tuple[float, ...], Outcome], ...]: The result.
        """
        ufuns = self._get_preferences()
        if not frontier:
            frontier, frontier_outcomes = self.pareto_frontier(max_cardinality)
        assert frontier_outcomes is not None
        # outcomes = tuple(self.discrete_outcomes(max_cardinality=max_cardinality))
        nash_pts = nash_points(ufuns, frontier, outcome_space=self.outcome_space)
        return tuple(
            (nash_utils, frontier_outcomes[indx]) for nash_utils, indx in nash_pts
        )

    def to_completed_run(
        self, source: str = "history", metadata: dict[str, Any] | None = None
    ) -> CompletedRun:
        """
        Creates a CompletedRun object from the current mechanism state.

        Args:
            source: The source of the history information. Options:
                - "history": Use the history attribute (list of states)
                - "full_trace": Use the full_trace attribute (if available)
                - "trace": Use the trace attribute (if available)
                - "extended_trace": Use the extended_trace attribute (if available)
            metadata: Arbitrary metadata to include in the CompletedRun.

        Returns:
            CompletedRun: A CompletedRun object containing the negotiation data.
        """
        from negmas.inout import Scenario

        # Get the trace data based on source
        if source == "full_trace" and hasattr(self, "full_trace"):
            trace_data = self.full_trace  # type: ignore
            history_type = "full_trace"
        elif source == "trace" and hasattr(self, "trace"):
            trace_data = self.trace  # type: ignore
            history_type = "trace"
        elif source == "extended_trace" and hasattr(self, "extended_trace"):
            trace_data = self.extended_trace  # type: ignore
            history_type = "extended_trace"
        else:
            trace_data = self.history
            history_type = "history"

        # Create scenario if possible
        scenario = None
        try:
            ufuns = tuple(
                n.preferences for n in self.negotiators if n.preferences is not None
            )
            if ufuns and self.outcome_space is not None:
                scenario = Scenario(
                    outcome_space=self.outcome_space,  # type: ignore
                    ufuns=ufuns,  # type: ignore
                    mechanism_type=self.__class__,
                    mechanism_params=self.params,
                )
        except Exception:
            scenario = None

        # Build config (only include YAML-serializable values)
        config = self.make_config()
        # Build outcome stats (basic outcome information)
        utilities: list[float | None] = []
        for n in self.negotiators:
            if n.ufun is not None and self.agreement is not None:
                try:
                    u = n.ufun(self.agreement)
                    utilities.append(float(u) if u is not None else None)
                except Exception:
                    utilities.append(None)
            else:
                utilities.append(None)

        outcome_stats = {
            "agreement": self.agreement,
            "broken": self.state.broken,
            "timedout": self.state.timedout,
            "utilities": utilities,
        }

        return CompletedRun(
            history=trace_data,
            history_type=history_type,
            scenario=scenario,
            agreement=self.agreement,
            agreement_stats=None,  # Agreement stats require scenario stats to be calculated
            outcome_stats=outcome_stats,
            config=config,
            metadata=metadata or {},
        )

    def make_config(self) -> dict[str, Any]:
        """Create a YAML-serializable configuration dict from the mechanism state.

        Returns:
            A dictionary containing mechanism configuration that can be safely
            serialized to YAML.
        """
        return dict(
            mechanism_type=self.__class__.__name__,
            n_negotiators=len(self.negotiators),
            negotiator_names=[n.name for n in self.negotiators],
            negotiator_types=[n.__class__.__name__ for n in self.negotiators],
            negotiator_ids=[n.id for n in self.negotiators],
            final_step=self.current_step,
            final_time=self.time,
            final_relative_time=self.relative_time,
            broken=self.state.broken,
            timedout=self.state.timedout,
            has_error=self.state.has_error,
            n_steps=self.nmi.n_steps,
            time_limit=self.nmi.time_limit,
            pend=self.nmi.pend,
            pend_per_second=self.nmi.pend_per_second,
            step_time_limit=self.nmi.step_time_limit,
            negotiator_time_limit=self.nmi.negotiator_time_limit,
            hidden_time_limit=self._hidden_time_limit,
            max_n_negotiators=self.max_n_negotiators,
            dynamic_entry=self.dynamic_entry,
            name=self.name,
            genius_port=self.genius_port,
            id=self.id,
            type_name=self.type_name,
            verbosity=self.verbosity,
            ignore_negotiator_exceptions=self.ignore_negotiator_exceptions,
        )

    def save(
        self,
        parent: Path | str,
        name: str,
        single_file: bool = False,
        per_negotiator: bool = False,
        save_scenario: bool = True,
        save_scenario_stats: bool = False,
        save_agreement_stats: bool = True,
        save_config: bool = True,
        source: str = "history",
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
        warn_if_existing: bool = True,
        storage_format: TableStorageFormat | None = DEFAULT_TABLE_STORAGE_FORMAT,
    ) -> Path:
        """
        Saves the negotiation in a standard NegMAS format.

        Args:
            parent: Where to save the negotiation
            name: Name to save to. It can be a directory name or a file name (without extension)
            single_file: If True, a single file with the negotiation history/trace will be saved. all save_* arguments will be ignored. The extension will be decided based on storage_format
            per_negotiator: If True, save traces per negotiator in a subdirectory.
                Files are named {type}@{index}_{name}.{ext}. Only works with full_trace source.
            save_scenario: If True (and single_file is False), save the negotiation scenario information with history (under a `scenario` folder)
            save_scenario_stats: If True (and single_file is False), save scenario stats and info (using Scenario.dumpas)
            save_agreement_stats: If True (and single_file is False), save agreement stats, the agreement itself and its utilities under `outcome_stats.yaml`.
                                  If utility functions are not found in negotiators, only the agreement will be saved there
            save_config: If True (and single_file is False),  save the configuration paramters of the mechanism run.
            source: The source of the history information to save. Default is "history" which means using the history attribute.
            metadata: arbitrary data to save under `meta_data.yaml`. Only supported if single_file is False
            overwrite: Overwrite existing files/folders
            warn_if_existing: Warn if existing files/folders are found
            storage_format: The format for storing tables.

        Returns:
            Path: The path to the saved file or directory.
        """
        completed_run = self.to_completed_run(source=source, metadata=metadata)
        return completed_run.save(
            parent=parent,
            name=name,
            single_file=single_file,
            per_negotiator=per_negotiator,
            save_scenario=save_scenario,
            save_scenario_stats=save_scenario_stats,
            save_agreement_stats=save_agreement_stats,
            save_config=save_config,
            overwrite=overwrite,
            warn_if_existing=warn_if_existing,
            storage_format=storage_format,
        )

    def plot(self, **kwargs) -> Any:
        """A method for plotting a negotiation session."""
        _ = kwargs

    def __iter__(self):
        """iter  ."""
        return self

    def __str__(self):
        """str  ."""
        d = self.__dict__.copy()
        return pprint.pformat(d)

    __repr__ = __str__


@runtime_checkable
class Traceable(Protocol):
    """A mechanism that can generate a trace"""

    @property
    def full_trace_with_utils(self) -> list[tuple]:
        """Returns the full trace and the utility of the negotiators at each step."""
        ...

    @property
    def full_trace(self) -> list[TraceElement]:
        """Returns the negotiation history as a list of relative_time/step/negotiator/offer tuples"""
        ...

    @property
    def trace(self) -> list[tuple[str, Outcome]]:
        """Basic trace keeping only outcomes"""
        ...

    def extended_trace(self) -> list[tuple[int, str, Outcome]]:
        """Returns the negotiation history as a list of step/negotiator/offer tuples"""
        ...

    def negotiator_full_trace(
        self, negotiator_id: str
    ) -> list[tuple[float, float, int, Outcome, str]]:
        """Returns the (time/relative-time/step/outcome/response) given by a negotiator (in order)"""
        ...
