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
        include_pareto_frontier: bool = False,
        storage_format: TableStorageFormat | None = DEFAULT_TABLE_STORAGE_FORMAT,
    ) -> Path:
        """Saves the completed run to disk.

        Args:
            parent: Parent directory where to save the run.
            name: Name for the saved run (directory name or file name without extension).
            single_file: If True, save only the trace as a single file.
            per_negotiator: Deprecated. This parameter is ignored. Per-negotiator traces
                are available in the main trace file grouped by negotiator.
            save_scenario: If True, save the scenario information.
            save_scenario_stats: If True, save scenario statistics.
            save_agreement_stats: If True, save agreement statistics.
            save_config: If True, save the configuration.
            overwrite: If True, overwrite existing files/directories.
            warn_if_existing: If True, warn when overwriting.
            include_pareto_frontier: If True the pareto frontier will be included in the scenario (if save_scenario)
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
        elif self.history_type == "full_trace_with_utils":
            # full_trace_with_utils has TRACE_ELEMENT_MEMBERS + negotiator utility columns
            # Use negotiator_ids for consistency with the 'negotiator' field in trace entries
            # (which uses IDs, not names). Fall back to names if IDs not available.
            negotiator_ids = self.config.get("negotiator_ids", [])
            if not negotiator_ids:
                negotiator_ids = self.config.get("negotiator_names", [])
            if not negotiator_ids:
                # Fallback: infer from first row
                if self.history and len(self.history) > 0:
                    first_row = self.history[0]
                    n_extra_cols = len(first_row) - len(TRACE_ELEMENT_MEMBERS)
                    negotiator_ids = [f"utility_{i}" for i in range(n_extra_cols)]
            trace_columns = TRACE_ELEMENT_MEMBERS + negotiator_ids
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

        # per_negotiator is deprecated - emit warning if True
        if per_negotiator:
            from negmas.warnings import deprecated

            deprecated(
                "per_negotiator parameter is deprecated and ignored. "
                "Per-negotiator traces are available in the main trace file grouped by negotiator."
            )

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
                    include_pareto_frontier=include_pareto_frontier,
                )
            except Exception:
                # If scenario saving fails, save basic info
                scenario_dir.mkdir(parents=True, exist_ok=True)
                dump(
                    {"outcome_space": str(self.scenario.outcome_space)},
                    scenario_dir / "outcome_space.yaml",
                )

        # Save outcome stats (always saved in directory mode)
        # Include agreement stats in outcome_stats for a complete record
        outcome_stats_to_save = dict(self.outcome_stats) if self.outcome_stats else {}
        if save_agreement_stats and self.agreement_stats is not None:
            # Merge agreement stats into outcome_stats
            outcome_stats_to_save.update(
                {
                    "pareto_optimality": self.agreement_stats.pareto_optimality,
                    "nash_optimality": self.agreement_stats.nash_optimality,
                    "kalai_optimality": self.agreement_stats.kalai_optimality,
                    "modified_kalai_optimality": self.agreement_stats.modified_kalai_optimality,
                    "max_welfare_optimality": self.agreement_stats.max_welfare_optimality,
                    "ks_optimality": self.agreement_stats.ks_optimality,
                    "modified_ks_optimality": self.agreement_stats.modified_ks_optimality,
                }
            )
        if outcome_stats_to_save:
            dump(outcome_stats_to_save, save_dir / "outcome_stats.yaml")

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

        Remarks:
            When loading scenarios, we will look at a scenario folder in the given path and if not
            for a scenario_path in the metadata

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

        # Load metadata if present
        metadata_path = path / "metadata.yaml"
        metadata = load(metadata_path) if metadata_path.exists() else {}

        # Load scenario if present and requested
        scenario = None
        if load_scenario:
            scenario_dir = path / "scenario"
            if not scenario_dir.exists():
                scenario_dir = metadata.get("scenario_path", None)
                if scenario_dir is not None:
                    scenario_dir = Path(scenario_dir)
            if scenario_dir is not None and scenario_dir.exists():
                try:
                    scenario = Scenario.load(
                        scenario_dir, load_stats=load_scenario_stats
                    )
                except Exception:
                    scenario = None

        # Load config if present and requested
        config: dict[str, Any] = {}
        if load_config:
            config_path = path / "config.yaml"
            config = load(config_path) if config_path.exists() else {}

        # Load outcome stats if present
        outcome_stats_path = path / "outcome_stats.yaml"
        outcome_stats = load(outcome_stats_path) if outcome_stats_path.exists() else {}

        # If agreement was not in run_info, try to get it from outcome_stats
        if agreement is None and outcome_stats:
            agreement = outcome_stats.get("agreement")

        # Extract agreement_stats from outcome_stats if present
        agreement_stats = None
        if load_agreement_stats and outcome_stats:
            # Check if outcome_stats contains optimality fields
            optimality_fields = [
                "pareto_optimality",
                "nash_optimality",
                "kalai_optimality",
                "modified_kalai_optimality",
                "max_welfare_optimality",
                "ks_optimality",
                "modified_ks_optimality",
            ]
            if any(field in outcome_stats for field in optimality_fields):
                agreement_stats = OutcomeOptimality(
                    pareto_optimality=outcome_stats.get(
                        "pareto_optimality", float("nan")
                    ),
                    nash_optimality=outcome_stats.get("nash_optimality", float("nan")),
                    kalai_optimality=outcome_stats.get(
                        "kalai_optimality", float("nan")
                    ),
                    modified_kalai_optimality=outcome_stats.get(
                        "modified_kalai_optimality", float("nan")
                    ),
                    max_welfare_optimality=outcome_stats.get(
                        "max_welfare_optimality", float("nan")
                    ),
                    ks_optimality=outcome_stats.get("ks_optimality", float("nan")),
                    modified_ks_optimality=outcome_stats.get(
                        "modified_ks_optimality", float("nan")
                    ),
                )

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

    def convert(
        self, target: str, scenario: "Scenario | None" = None
    ) -> "CompletedRun[TState]":
        """Convert the history to a different trace format.

        This method converts the current history to a different format. The conversion
        follows these rules:
        - Any format can be converted to a simpler format (losing information)
        - Converting to a more detailed format will set missing fields to None
        - Converting to `full_trace_with_utils` requires a scenario with utility functions
          (either from `self.scenario` or the provided `scenario` parameter)

        Format hierarchy (most detailed to least detailed):
            full_trace_with_utils > full_trace > extended_trace > trace > history

        Args:
            target: The target format. One of: "full_trace_with_utils", "full_trace",
                "extended_trace", "trace", "history".
            scenario: Optional scenario with utility functions. If provided, it will be
                used for computing utilities when converting to `full_trace_with_utils`.
                If not provided, `self.scenario` will be used.

        Returns:
            A new CompletedRun with the converted history.

        Raises:
            ValueError: If target is not a valid format.
            ValueError: If converting to full_trace_with_utils without a scenario with ufuns.

        Examples:
            >>> from negmas.sao import SAOMechanism, RandomNegotiator
            >>> m = SAOMechanism(outcomes=[(i,) for i in range(5)], n_steps=5)
            >>> _ = m.add(RandomNegotiator(name="n1"))
            >>> _ = m.add(RandomNegotiator(name="n2"))
            >>> _ = m.run()
            >>> # Create a CompletedRun with full_trace
            >>> run = m.to_completed_run(source="full_trace")
            >>> run.history_type
            'full_trace'
            >>> # Convert to trace format
            >>> converted = run.convert("trace")
            >>> converted.history_type
            'trace'
            >>> # Convert to extended_trace
            >>> converted2 = run.convert("extended_trace")
            >>> converted2.history_type
            'extended_trace'
        """

        valid_targets = {
            "full_trace_with_utils",
            "full_trace",
            "extended_trace",
            "trace",
            "history",
        }
        if target not in valid_targets:
            raise ValueError(
                f"Invalid target format: {target}. Must be one of: {valid_targets}"
            )

        # If already in target format, return a copy
        if self.history_type == target:
            return CompletedRun(
                history=list(self.history),
                history_type=self.history_type,
                scenario=self.scenario,
                agreement=self.agreement,
                agreement_stats=self.agreement_stats,
                outcome_stats=dict(self.outcome_stats),
                config=dict(self.config),
                metadata=dict(self.metadata),
            )

        # Use provided scenario or self.scenario
        effective_scenario = scenario if scenario is not None else self.scenario

        # For full_trace_with_utils, we need a scenario with ufuns
        if target == "full_trace_with_utils":
            if effective_scenario is None or not effective_scenario.ufuns:
                raise ValueError(
                    "Cannot convert to full_trace_with_utils without a scenario with "
                    "utility functions. Provide a scenario parameter or ensure "
                    "self.scenario has ufuns."
                )

        # Convert the history
        new_history = self._convert_history(
            self.history, self.history_type, target, effective_scenario
        )

        return CompletedRun(
            history=new_history,
            history_type=target,
            scenario=self.scenario,  # Keep original scenario
            agreement=self.agreement,
            agreement_stats=self.agreement_stats,
            outcome_stats=dict(self.outcome_stats),
            config=dict(self.config),
            metadata=dict(self.metadata),
        )

    def _convert_history(
        self,
        history: list,
        source_type: str,
        target_type: str,
        scenario: "Scenario | None",
    ) -> list:
        """Convert history from one format to another.

        Args:
            history: The source history data.
            source_type: The source format type.
            target_type: The target format type.
            scenario: Optional scenario for utility computation.

        Returns:
            The converted history in the target format.
        """
        # First, normalize to a common intermediate format (list of dicts with all fields)
        normalized = self._normalize_history(history, source_type)

        # Then convert to target format
        return self._denormalize_history(normalized, target_type, scenario)

    def _normalize_history(self, history: list, source_type: str) -> list[dict]:
        """Normalize history to a list of dicts with all possible fields.

        Missing fields are set to None.
        """
        normalized = []

        for entry in history:
            record: dict = {
                "time": None,
                "relative_time": None,
                "step": None,
                "negotiator": None,
                "offer": None,
                "responses": None,
                "state": None,
                "text": None,
                "data": None,
            }

            if source_type == "full_trace_with_utils":
                # TraceElement + utility columns as tuple
                if hasattr(entry, "_asdict"):
                    d = entry._asdict()
                elif isinstance(entry, dict):
                    d = entry
                elif isinstance(entry, (list, tuple)):
                    # Assume TRACE_ELEMENT_MEMBERS order + utilities
                    d = dict(
                        zip(TRACE_ELEMENT_MEMBERS, entry[: len(TRACE_ELEMENT_MEMBERS)])
                    )
                else:
                    d = {}
                record.update({k: v for k, v in d.items() if k in record})
                # Store utilities separately (they'll be recomputed if needed)

            elif source_type == "full_trace":
                # TraceElement namedtuple or dict
                if hasattr(entry, "_asdict"):
                    d = entry._asdict()
                elif isinstance(entry, dict):
                    d = entry
                elif isinstance(entry, (list, tuple)):
                    d = dict(zip(TRACE_ELEMENT_MEMBERS, entry))
                else:
                    d = {}
                record.update({k: v for k, v in d.items() if k in record})

            elif source_type == "extended_trace":
                # (step, negotiator, offer) tuple or dict
                if isinstance(entry, dict):
                    record["step"] = entry.get("step")
                    record["negotiator"] = entry.get("negotiator")
                    record["offer"] = entry.get("offer")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    record["step"] = entry[0]
                    record["negotiator"] = entry[1]
                    record["offer"] = entry[2]

            elif source_type == "trace":
                # (negotiator, offer) tuple or dict
                if isinstance(entry, dict):
                    record["negotiator"] = entry.get("negotiator")
                    record["offer"] = entry.get("offer")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    record["negotiator"] = entry[0]
                    record["offer"] = entry[1]

            else:  # "history" - MechanismState dicts
                if hasattr(entry, "asdict"):
                    d = entry.asdict()
                elif hasattr(entry, "_asdict"):
                    d = entry._asdict()
                elif isinstance(entry, dict):
                    d = entry
                else:
                    d = {}
                # Extract what we can from state
                record["step"] = d.get("step")
                record["time"] = d.get("time")
                record["relative_time"] = d.get("relative_time")
                # For history format, the whole entry is the state
                record["state"] = d

            normalized.append(record)

        return normalized

    def _denormalize_history(
        self, normalized: list[dict], target_type: str, scenario: "Scenario | None"
    ) -> list:
        """Convert normalized history to target format."""
        result = []

        for record in normalized:
            if target_type == "full_trace_with_utils":
                # Create TraceElement + utilities
                trace_elem = TraceElement(
                    time=record["time"],
                    relative_time=record["relative_time"],
                    step=record["step"],
                    negotiator=record["negotiator"],
                    offer=record["offer"],
                    responses=record["responses"],
                    state=record["state"],
                    text=record["text"],
                    data=record["data"],
                )
                # Compute utilities if we have a scenario and offer
                utilities: list = []
                if (
                    scenario is not None
                    and scenario.ufuns
                    and record["offer"] is not None
                ):
                    for ufun in scenario.ufuns:
                        try:
                            u = ufun(record["offer"])
                            utilities.append(float(u) if u is not None else None)
                        except Exception:
                            utilities.append(None)
                else:
                    # No scenario or no offer, fill with None
                    n_negotiators = (
                        len(scenario.ufuns) if scenario and scenario.ufuns else 0
                    )
                    utilities = [None] * n_negotiators

                # Return as tuple: TraceElement fields + utilities
                result.append(tuple(trace_elem) + tuple(utilities))

            elif target_type == "full_trace":
                # Create TraceElement
                result.append(
                    TraceElement(
                        time=record["time"],
                        relative_time=record["relative_time"],
                        step=record["step"],
                        negotiator=record["negotiator"],
                        offer=record["offer"],
                        responses=record["responses"],
                        state=record["state"],
                        text=record["text"],
                        data=record["data"],
                    )
                )

            elif target_type == "extended_trace":
                # (step, negotiator, offer)
                result.append((record["step"], record["negotiator"], record["offer"]))

            elif target_type == "trace":
                # (negotiator, offer)
                result.append((record["negotiator"], record["offer"]))

            else:  # "history"
                # Return the state dict if available, otherwise create minimal dict
                if record["state"] is not None:
                    result.append(record["state"])
                else:
                    # Create a minimal state-like dict
                    result.append(
                        {
                            "step": record["step"],
                            "time": record["time"],
                            "relative_time": record["relative_time"],
                        }
                    )

        return result

    @classmethod
    def infer_source(cls, path: Path | str) -> str:
        """Infer the source type from a saved negotiation file or folder.

        Args:
            path: Path to a file or directory containing saved negotiation data.

        Returns:
            The inferred source type: "history", "trace", "extended_trace",
            "full_trace", or "full_trace_with_utils".

        Raises:
            FileNotFoundError: If the path does not exist or no trace file is found.

        Examples:
            >>> from pathlib import Path
            >>> from negmas.sao import SAOMechanism, RandomNegotiator
            >>> import tempfile
            >>> import shutil
            >>> # Create and run a simple negotiation
            >>> m = SAOMechanism(outcomes=[(i,) for i in range(5)], n_steps=5)
            >>> _ = m.add(RandomNegotiator(name="n1"))
            >>> _ = m.add(RandomNegotiator(name="n2"))
            >>> _ = m.run()
            >>> # Save as full_trace
            >>> tmpdir = Path(tempfile.mkdtemp())
            >>> completed = m.to_completed_run(source="full_trace")
            >>> save_path = completed.save(tmpdir, "test", single_file=False)
            >>> # Infer the source type
            >>> inferred = CompletedRun.infer_source(save_path)
            >>> inferred == "full_trace"
            True
            >>> shutil.rmtree(tmpdir)
        """
        from negmas.helpers.inout import load, load_table

        path = Path(path).expanduser().absolute()

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # If directory, check for run_info.yaml first
        if path.is_dir():
            run_info_path = path / "run_info.yaml"
            if run_info_path.exists():
                run_info = load(run_info_path)
                return run_info.get("history_type", "history")

            # Find trace file in directory
            for ext in [".csv", ".csv.gz", ".parquet"]:
                trace_file = path / f"trace{ext}"
                if trace_file.exists():
                    path = trace_file
                    break
            else:
                return "history"  # fallback if no trace file found

        # Load first few rows to inspect columns
        history = load_table(path, as_dataframe=False)

        if not history:
            return "history"  # empty file, default

        first_row = history[0]
        if not isinstance(first_row, dict):
            return "history"  # can't determine from non-dict data

        keys = set(first_row.keys())

        # Define column sets
        full_trace_core_cols = {
            "time",
            "relative_time",
            "step",
            "negotiator",
            "offer",
            "state",
            "responses",
        }
        trace_cols = {"negotiator", "offer"}
        extended_trace_cols = {"step", "negotiator", "offer"}

        # Check for full_trace_with_utils (has extra utility columns beyond TRACE_ELEMENT_MEMBERS)
        # TRACE_ELEMENT_MEMBERS has 9 standard columns
        # If we have more than that and have the core columns, it's full_trace_with_utils
        trace_element_cols = set(TRACE_ELEMENT_MEMBERS)
        if full_trace_core_cols.issubset(keys) and len(keys) > len(trace_element_cols):
            # Has core columns plus extra columns (likely utility columns)
            return "full_trace_with_utils"
        elif full_trace_core_cols.issubset(keys):
            return "full_trace"
        elif keys == extended_trace_cols:
            return "extended_trace"
        elif keys == trace_cols:
            return "trace"
        else:
            return "history"

    def plot(
        self,
        plotting_negotiators: tuple[int, int] | tuple[str, str] = (0, 1),
        save_fig: bool = False,
        path: str | None = None,
        fig_name: str | None = None,
        image_format: str = "webp",
        ignore_none_offers: bool = True,
        with_lines: bool = True,
        show_agreement: bool = False,
        show_pareto_distance: bool = True,
        show_nash_distance: bool = True,
        show_kalai_distance: bool = True,
        show_ks_distance: bool = True,
        show_max_welfare_distance: bool = True,
        show_max_relative_welfare_distance: bool = False,
        show_end_reason: bool = True,
        show_annotations: bool = False,
        show_reserved: bool = True,
        show_total_time: bool = True,
        show_relative_time: bool = True,
        show_n_steps: bool = True,
        colors: list | None = None,
        markers: list[str] | None = None,
        colormap: str = "tab10",
        ylimits: tuple[float, float] | None = None,
        common_legend: bool = True,
        extra_annotation: str = "",
        xdim: str = "relative_time",
        colorizer: Any | None = None,
        only2d: bool = False,
        no2d: bool = False,
        fast: bool = False,
        simple_offers_view: bool = False,
        mark_offers_view: bool = True,
        mark_pareto_points: bool = True,
        mark_all_outcomes: bool = True,
        mark_nash_points: bool = True,
        mark_kalai_points: bool = True,
        mark_ks_points: bool = True,
        mark_max_welfare_points: bool = True,
        show: bool = True,
    ):
        """Visualize the completed negotiation run showing offers and utilities in 2D space.

        This method produces the same visualization as SAOMechanism.plot() but works with
        saved/completed runs. It requires that the CompletedRun has a scenario with utility
        functions, and that the history is in 'full_trace' format.

        Args:
            plotting_negotiators: Indices or IDs of two negotiators whose utilities form the axes.
            save_fig: Whether to save the figure to disk.
            path: Directory path for saving the figure.
            fig_name: Filename for the saved figure.
            image_format: Image format to use (default: "webp"). Supported: webp, png, jpg, svg, pdf.
            ignore_none_offers: Whether to skip None offers in the plot.
            with_lines: Whether to connect offers with lines showing progression.
            show_agreement: Whether to highlight the final agreement point.
            show_pareto_distance: Whether to display distance to Pareto frontier.
            show_nash_distance: Whether to display distance to Nash solution.
            show_kalai_distance: Whether to display distance to Kalai-Smorodinsky solution.
            show_ks_distance: Whether to display distance to KS point.
            show_max_welfare_distance: Whether to display distance to max welfare point.
            show_max_relative_welfare_distance: Whether to display distance to max relative welfare.
            show_end_reason: Whether to annotate why the negotiation ended.
            show_annotations: Whether to show offer annotations on the plot.
            show_reserved: Whether to show reservation values.
            show_total_time: Whether to display total negotiation time.
            show_relative_time: Whether to display relative time progress.
            show_n_steps: Whether to display the number of steps.
            colors: Custom color list for negotiators.
            markers: Custom marker list for negotiators.
            colormap: Matplotlib colormap name for coloring offers.
            ylimits: Y-axis limits as (min, max) tuple.
            common_legend: Whether to use a shared legend for all subplots.
            extra_annotation: Additional text to annotate on the plot.
            xdim: Dimension for x-axis ("relative_time", "step", "time").
            colorizer: Optional colorizer function for offers.
            only2d: Whether to show only the 2D utility space plot.
            no2d: Whether to skip the 2D utility space plot.
            fast: Whether to use faster but less detailed rendering.
            simple_offers_view: Whether to use simplified offer visualization.
            mark_offers_view: Whether to mark offers in the utility space view.
            mark_pareto_points: Whether to mark Pareto optimal points.
            mark_all_outcomes: Whether to mark all possible outcomes.
            mark_nash_points: Whether to mark Nash bargaining solution.
            mark_kalai_points: Whether to mark Kalai-Smorodinsky solution.
            mark_ks_points: Whether to mark KS points.
            mark_max_welfare_points: Whether to mark maximum welfare points.
            show: Whether to display the figure immediately.

        Returns:
            Plotly figure object if show=False, None otherwise.

        Raises:
            ValueError: If scenario is None, history_type is not 'full_trace', or utility functions are missing.
        """
        from negmas.plots.util import plot_offline_run

        # Validate that we have the necessary data
        if self.scenario is None:
            raise ValueError(
                "Cannot plot CompletedRun without a scenario. "
                "Load the CompletedRun with load_scenario=True."
            )

        if self.history_type != "full_trace":
            raise ValueError(
                f"Can only plot CompletedRun with history_type='full_trace', "
                f"got '{self.history_type}'. When saving the run, use "
                f"to_completed_run(trace='full_trace')."
            )

        if not self.scenario.ufuns:
            raise ValueError(
                "Cannot plot CompletedRun without utility functions in the scenario."
            )

        # Extract negotiator info from config
        negotiator_ids = self.config.get("negotiator_ids", [])
        negotiator_names = self.config.get("negotiator_names", [])

        # If we don't have IDs/names in config, try to extract from history
        if not negotiator_ids and self.history:
            # Extract unique negotiator IDs from trace
            seen_ids = []
            for entry in self.history:
                if hasattr(entry, "negotiator"):
                    nid = entry.negotiator  # type: ignore[union-attr]
                    if nid is not None and nid not in seen_ids:
                        seen_ids.append(nid)
            negotiator_ids = seen_ids

        if not negotiator_names:
            negotiator_names = negotiator_ids

        # Get outcome stats to determine end state
        timedout = self.outcome_stats.get("timedout", False)
        broken = self.outcome_stats.get("broken", False)
        has_error = self.outcome_stats.get("has_error", False)
        errstr = self.outcome_stats.get("error_details", "")

        # Convert history to list of TraceElement if needed
        trace = cast(list[TraceElement], self.history)

        return plot_offline_run(
            trace=trace,
            ids=negotiator_ids,
            ufuns=self.scenario.ufuns,
            agreement=self.agreement,
            timedout=timedout,
            broken=broken,
            has_error=has_error,
            errstr=errstr,
            names=negotiator_names,
            negotiators=plotting_negotiators,
            save_fig=save_fig,
            path=path,
            fig_name=fig_name,
            image_format=image_format,
            ignore_none_offers=ignore_none_offers,
            with_lines=with_lines,
            show_agreement=show_agreement,
            show_pareto_distance=show_pareto_distance,
            show_nash_distance=show_nash_distance,
            show_kalai_distance=show_kalai_distance,
            show_ks_distance=show_ks_distance,
            show_max_welfare_distance=show_max_welfare_distance,
            show_max_relative_welfare_distance=show_max_relative_welfare_distance,
            show_end_reason=show_end_reason,
            show_annotations=show_annotations,
            show_reserved=show_reserved,
            show_total_time=show_total_time,
            show_relative_time=show_relative_time,
            show_n_steps=show_n_steps,
            colors=colors,
            markers=markers,
            colormap=colormap,
            ylimits=ylimits,
            common_legend=common_legend,
            extra_annotation=extra_annotation,
            xdim=xdim,
            colorizer=colorizer,
            only2d=only2d,
            no2d=no2d,
            fast=fast,
            simple_offers_view=simple_offers_view,
            mark_offers_view=mark_offers_view,
            mark_pareto_points=mark_pareto_points,
            mark_all_outcomes=mark_all_outcomes,
            mark_nash_points=mark_nash_points,
            mark_kalai_points=mark_kalai_points,
            mark_ks_points=mark_ks_points,
            mark_max_welfare_points=mark_max_welfare_points,
            show=show,
        )


class Mechanism(
    NamedObject,
    EventSource,
    CheckpointMixin,
    Generic[TNMI, TState, TAction, TNegotiator],
    ABC,
):
    """Base class for all negotiation Mechanisms.

    Override the `round` function of this class to implement a round of your mechanism.

    Args:
        initial_state: Initial mechanism state. If None, a default MechanismState will be created.
        outcome_space: The negotiation agenda as an OutcomeSpace object. Use this for complex outcome spaces.
        issues: List of Issue objects defining the negotiation dimensions. Alternative to outcome_space.
        outcomes: List of valid outcomes or an integer specifying the number of outcomes. Alternative to outcome_space.
        n_steps: Maximum number of negotiation rounds/steps. None means unlimited. This is a shared limit visible to all negotiators.
        time_limit: Maximum negotiation duration in seconds. None means unlimited. This is a shared limit visible to all negotiators.
        pend: Probability (0-1) of ending negotiation at each step. 0 means disabled.
        pend_per_second: Probability (0-1) of ending negotiation each second. 0 means disabled.
        step_time_limit: Maximum time in seconds allowed for each negotiation step/round. None means unlimited.
        negotiator_time_limit: Maximum cumulative time in seconds for each negotiator's responses. None means unlimited.
        hidden_time_limit: Secret time limit not visible to negotiators. Used for testing or forcing timeouts.
        max_n_negotiators: Maximum number of negotiators allowed to join. None means unlimited.
        dynamic_entry: If True, negotiators can join/leave after negotiation starts. If False, all must join before start.
        annotation: Dictionary of arbitrary metadata attached to this mechanism session.
        nmi_factory: Class to use for creating NegotiatorMechanismInterface instances. Defaults to NegotiatorMechanismInterface.
        extra_callbacks: If True, call additional negotiator callbacks like on_round_start/end, on_leave, etc.
        checkpoint_every: Save checkpoint every N steps. Set to <=0 to disable checkpointing.
        checkpoint_folder: Directory path for saving checkpoints. None disables checkpointing.
        checkpoint_filename: Base name for checkpoint files. Step numbers will be prefixed.
        extra_checkpoint_info: Additional data to save in checkpoint metadata as a dictionary.
        single_checkpoint: If True, only keep the most recent checkpoint (overwrite previous ones).
        exist_ok: If True, allow overwriting existing checkpoint files.
        name: Human-readable name for this mechanism session. Auto-generated if not provided.
        genius_port: Port number for Genius bridge connection. Use 0 for automatic port selection.
        id: Unique system-wide identifier. Usually auto-generated; only set explicitly during deserialization.
        type_name: Custom type name for this mechanism. Auto-generated from class name if not provided.
        verbosity: Logging verbosity level (0=quiet, higher=more verbose).
        ignore_negotiator_exceptions: If True, mechanism continues even if negotiators raise exceptions.

    Remarks:
        - You can specify per-negotiator time limits and step limits when calling add() to add negotiators.
        - The mechanism tracks three types of limits:
          * shared_*: Limits visible to all negotiators (passed here)
          * private_*: Per-negotiator limits (passed to add())
          * internal_*: Effective limits (minimum of shared and all private limits)
        - Negotiators see their own relative_time based on their private limits
        - History and internal tracking use internal limits (strictest across all negotiators)
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
        """Initialize a negotiation mechanism.

        Args:
            initial_state: Starting state for the mechanism. Auto-created if None.
            outcome_space: OutcomeSpace defining valid outcomes. Alternative to issues/outcomes.
            issues: List of Issue objects. Alternative to outcome_space.
            outcomes: Explicit outcome list or count. Alternative to outcome_space.
            n_steps: Shared maximum rounds. Visible to all negotiators.
            time_limit: Shared maximum seconds. Visible to all negotiators.
            pend: Step-wise termination probability [0-1].
            pend_per_second: Time-wise termination probability [0-1].
            step_time_limit: Maximum seconds per step.
            negotiator_time_limit: Maximum cumulative seconds per negotiator.
            hidden_time_limit: Secret timeout not shown to negotiators.
            max_n_negotiators: Maximum participant count.
            dynamic_entry: Allow join/leave after start.
            annotation: Mechanism metadata dictionary.
            nmi_factory: Class for creating NMI instances.
            extra_callbacks: Enable extended negotiator callbacks.
            checkpoint_every: Checkpoint frequency in steps (<=0 disables).
            checkpoint_folder: Checkpoint save directory.
            checkpoint_filename: Base checkpoint filename.
            extra_checkpoint_info: Additional checkpoint metadata.
            single_checkpoint: Keep only latest checkpoint.
            exist_ok: Allow checkpoint file overwriting.
            name: Mechanism session identifier.
            genius_port: Genius bridge port (0=auto).
            id: System-wide unique ID (auto-generated).
            type_name: Custom type identifier (auto-generated).
            verbosity: Logging level (0=quiet).
            ignore_negotiator_exceptions: Continue on negotiator errors.
        """
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        self.__verbosity = verbosity
        self._negotiator_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._negotiator_time_limits = defaultdict(lambda: float("inf"))
        self._negotiator_n_steps = defaultdict(lambda: float("inf"))
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
        self._nmi_params = dict(
            id=self.id,
            n_outcomes=outcome_space.cardinality,
            outcome_space=outcome_space,
            shared_time_limit=time_limit,
            pend=pend,
            pend_per_second=pend_per_second,
            shared_n_steps=n_steps,
            step_time_limit=step_time_limit,
            negotiator_time_limit=negotiator_time_limit,
            dynamic_entry=dynamic_entry,
            max_n_negotiators=max_n_negotiators,
            annotation=annotation if annotation is not None else dict(),
            _mechanism=self,
            private_time_limit=float("inf"),
            private_n_steps=None,
        )
        self._nmi_factory = nmi_factory
        self._internal_nmi = nmi_factory(**self._nmi_params)  # type: ignore
        self._shared_nmi = nmi_factory(**self._nmi_params)  # type: ignore
        self._nmis = defaultdict(lambda: self._shared_nmi)

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

    @property
    def nmi(self) -> TNMI:
        """The Negotiation Mechanism Interface (NMI) with shared information available to all negotiators.

        Returns:
            TNMI: The NMI instance
        """

        warnings.deprecated(
            "Mechanism.nmi is depricated. Use `internal_nmi` or `shared_nmi` instead of it. "
            "The former takes into account private time limits and n steps limits of negotiators "
            "while the latter does not."
        )
        return self._internal_nmi

    @property
    def internal_nmi(self) -> TNMI:
        """The Negotiation Mechanism Interface (NMI) with combined deadline information from all negotiators.

        Returns:
            TNMI: The NMI instance
        """
        return self._internal_nmi

    @property
    def shared_nmi(self) -> TNMI:
        """The Negotiation Mechanism Interface (NMI) with shared information available to all negotiators.

        Returns:
            TNMI: The NMI instance
        """
        return self._shared_nmi

    def get_nmi_for(self, negotiator: TNegotiator) -> TNMI:
        """Returns the NMI instance for the given negotiator.

        By default, all negotiators share the same NMI instance, but
        mechanisms can override this to provide different NMIs to different
        negotiators.

        Args:
            negotiator: The negotiator for whom to get the NMI.
        """
        return self._nmis[negotiator.id]

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
        return outcome in self._internal_nmi.outcome_space

    @property
    def outcome_space(self) -> OutcomeSpace:
        """The space of all possible negotiation outcomes.

        Returns:
            OutcomeSpace: Defines valid outcomes including issues, values, and constraints
        """
        return self._internal_nmi.outcome_space

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
        return self._internal_nmi.outcomes

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
        """Elapsed time since mechanism started in seconds (nanosecond precision).

        0.0 if the mechanism did not start running
        """
        if self._start_time is None:
            return 0.0

        return (time.perf_counter_ns() - self._start_time) / 1_000_000_000

    @property
    def remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds.

        None if no time limit is given.
        """
        if self._internal_nmi.time_limit == float("+inf"):
            return None
        if not self._start_time:
            return self._internal_nmi.time_limit

        limit = (
            self._internal_nmi.time_limit
            - (time.perf_counter_ns() - self._start_time) / 1_000_000_000
        )
        if limit < 0.0:
            return 0.0

        return limit

    @property
    def state(self) -> TState:
        """
        The current state.

        Override `extra_state` if you want to keep extra state
        """
        return self._current_state

    def _relative_time_for(self, nmi: TNMI) -> float:
        """Calculate relative_time for a specific NMI (shared or per-negotiator)."""
        n_steps = nmi.n_steps
        time_limit = nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if nmi.pend <= 0 and nmi.pend_per_second <= 0:
                return 0.0
            if nmi.pend > 0:
                n_steps = int(math.ceil(1 / nmi.pend))
            if nmi.pend_per_second > 0:
                time_limit = 1 / nmi.pend_per_second

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit != float("+inf") else -1.0
        return min(1.0, max([relative_step, relative_time]))

    def _expected_relative_time_for(self, nmi: TNMI) -> float:
        """Calculate expected_relative_time for a specific NMI (shared or per-negotiator)."""
        n_steps = nmi.n_steps
        time_limit = nmi.time_limit
        if time_limit == float("+inf") and n_steps is None:
            if nmi.pend <= 0 and nmi.pend_per_second <= 0:
                return 0.0
        if n_steps is None:
            # set the expected number of steps to the reciprocal of the probability of ending at every step
            n_steps = int(math.ceil(1 / nmi.pend)) if nmi.pend > 0 else n_steps
        else:
            n_steps = (
                min(int(math.ceil(1 / nmi.pend)), n_steps) if nmi.pend > 0 else n_steps
            )
        if time_limit == float("inf"):
            # set the expected number of seconds to the reciprocal of the probability of ending every second
            time_limit = (
                int(math.ceil(1 / nmi.pend_per_second))
                if nmi.pend_per_second > 0
                else time_limit
            )
        else:
            time_limit = (
                min(time_limit, int(math.ceil(1 / nmi.pend_per_second)))
                if nmi.pend_per_second > 0
                else time_limit
            )

        relative_step = (
            (self._current_state.step + 1) / (n_steps + 1)
            if n_steps is not None
            else -1.0
        )
        relative_time = self.time / time_limit if time_limit != float("+inf") else -1.0
        return max([relative_step, relative_time])

    @property
    def relative_time(self) -> float:
        """
        Returns a number between ``0`` and ``1`` indicating elapsed relative time or steps.

        This uses shared time limits (visible to all negotiators).

        Remarks:
            - If pend or pend_per_second are defined in the `NegotiatorMechanismInterface`,
              and time_limit/n_steps are not given, this becomes an expectation that is limited above by one.
        """
        return self._relative_time_for(self._shared_nmi)

    @property
    def expected_relative_time(self) -> float:
        """
        Returns a positive number indicating elapsed relative time or steps.

        This uses internal time limits (the strictest across all negotiators).

        Remarks:
            - This is relative to the expected time/step at which the negotiation ends given all timing
              conditions (time_limit, n_step, pend, pend_per_second).
        """
        return self._expected_relative_time_for(self._internal_nmi)

    @property
    def expected_remaining_time(self) -> float | None:
        """
        Returns remaining time in seconds (expectation).

        None if no time limit or pend_per_second is given.
        """
        rem = self.remaining_time
        pend = self._internal_nmi.pend_per_second
        if pend <= 0:
            return rem
        return min(rem, (1 / pend)) if rem is not None else (1 / pend)

    @property
    def remaining_steps(self) -> int | None:
        """
        Returns the remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        if self._internal_nmi.n_steps is None:
            return None

        return self._internal_nmi.n_steps - self._current_state.step

    @property
    def expected_remaining_steps(self) -> int | None:
        """
        Returns the expected remaining number of steps until the end of the mechanism run.

        None if unlimited
        """
        rem = self.remaining_steps
        pend = self._internal_nmi.pend
        if pend <= 0:
            return rem

        return (
            min(rem, int(math.ceil(1 / pend)))
            if rem is not None
            else int(math.ceil(1 / pend))
        )

    def _state_for_nmi(self, nmi: TNMI) -> TState:
        """
        Returns a state object with relative_time calculated for the given NMI.

        The state is a copy of the current state with only relative_time
        modified based on the NMI's time limits.
        """
        # Always create a copy with modified relative_time field based on the NMI
        # We cannot shortcut even when nmi == shared_nmi because _current_state
        # uses _internal_nmi (strictest limits), not shared_nmi
        state = copy.copy(self._current_state)
        object.__setattr__(state, "relative_time", self._relative_time_for(nmi))
        return state

    def state_for(self, negotiator: TNegotiator) -> TState:
        """
        Returns a state object for the given negotiator with per-negotiator relative_time.

        The state is a copy of the current state with only relative_time
        modified based on the negotiator's private time limits.
        """
        nmi = self._nmis[negotiator.id]
        return self._state_for_nmi(nmi)

    def _get_nmi(self, negotiator: TNegotiator) -> TNMI:
        return self.get_nmi_for(negotiator)

    def _get_ami(self, negotiator: TNegotiator) -> TNMI:
        warnings.deprecated("_get_ami is depricated. Use `get_nmi` instead of it")
        return self._nmis[negotiator.id]

    def add(
        self,
        negotiator: TNegotiator,
        *,
        preferences: Preferences | None = None,
        role: str | None = None,
        ufun: BaseUtilityFunction | None = None,
        time_limit: float | None = float("inf"),
        n_steps: int | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> bool | None:
        """Add a negotiator to the negotiation.

        Args:
            negotiator: The negotiator to be added.
            preferences: The utility function to use. If None, then the negotiator must already have a stored
                  utility function otherwise it will fail to enter the negotiation.
            ufun: [deprecated] same as preferences but must be a `UFun` object.
            role: The role the negotiator plays in the negotiation mechanism. It is expected that mechanisms inheriting from
                  this class will check this parameter to ensure that the role is a valid role and is still possible for
                  negotiators to join on that role. Roles may include things like moderator, representative etc based
                  on the mechanism.
            time_limit: Per-negotiator time limit in seconds. This creates a private time limit for this negotiator.
                  The negotiator will see `nmi.time_limit = min(shared_time_limit, time_limit)` and their `relative_time`
                  will be calculated based on this effective limit. If None or inf, only the shared time limit applies.
                  This allows different negotiators to have different time constraints in the same negotiation.
            n_steps: Per-negotiator step limit. This creates a private step limit for this negotiator.
                  The negotiator will see `nmi.n_steps = min(shared_n_steps, n_steps)` and their `relative_time`
                  will be calculated based on this effective limit. If None, only the shared step limit applies.
                  This allows different negotiators to have different step constraints in the same negotiation.
            annotation: Additional metadata to attach to this negotiator's NMI.

        Returns:
            * True if the negotiator was added.
            * False if the negotiator was already in the negotiation.
            * None if the negotiator cannot be added. This can happen in the following cases:

              1. The capabilities of the negotiator do not match the requirements of the negotiation
              2. The outcome-space of the negotiator's preferences do not contain the outcome-space of the negotiation
              3. The negotiator refuses to join (by returning False from its `join` method) see `Negotiator.join` for possible reasons of that

        Notes:
            Per-negotiator limits enable scenarios where different negotiators have different time/step constraints:

            - A negotiator with `time_limit=30` in a mechanism with `time_limit=60` will see `nmi.time_limit=30`
            - A negotiator with `time_limit=90` in a mechanism with `time_limit=60` will see `nmi.time_limit=60`
            - A negotiator with `time_limit=None` or `time_limit=inf` will see the shared limit
            - The mechanism's internal tracking uses the strictest limit across all negotiators
            - Each negotiator's `relative_time` is calculated based on their individual effective limit

            This maintains backward compatibility: negotiators added without private limits behave exactly as before.
        """
        if annotation is None:
            annotation = dict()

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

        nmi = self._nmi_factory(
            **self._nmi_params  # type: ignore
            | dict(
                private_time_limit=time_limit
                if time_limit is not None
                else float("inf"),
                private_n_steps=n_steps,
                annotation=self._nmi_params.get("annotation", dict()) | annotation,  # type: ignore
            )
        )

        # Add NMI to _nmis before calling join so state_for() works
        self._nmis[negotiator.id] = nmi

        if negotiator.join(
            nmi=nmi,
            state=self.state_for(negotiator),
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
            object.__setattr__(
                self._internal_nmi,
                "time_limit",
                min(self._internal_nmi.time_limit, nmi.time_limit),
            )

            def min_steps(a: int | None, b: int | None) -> int | None:
                if a is None:
                    return b
                if b is None:
                    return a
                return min(a, b)

            object.__setattr__(
                self._internal_nmi,
                "n_steps",
                min_steps(self._internal_nmi.n_steps, nmi.n_steps),
            )
            return True

        # Remove NMI if join failed
        del self._nmis[negotiator.id]
        return None

    def can_participate(self, negotiator: TNegotiator) -> bool:
        """Checks if the negotiator can participate in this type of negotiation in
        general.

        Returns:
            bool: True if the negotiator  can participate

        Remarks:
            The only reason this may return `False` is if the mechanism requires some requirements
            that are not within the capabilities of the negotiator.

        """
        return self.is_satisfying(negotiator.capabilities)

    def is_satisfying(self, capabilities: dict) -> bool:
        """Check if given capabilities satisfy the mechanism's requirements.

        Args:
            capabilities: Dictionary of capabilities to check

        Returns:
            True if capabilities satisfy all requirements, False otherwise

        Remarks:
            - None requirement value means any capability value is acceptable
            - Set/list requirement means capability must match one of the values
            - Tuple of 2 numbers is treated as a range [min, max] - capability can be value in range or overlapping range
            - For exact equality check after set conversion if both are collections
        """

        def is_range(val):
            """Check if value is a numeric range (tuple of exactly 2 numbers)"""
            return (
                isinstance(val, tuple)
                and len(val) == 2
                and isinstance(val[0], (int, float))
                and isinstance(val[1], (int, float))
            )

        def ranges_overlap(r1, r2):
            """Check if two ranges overlap"""
            return max(r1[0], r2[0]) <= min(r1[1], r2[1])

        def value_in_range(val, rng):
            """Check if a value is within a range"""
            return rng[0] <= val <= rng[1]

        for req, req_val in self._requirements.items():
            cap_val = capabilities.get(req)

            # No capability for this requirement
            if cap_val is None and req_val is not None:
                return False

            # None requirement means any value is acceptable
            if req_val is None:
                continue

            # Handle range requirements
            if is_range(req_val):
                if is_range(cap_val):
                    # Both are ranges - check for overlap
                    if not ranges_overlap(req_val, cap_val):
                        return False
                elif isinstance(cap_val, (int, float)):
                    # Capability is a single value - check if in range
                    if not value_in_range(cap_val, req_val):
                        return False
                elif isinstance(cap_val, (list, set)):
                    # Capability is a list/set - check if any value is in range
                    if not any(
                        value_in_range(v, req_val)
                        for v in cap_val
                        if isinstance(v, (int, float))
                    ):
                        return False
                else:
                    return False
                continue

            # Handle range capabilities (requirement is not a range)
            if is_range(cap_val):
                if isinstance(req_val, (int, float)):
                    # Requirement is a single value - check if in capability range
                    if not value_in_range(req_val, cap_val):
                        return False
                elif isinstance(req_val, (list, set)):
                    # Requirement is a list/set - check if any value is in range
                    if not any(
                        value_in_range(v, cap_val)
                        for v in req_val
                        if isinstance(v, (int, float))
                    ):
                        return False
                else:
                    return False
                continue

            # Convert to sets for comparison if they're collections (but not ranges)
            req_set = set(req_val) if isinstance(req_val, (list, set)) else None
            cap_set = set(cap_val) if isinstance(cap_val, (list, set)) else None

            # Both are collections - check for overlap
            if req_set is not None and cap_set is not None:
                if not req_set.intersection(cap_set):
                    return False
                continue

            # Requirement is a collection, capability is a single value
            if req_set is not None:
                if cap_val not in req_set:
                    return False
                continue

            # Capability is a collection, requirement is a single value
            if cap_set is not None:
                if req_val not in cap_set:
                    return False
                continue

            # Both are single values - must match exactly
            if req_val != cap_val:
                return False

        return True

    def can_accept_more_negotiators(self) -> bool:
        """Whether the mechanism can **currently** accept more negotiators."""
        return (
            True
            if self._internal_nmi.max_n_negotiators is None or self._negotiators is None
            else len(self._negotiators) < self._internal_nmi.max_n_negotiators
        )

    def can_enter(self, negotiator: TNegotiator) -> bool:
        """Whether the negotiator can enter the negotiation now."""
        return self.can_accept_more_negotiators() and self.can_participate(negotiator)

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
            if self._internal_nmi.dynamic_entry
            else not self._internal_nmi.state.running
            and negotiator in self._negotiators
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
            self._call(negotiator, negotiator.on_leave, self._shared_nmi.state)
            self._negotiator_times[negotiator.id] += time.perf_counter() - strt
        return True

    @property
    def requirements(self) -> dict:
        """Protocol requirements dictionary.

        Returns:
            Dictionary mapping requirement names to acceptable values
        """
        return self._requirements

    @requirements.setter
    def requirements(self, value: dict) -> None:
        """Set protocol requirements, converting lists to sets.

        Args:
            value: Dictionary mapping requirement names to acceptable values
        """
        self._requirements = {
            k: set(v) if isinstance(v, list) else v for k, v in value.items()
        }

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
        return self._internal_nmi.n_outcomes

    @property
    def issues(self) -> list[Issue]:
        """Returns the issues of the outcome space (if defined).

        Will raise an exception if the outcome space has no defined
        issues
        """
        return getattr(self._internal_nmi.outcome_space, "issues")

    @property
    def completed(self):
        """Ended without timing out (either with agreement or broken by a negotiator)"""
        return self.agreement is not None or self._current_state.broken

    @property
    def ended(self):
        """Ended in any way"""
        return self._current_state.ended

    def n_steps_for(self, nid: str):
        """Returns the maximum number of negotiation steps allowed as seen by a given negotiator, or None if unlimited."""
        return self._nmis[nid].n_steps

    def time_limit_for(self, nid: str):
        """Returns the maximum negotiation time in seconds as seen by a given negotiator, or infinity if unlimited."""
        return self._nmis[nid].time_limit

    @property
    def n_steps(self):
        """Returns the maximum number of negotiation steps allowed taking into account individual negotiator limits, or None if unlimited."""
        return self._internal_nmi.n_steps

    @property
    def time_limit(self):
        """Returns the maximum negotiation time in seconds taking into account individual negotiator limits, or infinity if unlimited."""
        return self._internal_nmi.time_limit

    @property
    def shared_n_steps(self):
        """Returns the maximum number of negotiation steps allowed according to shared information between negotiators, or None if unlimited."""
        return self._shared_nmi.n_steps

    @property
    def shared_time_limit(self):
        """Returns the maximum negotiation time in seconds according to shared information between negotiators, or infinity if unlimited."""
        return self._shared_nmi.time_limit

    @property
    def running(self):
        """Running."""
        return self._current_state.running

    @property
    def dynamic_entry(self):
        """Returns whether negotiators can join/leave during negotiation."""
        return self._internal_nmi.dynamic_entry

    @property
    def max_n_negotiators(self):
        """Max n negotiators."""
        return self._internal_nmi.max_n_negotiators

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
        if self._extra_callbacks:
            for a in self.negotiators:
                strt = time.perf_counter()
                self._call(a, a.on_mechanism_error, state=self.state_for(a))
                self._negotiator_times[a.id] += time.perf_counter() - strt

    def on_negotiation_end(self) -> None:
        """Called at the end of each negotiation.

        Remarks:
            - When overriding this function you **MUST** call the base class version
        """
        state = self.state
        for a in self.negotiators:
            strt = time.perf_counter()
            self._call(a, a._on_negotiation_end, state=self.state_for(a))
            self._negotiator_times[a.id] += time.perf_counter() - strt
        self.announce(
            Event(
                type="negotiation_end",
                data={
                    "agreement": self.agreement,
                    "state": state,
                    "annotation": self._internal_nmi.annotation,
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
            self._start_time = time.perf_counter_ns()
        if self.__verbosity >= 1:
            if self.current_step == 0:
                print(
                    f"{self.name}: Step {self.current_step} starting after {datetime.now()}",
                    flush=True,
                )
            else:
                _elapsed = (time.perf_counter_ns() - self._start_time) / 1_000_000_000
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
        state4history = self.state4history
        rs, rt = random.random(), 2

        # end with a timeout if condition is met
        current_time = self.time
        if self.__last_second_tried < int(current_time):
            rt, self.__last_second_tried = random.random(), int(current_time)

        if (
            (current_time > self.time_limit)
            or (
                self._internal_nmi.n_steps
                and self._current_state.step >= self._internal_nmi.n_steps
            )
            or current_time > self._hidden_time_limit
            or rs < self._internal_nmi.pend - 1e-8
            or rt < self._internal_nmi.pend_per_second - 1e-8
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
            if self._internal_nmi.dynamic_entry:
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
            # Initialize with internal_nmi for history
            self._current_state.relative_time = self._relative_time_for(
                self._internal_nmi
            )
            self._start_time = time.perf_counter_ns()
            self._current_state.started = True
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
                self._call(a, a._on_negotiation_start, state=self.state_for(a))
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
                self._call(
                    negotiator,
                    negotiator.on_round_start,
                    state=self.state_for(negotiator),
                )
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
            self._internal_nmi.step_time_limit is not None
            and step_time > self._internal_nmi.step_time_limit
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
        if not self._current_state.waiting and result.completed:
            state4history = self.state4history
            if self._extra_callbacks:
                for negotiator in self._negotiators:
                    strt = time.perf_counter()
                    self._call(
                        negotiator,
                        negotiator.on_round_end,
                        state=self.state_for(negotiator),
                    )
                    self._negotiator_times[negotiator.id] += time.perf_counter() - strt
            self._add_to_history(state4history)
            # we only indicate a new step if no one is waiting
            self._current_state.step += 1
            self._current_state.time = self.time
            # History uses internal_nmi (strictest limits) for relative_time
            self._current_state.relative_time = self._relative_time_for(
                self._internal_nmi
            )

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
                self._call(
                    negotiator,
                    negotiator.on_round_end,
                    state=self.state_for(negotiator),
                )
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
        start_callback: Callable[[int, Mechanism], None] | None = None,
        progress_callback: Callable[[int, Mechanism], None] | None = None,
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
            start_callback: Optional callback called once at the start of each mechanism with (negotiation_id, mechanism)
            progress_callback: Optional callback called after each step of each mechanism with (negotiation_id, mechanism)
            completion_callback: Optional callback called once at the end of each mechanism with (negotiation_id, mechanism)
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
                if start_callback:
                    start_callback(i, mechanism)
                try:
                    mechanism.run(
                        start_callback=None,  # Already called above
                        progress_callback=(
                            lambda state: progress_callback(i, mechanism)
                            if progress_callback
                            else None
                        ),
                        completion_callback=None,  # Will call completion_callback below
                    )
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
            started = [False] * len(mechanisms)  # Track which mechanisms have started
            states: list[TState | None] = [None] * len(mechanisms)
            allindices = list(range(len(list(mechanisms))))
            indices = allindices if not ordering else list(ordering)
            notmentioned = set(allindices).difference(indices)
            assert len(notmentioned) == 0, (
                f"Mechanisms {notmentioned} are never mentioned in the ordering."
            )
            if ordering_fun:
                j = 0
                while not all(completed):
                    states = [_.state for _ in mechanisms]
                    i = ordering_fun(j, states)
                    j += 1
                    mechanism = mechanisms[i]
                    if completed[i]:
                        continue
                    if not started[i]:
                        if start_callback:
                            start_callback(i, mechanism)
                        started[i] = True
                    try:
                        result = mechanism.step()
                        if progress_callback:
                            progress_callback(i, mechanism)
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
                        if not started[i]:
                            if start_callback:
                                start_callback(i, mechanism)
                            started[i] = True
                        try:
                            result = mechanism.step()
                            if progress_callback:
                                progress_callback(i, mechanism)
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

    def run_with_progress(
        self,
        timeout=None,
        start_callback: Callable[[TState], None] | None = None,
        progress_callback: Callable[[TState], None] | None = None,
        completion_callback: Callable[[TState], None] | None = None,
    ) -> TState:
        """Run with a progress bar (using rich).

        Args:
            timeout: Maximum time in seconds to run, or None for no limit
            start_callback: Optional callback called once at the start of negotiation with the initial state
            progress_callback: Optional callback called after each negotiation step with the current state
            completion_callback: Optional callback called once at the end of negotiation with the final state

        Returns:
            TState: The final negotiation state after completion
        """
        from rich.progress import Progress

        if start_callback:
            start_callback(self._current_state)
        if timeout is None:
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    if progress_callback:
                        progress_callback(self._current_state)
                    progress.update(task, completed=int(self.relative_time * 100))
        else:
            start_time = time.perf_counter()
            with Progress() as progress:
                task = progress.add_task("Negotiating ...", total=100)
                for _ in self:
                    if progress_callback:
                        progress_callback(self._current_state)
                    progress.update(task, completed=int(self.relative_time * 100))
                    if time.perf_counter() - start_time > timeout:
                        (
                            self._current_state.running,
                            self._current_state.timedout,
                            self._current_state.broken,
                        ) = (False, True, False)
                        self.on_negotiation_end()
                        break

        if completion_callback:
            completion_callback(self._current_state)
        return self.state

    def run(
        self,
        timeout=None,
        start_callback: Callable[[TState], None] | None = None,
        progress_callback: Callable[[TState], None] | None = None,
        completion_callback: Callable[[TState], None] | None = None,
    ) -> TState:
        """Execute the negotiation mechanism until completion or timeout.

        Args:
            timeout: Maximum time in seconds to run, or None for no limit
            start_callback: Optional callback called once at the start of negotiation with the initial state
            progress_callback: Optional callback called after each negotiation step with the current state
            completion_callback: Optional callback called once at the end of negotiation with the final state

        Returns:
            TState: Final negotiation state after completion
        """
        if start_callback:
            start_callback(self._current_state)
        if timeout is None:
            for _ in self:
                if progress_callback:
                    progress_callback(self._current_state)
        else:
            start_time = time.perf_counter()
            for _ in self:
                if progress_callback:
                    progress_callback(self.state)
                if time.perf_counter() - start_time > timeout:
                    (
                        self._current_state.running,
                        self._current_state.timedout,
                        self._current_state.broken,
                    ) = (False, True, False)
                    self.on_negotiation_end()
                    break
        if completion_callback:
            completion_callback(self.state)
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
        self,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        agreement_stats: OutcomeOptimality | None = None,
        calc_agreement_stats: bool = False,
    ) -> CompletedRun:
        """
        Creates a CompletedRun object from the current mechanism state.

        Args:
            source: The source of the history information. Options:
                - None: Auto-detect the best available source in priority order:
                  full_trace_with_utils > full_trace > extended_trace > trace > history
                - "history": Use the history attribute (list of states)
                - "full_trace": Use the full_trace attribute (if available)
                - "full_trace_with_utils": Use full_trace_with_utils (if available)
                - "trace": Use the trace attribute (if available)
                - "extended_trace": Use the extended_trace attribute (if available)
            metadata: Arbitrary metadata to include in the CompletedRun.
            agreement_stats: Pre-calculated agreement optimality statistics. If provided,
                these will be used directly.
            calc_agreement_stats: If True and agreement_stats is None, calculate
                agreement_stats from the scenario (requires ufuns to be available).
                This involves calculating scenario stats which can be expensive.

        Returns:
            CompletedRun: A CompletedRun object containing the negotiation data.
        """
        from negmas.inout import Scenario

        # Auto-detect source if None - use best available in priority order
        if source is None:
            if hasattr(self, "full_trace_with_utils"):
                source = "full_trace_with_utils"
            elif hasattr(self, "full_trace"):
                source = "full_trace"
            elif hasattr(self, "extended_trace"):
                source = "extended_trace"
            elif hasattr(self, "trace"):
                source = "trace"
            else:
                source = "history"

        # Get the trace data based on source
        if source == "full_trace_with_utils" and hasattr(self, "full_trace_with_utils"):
            trace_data = self.full_trace_with_utils  # type: ignore
            history_type = "full_trace_with_utils"
        elif source == "full_trace" and hasattr(self, "full_trace"):
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
            "has_error": self.state.has_error,
            "error_details": str(self.state.error_details)
            if self.state.error_details
            else "",
            "utilities": utilities,
        }

        # Calculate agreement_stats if requested and not provided
        if (
            agreement_stats is None
            and calc_agreement_stats
            and self.agreement is not None
        ):
            try:
                from negmas.preferences.ops import (
                    calc_outcome_distances,
                    calc_outcome_optimality,
                    calc_scenario_stats,
                    estimate_max_dist,
                )

                ufuns = [n.ufun for n in self.negotiators if n.ufun is not None]
                if ufuns and len(ufuns) == len(self.negotiators):
                    # Calculate scenario stats (expensive)
                    stats = calc_scenario_stats(ufuns)
                    # Calculate agreement utilities
                    agreement_utils = tuple(
                        float(u(self.agreement)) if u is not None else 0.0
                        for u in ufuns
                    )
                    # Calculate distances and optimality
                    dists = calc_outcome_distances(agreement_utils, stats)
                    agreement_stats = calc_outcome_optimality(
                        dists, stats, estimate_max_dist(ufuns)
                    )
            except Exception:
                agreement_stats = None

                ufuns = tuple(
                    n.preferences for n in self.negotiators if n.preferences is not None
                )
                if ufuns and len(ufuns) == len(self.negotiators):
                    # Calculate scenario stats (expensive)
                    stats = calc_scenario_stats(ufuns)
                    # Calculate agreement utilities
                    agreement_utils = tuple(
                        float(u(self.agreement)) if u is not None else 0.0
                        for u in ufuns
                    )
                    # Calculate distances and optimality
                    dists = calc_outcome_distances(agreement_utils, stats)
                    agreement_stats = calc_outcome_optimality(
                        dists, stats, estimate_max_dist(ufuns)
                    )
            except Exception:
                agreement_stats = None

        return CompletedRun(
            history=trace_data,
            history_type=history_type,
            scenario=scenario,
            agreement=self.agreement,
            agreement_stats=agreement_stats,
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
            n_steps=self._internal_nmi.n_steps,
            time_limit=self._internal_nmi.time_limit,
            shared_n_steps=self._shared_nmi.n_steps,
            shared_time_limit=self._shared_nmi.time_limit,
            pend=self._internal_nmi.pend,
            pend_per_second=self._internal_nmi.pend_per_second,
            step_time_limit=self._internal_nmi.step_time_limit,
            negotiator_time_limit=self._internal_nmi.negotiator_time_limit,
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

    def available_save_sources(self) -> list[str]:
        """Returns the available sources for saving history.

        Returns:
            list[str]: The available sources for save().
        """
        sources = ["history"]
        if hasattr(self, "full_trace_with_utils"):
            sources.append("full_trace_with_utils")  # type: ignore
        if hasattr(self, "full_trace"):
            sources.append("full_trace")  # type: ignore
        if hasattr(self, "extended_trace"):
            sources.append("extended_trace")  # type: ignore
        if hasattr(self, "trace"):
            sources.append("trace")  # type: ignore
        return sources

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
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        agreement_stats: OutcomeOptimality | None = None,
        calc_agreement_stats: bool = False,
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
            source: The source of the history information to save. If None (default), auto-detect the best
                available source in priority order: full_trace_with_utils > full_trace > extended_trace > trace > history.
            metadata: arbitrary data to save under `meta_data.yaml`. Only supported if single_file is False
            agreement_stats: Pre-calculated agreement optimality statistics. If provided, these will be saved
                            instead of calculating them (which requires scenario stats).
            calc_agreement_stats: If True and agreement_stats is None, calculate agreement_stats
                                 from the scenario. This is expensive as it requires calculating scenario stats.
            overwrite: Overwrite existing files/folders
            warn_if_existing: Warn if existing files/folders are found
            storage_format: The format for storing tables.

        Returns:
            Path: The path to the saved file or directory.
        """
        completed_run = self.to_completed_run(
            source=source,
            metadata=metadata,
            agreement_stats=agreement_stats,
            calc_agreement_stats=calc_agreement_stats,
        )
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
