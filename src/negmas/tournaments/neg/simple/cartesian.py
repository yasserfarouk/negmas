"""
Negotiation tournaments module.
"""

from __future__ import annotations

import ast
import base64
import copy
import datetime
import os
import shutil
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import product
from math import exp, isinf, isnan, log
from os import cpu_count
from pathlib import Path
from random import randint, random, shuffle
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal

if TYPE_CHECKING:
    from negmas.outcomes.protocols import OutcomeSpace
    from negmas.preferences.base_ufun import BaseUtilityFunction

import cloudpickle
import pandas as pd
from attr import asdict, define, field
from rich import print
from rich.progress import track

from negmas.common import MechanismState
from negmas.helpers import unique_name
from negmas.helpers.inout import (
    DEFAULT_TABLE_STORAGE_FORMAT,
    dump,
    has_needed_files,
    load,
)
from negmas.helpers.strings import encode_params, humanize_time, shortest_unique_names
from negmas.helpers.types import get_class, get_full_type_name
from negmas.inout import Scenario, scenario_size
from negmas.mechanisms import Mechanism
from negmas.negotiators import Negotiator
from negmas.plots.util import plot_offline_run
from negmas.preferences.discounted import DiscountedUtilityFunction
from negmas.preferences.ops import (
    COMPARE_UFUN_METHOD_TYPE,
    OutcomeOptimality,
    ScenarioStats,
    calc_outcome_distances,
    calc_outcome_optimality,
    estimate_max_dist,
    compare_ufuns,
)
from negmas.sao.common import SAOState
from negmas.sao.mechanism import SAOMechanism
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, serialize, to_flat_dict
from negmas.warnings import NegmasUnexpectedValueWarning, deprecated, warn

OptimizationLevel = Literal["speed", "time", "none", "balanced", "space", "max"]
StorageFormat = Literal["csv", "gzip", "parquet"]


def hash_to_base64(n: int) -> str:
    """
    Deterministically encodes a large integer into a URL-safe Base64 string.

    Handles negative integers by using signed=True for to_bytes().
    """
    if n == 0:
        return "AA"  # Minimal representation for zero

    # 1. Calculate how many bytes are needed to represent the integer
    # For signed integers, we need an extra bit for the sign
    byte_length = (n.bit_length() + 8) // 8  # +8 instead of +7 for sign bit

    # 2. Convert integer to bytes (Big Endian, signed to handle negative values)
    num_bytes = n.to_bytes(byte_length, byteorder="big", signed=True)

    # 3. Encode to Base64 and clean up padding characters (=)
    encoded = base64.urlsafe_b64encode(num_bytes).decode("utf-8")
    return encoded.rstrip("=")


class _PicklableCallback:
    """Wrapper to make callback with run_id picklable for parallel execution.

    Uses cloudpickle to serialize the callback function, allowing local closures
    and other complex callables to work in parallel mode with ProcessPoolExecutor.
    """

    def __init__(self, callback: Callable, run_id: int | str):
        # Serialize the callback with cloudpickle for cross-process compatibility
        self._callback_bytes = cloudpickle.dumps(callback)
        self.run_id = run_id
        self._callback = None  # Lazily deserialized

    def __call__(self, state):
        """Call the wrapped callback with run_id as first argument."""
        # Deserialize callback on first use (in the worker process)
        if self._callback is None:
            self._callback = cloudpickle.loads(self._callback_bytes)
        return self._callback(self.run_id, state)

    def __getstate__(self):
        """Custom pickle support - only serialize the bytes and run_id."""
        return {"_callback_bytes": self._callback_bytes, "run_id": self.run_id}

    def __setstate__(self, state):
        """Custom unpickle support."""
        self._callback_bytes = state["_callback_bytes"]
        self.run_id = state["run_id"]
        self._callback = None


def _normalize_optimization_level(level: OptimizationLevel) -> OptimizationLevel:
    """Normalize optimization level aliases to canonical values.

    Synonyms:
        - "time", "none" -> "speed" (no optimization, keep everything)
        - "max" -> "space" (maximum optimization)
    """
    if level in ("time", "none"):
        return "speed"
    if level == "max":
        return "space"
    return level


def _get_default_storage_format(
    storage_optimization: OptimizationLevel,
) -> StorageFormat:
    """Get the default storage format based on optimization level."""
    if storage_optimization == "speed":
        return "csv"
    elif storage_optimization == "balanced":
        return "gzip"
    else:  # "space"
        return "parquet"


def _get_file_extension(storage_format: StorageFormat) -> str:
    """Get file extension for a storage format."""
    if storage_format == "csv":
        return ".csv"
    elif storage_format == "gzip":
        return ".csv.gz"
    else:  # "parquet"
        return ".parquet"


def _check_and_correct_reserved_values(
    scenario: Scenario, scenario_name: str, eps: float = 0.0
) -> None:
    """Check for problematic reserved values in scenario ufuns, correct them, and warn.

    Corrects reserved values that are None, inf, -inf, or NaN to ufun.min() - eps.

    Args:
        scenario: The scenario to check and correct
        scenario_name: Name of the scenario for warning message
        eps: Epsilon value to subtract from ufun.min() when correcting reserved values
    """
    from negmas.preferences.ops import correct_reserved_value

    corrected_ufuns = []
    for i, ufun in enumerate(scenario.ufuns):
        if hasattr(ufun, "reserved_value"):
            rv = ufun.reserved_value
            try:
                corrected_rv, was_corrected = correct_reserved_value(
                    rv, ufun, eps=eps, warn=False
                )
                if was_corrected:
                    ufun.reserved_value = corrected_rv
                    corrected_ufuns.append((i, ufun.name, rv, corrected_rv))
            except Exception as e:
                # If we can't correct, just log the problematic ufun
                warn(
                    f"Scenario '{scenario_name}' ufun[{i}] '{ufun.name}' has problematic reserved value ({rv}) "
                    f"but could not correct it: {e}",
                    NegmasUnexpectedValueWarning,
                )

    if corrected_ufuns:
        ufun_details = ", ".join(
            f"ufun[{i}] '{name}' (was {old_rv}, now {new_rv})"
            for i, name, old_rv, new_rv in corrected_ufuns
        )
        warn(
            f"Scenario '{scenario_name}' had utility functions with problematic reserved values that were corrected: {ufun_details}. "
            f"Corrected values are set to ufun.min() - {eps}.",
            NegmasUnexpectedValueWarning,
        )


def _convert_complex_columns_to_json(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert complex object columns (lists, dicts) to JSON strings for parquet compatibility.

    Args:
        df: DataFrame to convert

    Returns:
        Tuple of (converted DataFrame, list of column names that were converted)
    """
    import json

    df = df.copy()
    converted_cols = []

    for col in df.columns:
        if df[col].dtype == object:
            # Check if any non-null value is a list or dict
            sample = df[col].dropna().head(10)
            if len(sample) > 0 and any(
                isinstance(v, (list, dict, tuple)) for v in sample
            ):
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict, tuple)) else x
                )
                converted_cols.append(col)

    return df, converted_cols


def _convert_json_columns_from_parquet(
    df: pd.DataFrame, json_columns: list[str] | None = None
) -> pd.DataFrame:
    """Convert JSON string columns back to Python objects after loading from parquet.

    Args:
        df: DataFrame loaded from parquet
        json_columns: List of column names to convert (if None, auto-detect)

    Returns:
        DataFrame with JSON columns converted back to Python objects
    """
    import json

    df = df.copy()

    # Auto-detect JSON columns if not specified
    if json_columns is None:
        json_columns = []
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # Check if values look like JSON strings
                    for v in sample:
                        if isinstance(v, str) and v.startswith(("[", "{")):
                            json_columns.append(col)
                            break

    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x)
                if isinstance(x, str) and x.startswith(("[", "{"))
                else x
            )

    return df


def _save_dataframe(
    df: pd.DataFrame,
    path: Path,
    base_name: str,
    storage_format: StorageFormat,
    index_label: str = "index",
) -> Path:
    """Save a DataFrame in the specified format.

    Args:
        df: DataFrame to save
        path: Directory to save to
        base_name: Base filename without extension (e.g., "details", "all_scores")
        storage_format: Format to use (csv, gzip, parquet)
        index_label: Label for the index column (used for CSV formats)

    Returns:
        Path to the saved file
    """
    ext = _get_file_extension(storage_format)
    file_path = path / f"{base_name}{ext}"

    if storage_format == "csv":
        df.to_csv(file_path, index_label=index_label)
    elif storage_format == "gzip":
        df.to_csv(file_path, index_label=index_label, compression="gzip")
    else:  # "parquet"
        # Reset index to include it as a column, then save
        df_to_save = df.reset_index(drop=False)
        # Convert complex object columns to JSON strings for parquet compatibility
        df_to_save, _ = _convert_complex_columns_to_json(df_to_save)
        df_to_save.to_parquet(file_path, index=False)

    return file_path


def _load_dataframe(
    path: Path, base_name: str, header: int | list[int] = 0, index_col: int = 0
) -> pd.DataFrame | None:
    """Load a DataFrame, auto-detecting the format from available files.

    Tries to load in order: parquet, gzip, csv

    Args:
        path: Directory containing the file
        base_name: Base filename without extension (e.g., "details", "all_scores")
        header: Row(s) to use as header (for CSV formats)
        index_col: Column to use as index (for CSV formats)

    Returns:
        Loaded DataFrame or None if no file found
    """
    # Try parquet first (best format)
    parquet_path = path / f"{base_name}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        # Restore index from the 'index' column if present
        if "index" in df.columns:
            df = df.set_index("index")
        # Convert JSON string columns back to Python objects
        df = _convert_json_columns_from_parquet(df)
        return df

    # Try gzip
    gzip_path = path / f"{base_name}.csv.gz"
    if gzip_path.exists():
        return pd.read_csv(gzip_path, header=header, index_col=index_col)

    # Try plain csv
    csv_path = path / f"{base_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, header=header, index_col=index_col)

    return None


def _find_dataframe_file(path: Path, base_name: str) -> Path | None:
    """Find the file for a DataFrame, checking all supported formats.

    Args:
        path: Directory to search
        base_name: Base filename without extension

    Returns:
        Path to the file if found, None otherwise
    """
    for ext in (".parquet", ".csv.gz", ".csv"):
        file_path = path / f"{base_name}{ext}"
        if file_path.exists():
            return file_path
    return None


__all__ = [
    "run_negotiation",
    "cartesian_tournament",
    "continue_cartesian_tournament",
    "SimpleTournamentResults",
    "combine_tournaments",
    "RunInfo",
    "ConstructedNegInfo",
    "ProgressCallback",
    "BeforeStartCallback",
    "AfterEndCallback",
    "AfterConstructionCallback",
    "OptimizationLevel",
    "StorageFormat",
]
MAX_TASKS_PER_CHILD = 10
LOG_UNIFORM_LIMIT = 10
TERMINATION_WAIT_TIME = 10.0

DEFAULT_IMAGE_FORMAT = "webp"
SUPPORTED_IMAGE_FORMATS = {"webp", "png", "jpg", "jpeg", "svg", "pdf"}

EXTENSION = DEFAULT_TABLE_STORAGE_FORMAT
ALL_SCORES_FILE_NAME = f"all_scores{EXTENSION}"
ALL_RESULTS_FILE_NAME = f"details{EXTENSION}"
TYPE_SCORES_FILE_NAME = f"type_scores{EXTENSION}"
FINAL_SCORES_FILE_NAME = f"scores{EXTENSION}"
NEGOTIATOR_BEHAVIOR_DIR_NAME = "negotiator_behavior"
CONFIG_FILE_NAME = "config.yaml"
SCENARIOS_DIR_NAME = "scenarios"
PLOTS_DIR_NAME = "plots"
NEGOTIATIONS_DIR_NAME = "negotiations"
RESULTS_DIR_NAME = "results"
TOURNAMENT_COL_NAME = "tournament"
OPTIONAL_COLS = (TOURNAMENT_COL_NAME,)
OPTIMALITY_COLS = (
    "nash_optimality",
    "kalai_optimality",
    "ks_optimality",
    "max_welfare_optimality",
    "pareto_optimality",
    "fairness",  # max(nash_optimality, kalai_optimality, ks_optimality)
    "modified_kalai_optimality",
    "modified_ks_optimality",
)

# Note: NEGOTIATOR_BEHAVIOR_DIR_NAME is deprecated and kept only for backward
# compatibility when loading old tournaments. New tournaments save negotiation
# traces in the negotiations/ folder.
TOURNAMENT_DIRS = [
    SCENARIOS_DIR_NAME,
    NEGOTIATOR_BEHAVIOR_DIR_NAME,  # Deprecated: kept for backward compatibility
    PLOTS_DIR_NAME,
    NEGOTIATIONS_DIR_NAME,
    RESULTS_DIR_NAME,
]
TOURNAMENT_FILES = [
    ALL_SCORES_FILE_NAME,
    ALL_RESULTS_FILE_NAME,
    TYPE_SCORES_FILE_NAME,
    FINAL_SCORES_FILE_NAME,
]
MECHANISM_FILE_NAME = "mechanism.json"

# Columns in details.csv that contain list/tuple values (need parsing when reading CSV)
LIST_COLUMNS = (
    "utilities",
    "max_utils",
    "reserved_values",
    "partners",
    "params",
    "negotiator_names",
    "negotiator_ids",
    "negotiator_types",
    "negotiator_times",
    "agreement",
    "scored_indices",
)


def _parse_list_value(value: Any) -> Any:
    """Parse a string representation of a list/tuple back to a Python object.

    When DataFrames with list columns are saved to CSV, the lists become string
    representations like "[1, 2, 3]" or "('a', 'b')". This function parses them back.

    Args:
        value: The value to parse. If it's a string that looks like a list/tuple,
               it will be parsed. Otherwise, the original value is returned.

    Returns:
        The parsed Python object, or the original value if parsing fails or isn't needed.
    """
    if pd.isna(value):
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    # Try to parse as a Python literal (list, tuple, dict, etc.)
    if value.startswith(("[", "(", "{")):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    # Handle "None" string
    if value == "None":
        return None
    return value


def _details_df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a details DataFrame back to a list of record dictionaries.

    This function parses list-like string columns back into actual Python lists/tuples
    so that the records can be used with make_scores().

    Args:
        df: DataFrame loaded from details.csv

    Returns:
        List of record dictionaries with properly parsed list columns.
    """
    records = df.to_dict("records")
    for record in records:
        for col in LIST_COLUMNS:
            if col in record:
                record[col] = _parse_list_value(record[col])
    return records


def _combine_configs(
    configs: list[dict[str, Any]], n_unique_scenarios: int
) -> dict[str, Any]:
    """Combine multiple tournament configs into a single config.

    For boolean values that differ across configs, the combined value is None.
    For values that are the same across all configs, the combined value is that value.
    n_scenarios is set to the provided n_unique_scenarios count.

    Args:
        configs: List of config dictionaries to combine
        n_unique_scenarios: Number of unique scenarios in the combined tournament

    Returns:
        Combined config dictionary

    Raises:
        ValueError: If competitors or opponents don't match across configs
    """
    if not configs:
        return {}

    if len(configs) == 1:
        result = configs[0].copy()
        result["n_scenarios"] = n_unique_scenarios
        return result

    # Validate that competitors and opponents match
    first = configs[0]
    for i, cfg in enumerate(configs[1:], 2):
        if cfg.get("competitors") != first.get("competitors"):
            raise ValueError(
                f"Cannot combine tournaments with different competitors. "
                f"Tournament 1 has {first.get('competitors')}, "
                f"tournament {i} has {cfg.get('competitors')}"
            )
        if cfg.get("competitor_names") != first.get("competitor_names"):
            raise ValueError(
                f"Cannot combine tournaments with different competitor_names. "
                f"Tournament 1 has {first.get('competitor_names')}, "
                f"tournament {i} has {cfg.get('competitor_names')}"
            )
        if cfg.get("competitor_params") != first.get("competitor_params"):
            raise ValueError(
                f"Cannot combine tournaments with different competitor_params. "
                f"Tournament 1 has {first.get('competitor_params')}, "
                f"tournament {i} has {cfg.get('competitor_params')}"
            )
        if cfg.get("opponents") != first.get("opponents"):
            raise ValueError(
                f"Cannot combine tournaments with different opponents. "
                f"Tournament 1 has {first.get('opponents')}, "
                f"tournament {i} has {cfg.get('opponents')}"
            )
        if cfg.get("opponent_names") != first.get("opponent_names"):
            raise ValueError(
                f"Cannot combine tournaments with different opponent_names. "
                f"Tournament 1 has {first.get('opponent_names')}, "
                f"tournament {i} has {cfg.get('opponent_names')}"
            )
        if cfg.get("opponent_params") != first.get("opponent_params"):
            raise ValueError(
                f"Cannot combine tournaments with different opponent_params. "
                f"Tournament 1 has {first.get('opponent_params')}, "
                f"tournament {i} has {cfg.get('opponent_params')}"
            )

    # Get all keys from all configs
    all_keys = set()
    for cfg in configs:
        all_keys.update(cfg.keys())

    result: dict[str, Any] = {}
    for key in all_keys:
        values = [cfg.get(key) for cfg in configs if key in cfg]

        if not values:
            continue

        # Check if all values are the same
        first_value = values[0]
        all_same = all(v == first_value for v in values)

        if all_same:
            result[key] = first_value
        elif all(isinstance(v, bool) or v is None for v in values):
            # Boolean values that differ become None
            result[key] = None
        else:
            # For non-boolean differing values, use None
            result[key] = None

    # Override specific fields
    result["n_scenarios"] = n_unique_scenarios
    result["path"] = None  # Combined tournament doesn't have a single source path

    return result


@define
class RunInfo:
    """Information about a negotiation run before it starts.

    Passed to before_start_callback in cartesian_tournament to allow inspection
    and logging of negotiation parameters before execution.

    Attributes:
        s: The scenario containing outcome space and utility functions
        partners: Tuple of negotiator classes that will participate
        mechanism_type: The mechanism class that will be instantiated
        mechanism_params: Parameters that will be passed to the mechanism
        n_repetitions: Total number of repetitions for this scenario/partner combination
        rep: Current repetition index (0-based)
        scenario_index: Index of this scenario in the tournament
        negotiation_id: Unique identifier for this negotiation run
        scored_indices: Indices of negotiators whose scores will be recorded (None means all)
        config: Tournament configuration dictionary (same as saved to config.yaml)
    """

    s: Scenario
    run_id: int | str
    partners: tuple[type[Negotiator], ...]
    partner_names: tuple[str] | None = None
    partner_params: tuple[dict[str, Any]] | None = None
    rep: int = 0
    path: Path | None = None
    mechanism_type: type[Mechanism] = SAOMechanism
    mechanism_params: dict[str, Any] | None = None
    full_names: bool = True
    verbosity: int = 0
    plot: bool = False
    plot_params: dict[str, Any] | None = None
    stats: ScenarioStats | None = None
    annotation: dict[str, Any] | None = None
    private_infos: tuple[dict[str, Any] | None] | None = None
    id_reveals_type: bool = False
    name_reveals_type: bool = True
    mask_scenario_name: bool = True
    ignore_exceptions: bool = False
    scored_indices: list[int] | None = None
    n_repetitions: int = 1
    scenario_index: int = 0
    config: dict[str, Any] = field(factory=dict)


@define
class ConstructedNegInfo:
    """Information about a negotiation after mechanism construction.

    Passed to after_construction_callback in cartesian_tournament to allow inspection
    and modification of the mechanism before execution.

    Attributes:
        run_id: Unique identifier for this negotiation run
        mechanism: The constructed mechanism instance
        failures: Dictionary of any failures during construction
        scenario: The scenario being negotiated
        real_scenario_name: The actual scenario name (may differ from masked name)
        config: Tournament configuration dictionary (same as saved to config.yaml)
    """

    run_id: int | str | None
    mechanism: Mechanism
    failures: dict
    scenario: Scenario
    real_scenario_name: str | None
    config: dict[str, Any] = field(factory=dict)


BeforeStartCallback = Callable[[RunInfo], None]
NegStartCallback = Callable[[str | int, SAOState], None]
NegProgressCallback = Callable[[str | int, SAOState], None]
NegEndCallback = Callable[[str | int, SAOState], None]
AfterConstructionCallback = Callable[[ConstructedNegInfo], None]
# AfterEndCallback can accept either (record) or (record, config)
AfterEndCallback = (
    Callable[[dict[str, Any]], None] | Callable[[dict[str, Any], dict[str, Any]], None]
)
# ProgressCallback can accept either (message, current, total) or (message, current, total, config)
ProgressCallback = (
    Callable[[str, int, int], None]
    | Callable[[str, int, int, dict[str, Any] | None], None]
)


def _call_after_end_callback(
    callback: AfterEndCallback, record: dict[str, Any], config: dict[str, Any] | None
) -> None:
    """Call after_end_callback with backwards compatibility.

    Tries to call with (record, config) first. If that fails with TypeError
    (wrong number of arguments), falls back to (record) only.
    """
    try:
        callback(record, config)  # type: ignore
    except TypeError:
        # Fall back to old signature without config
        callback(record)  # type: ignore


def _call_progress_callback(
    callback: ProgressCallback,
    message: str,
    current: int,
    total: int,
    config: dict[str, Any] | None,
) -> None:
    """Call progress_callback with backwards compatibility.

    Tries to call with (message, current, total, config) first. If that fails
    with TypeError (wrong number of arguments), falls back to (message, current, total) only.
    """
    try:
        callback(message, current, total, config)  # type: ignore
    except TypeError:
        # Fall back to old signature without config
        callback(message, current, total)  # type: ignore


class SimpleTournamentResults:
    """Tournament results with optional lazy loading and memory optimization.

    This class stores tournament results and supports different memory optimization
    levels that control whether data is kept in memory or loaded from disk on demand.

    Attributes:
        scores: All scores per negotiator (may be loaded lazily)
        details: All negotiation results (may be loaded lazily)
        scores_summary: All score statistics summarized per type (always in memory)
        final_scores: Final rankings sorted from highest to lowest score (always in memory)
        path: Location at which the logs are stored
        memory_optimization: Memory optimization level ("speed", "balanced", "space")
        storage_optimization: Storage optimization level ("speed", "balanced", "space")
        storage_format: Storage format for data files ("csv", "gzip", "parquet")
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        scores: pd.DataFrame | None = None,
        details: pd.DataFrame | None = None,
        scores_summary: pd.DataFrame | None = None,
        final_scores: pd.DataFrame | None = None,
        path: Path | None = None,
        memory_optimization: OptimizationLevel = "balanced",
        storage_optimization: OptimizationLevel = "space",
        storage_format: StorageFormat | None = None,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
    ):
        """Initialize SimpleTournamentResults.

        Args:
            config: The complete configuration of the tournament
            scores: All scores per negotiator
            details: All negotiation results
            scores_summary: All score statistics summarized per type
            final_scores: Final rankings
            path: Location at which the logs are stored
            memory_optimization: Memory optimization level:
                - "speed"/"time"/"none": Keep all data in memory
                - "balanced": Keep details + final_scores + scores_summary in memory,
                              compute scores on demand then cache (default)
                - "space"/"max": Keep only final_scores + scores_summary in memory,
                          load details/scores from disk when needed
            storage_optimization: Storage optimization level:
                - "speed"/"time"/"none": Keep all files (results/, all_scores.csv, details.csv, etc.)
                - "balanced": Remove results/ folder after details.csv is created
                - "space"/"max": Remove both results/ folder AND all_scores.csv
            storage_format: Storage format for data files:
                - "csv": Plain CSV files (human-readable, larger size)
                - "gzip": Gzip-compressed CSV files (good compression, human-readable when decompressed)
                - "parquet": Parquet binary format (best compression, preserves types, fastest)
                - None: Auto-select based on storage_optimization (csv for speed, gzip for balanced, parquet for space)
            final_score_stat: Tuple of (measure, statistic) for final score calculation
        """
        self._path = Path(path) if path else None
        # Normalize optimization levels (treat "time"/"none" as "speed", "max" as "space")
        memory_optimization = _normalize_optimization_level(memory_optimization)
        storage_optimization = _normalize_optimization_level(storage_optimization)
        self._memory_optimization: OptimizationLevel = memory_optimization
        self._storage_optimization: OptimizationLevel = storage_optimization
        # Set storage format (auto-select if None)
        self._storage_format: StorageFormat = (
            storage_format
            if storage_format is not None
            else _get_default_storage_format(storage_optimization)
        )
        self._final_score_stat = final_score_stat

        # If no path is provided, we must keep everything in memory
        effective_memory_opt = memory_optimization if self._path else "speed"

        # scores_summary and final_scores are always kept in memory (they're small)
        self._scores_summary = (
            scores_summary if scores_summary is not None else pd.DataFrame()
        )
        self._final_scores = (
            final_scores if final_scores is not None else pd.DataFrame()
        )
        self._config = config

        # For scores and details, behavior depends on memory optimization
        if effective_memory_opt == "speed":
            # Keep everything in memory
            self._scores: pd.DataFrame | None = (
                scores if scores is not None else pd.DataFrame()
            )
            self._details: pd.DataFrame | None = (
                details if details is not None else pd.DataFrame()
            )
            self._scores_cached = True
            self._details_cached = True
        elif effective_memory_opt == "balanced":
            # Keep details in memory, compute scores on demand
            self._details = details if details is not None else pd.DataFrame()
            self._details_cached = True
            self._scores = scores  # May be None, will compute on demand
            self._scores_cached = scores is not None
        else:  # "space"
            # Load everything from disk on demand
            self._scores = scores  # May be None
            self._details = details  # May be None
            self._scores_cached = scores is not None
            self._details_cached = details is not None

    @property
    def config(self) -> dict[str, Any] | None:
        """The complete configuration of the tournament."""
        return self._config

    @property
    def path(self) -> Path | None:
        """Location at which the logs are stored."""
        return self._path

    @property
    def memory_optimization(self) -> OptimizationLevel:
        """Memory optimization level."""
        return self._memory_optimization

    @property
    def storage_optimization(self) -> OptimizationLevel:
        """Storage optimization level."""
        return self._storage_optimization

    @property
    def storage_format(self) -> StorageFormat:
        """Storage format for data files."""
        return self._storage_format

    @property
    def scores_summary(self) -> pd.DataFrame:
        """All score statistics summarized per type."""
        return self._scores_summary

    @property
    def final_scores(self) -> pd.DataFrame:
        """Final rankings sorted from highest to lowest score."""
        return self._final_scores

    @property
    def details(self) -> pd.DataFrame:
        """All negotiation results (may be loaded from disk)."""
        if self._details is not None and (
            self._details_cached or len(self._details) > 0
        ):
            return self._details

        # Try to load from disk (auto-detect format)
        if self._path is not None:
            df = _load_dataframe(self._path, "details", header=0, index_col=0)
            if df is not None:
                self._details = df
                self._details_cached = True
                return self._details

            # Try to reconstruct from results/ folder
            results_dir = self._path / RESULTS_DIR_NAME
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                if json_files:
                    self._details = pd.DataFrame.from_records(
                        [load(_) for _ in json_files]
                    )
                    self._details_cached = True
                    return self._details

        # Return empty DataFrame if nothing available
        if self._details is None:
            self._details = pd.DataFrame()
        return self._details

    @property
    def scores(self) -> pd.DataFrame:
        """All scores per negotiator (may be computed or loaded from disk)."""
        if self._scores is not None and (self._scores_cached or len(self._scores) > 0):
            return self._scores

        # Try to load from disk first (auto-detect format)
        if self._path is not None:
            df = _load_dataframe(self._path, "all_scores", header=0, index_col=0)
            if df is not None:
                self._scores = df
                self._scores_cached = True
                return self._scores

        # Try to compute from results/ folder JSON files (best source - preserves types)
        if self._path is not None:
            results_dir = self._path / RESULTS_DIR_NAME
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                if json_files:
                    records = [load(_) for _ in json_files]
                    # make_scores returns a list of score dicts per record
                    # We need to flatten all these lists into one list
                    all_scores = []
                    for record in records:
                        all_scores += make_scores(
                            record, scored_indices=record.get("scored_indices")
                        )
                    self._scores = pd.DataFrame.from_records(all_scores)
                    self._scores_cached = True
                    return self._scores

        # Try to reconstruct from details DataFrame (handles parsed CSV)
        details_df = self.details
        if len(details_df) > 0:
            records = _details_df_to_records(details_df)
            all_scores = []
            for record in records:
                all_scores += make_scores(
                    record, scored_indices=record.get("scored_indices")
                )
            if all_scores:
                self._scores = pd.DataFrame.from_records(all_scores)
                self._scores_cached = True
                return self._scores

        # Return empty DataFrame if nothing available
        if self._scores is None:
            self._scores = pd.DataFrame()
        return self._scores

    @classmethod
    def from_records(
        cls,
        config: dict[str, Any] | None = None,
        scores: list[dict[str, Any]] | pd.DataFrame | None = None,
        results: list[dict[str, Any]] | pd.DataFrame | None = None,
        type_scores: pd.DataFrame | None = None,
        final_scores: pd.DataFrame | None = None,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
        path: Path | None = None,
        stats_aggregated_metrics: dict[
            str, Callable[[dict[tuple[str, str], float]], float]
        ]
        | None = None,
    ) -> "SimpleTournamentResults":
        """Creates SimpleTournamentResults from records of results

        Args:
            scores: The scores of negotiators in all negotiations (If not given, `results` can be used to calculate it).
            results: Results of all negotiations (If not given, the resulting SimpleTournamentResults object will lack details)
            type_scores: Optionally, type-scores. If not given, it will be calculated from scores
            final_scores: Optionally, final scores. If not given, `final_scoer_stat` will be used to calculate them
            final_score_stat: A tuple of the measure used and the statistic applied to it for calculating final score. See `cartesian_tournament` for more details
            path: The path in which the data for this tournament is stored.
            stats_aggregated_metrics: Optional dict mapping new metric names to callables that receive
                                    a dict of (metric, stat) tuples to values and return a combined score.

        Raises:
            ValueError: If no scores or results are given

        Returns:
            A new SimpleTournamentResults with the given data
        """
        if scores is None and results is None:
            raise ValueError("Cannot pass both scoers and results as None")
        if config is None:
            config = load(path / CONFIG_FILE_NAME) if path else dict()
        if scores is None or (
            len(scores) == 0 and results is not None and len(results) > 0
        ):
            rd = (
                results.to_dict("records")
                if isinstance(results, pd.DataFrame)
                else results
            )
            assert rd is not None
            scores = pd.DataFrame.from_records(
                [make_scores(_, scored_indices=_.get("scored_indices")) for _ in (rd)]
            )
        if results is None:
            results = pd.DataFrame()

        if not isinstance(scores, pd.DataFrame):
            scores_df = pd.DataFrame.from_records(scores)
        else:
            scores_df = scores
        if len(scores_df) > 0:
            if type_scores is None:
                type_scores = pd.DataFrame()
                if scores_df is not None and len(scores_df) > 0:
                    cols = [
                        _
                        for _ in scores_df.columns
                        if _ not in ("scenario", "partners")
                    ]
                    # Create type_scores without sorting first - we'll sort after applying stats_aggregated_metrics
                    type_scores = scores_df.loc[:, cols].groupby("strategy").describe()

            # Apply stats_aggregated_metrics to add new columns to type_scores
            if (
                stats_aggregated_metrics
                and type_scores is not None
                and len(type_scores) > 0
            ):
                for metric_name, aggregator in stats_aggregated_metrics.items():
                    new_values = []
                    for strategy in type_scores.index:
                        # Build dict of (metric, stat) -> value for this strategy
                        stats_dict: dict[tuple[str, str], float] = {}
                        for col in type_scores.columns:
                            if isinstance(col, tuple) and len(col) == 2:
                                metric, stat = col
                                try:
                                    val = type_scores.loc[strategy, col]
                                    if isinstance(val, (int, float)):
                                        stats_dict[(metric, stat)] = float(val)
                                except Exception:
                                    pass
                        try:
                            new_values.append(aggregator(stats_dict))
                        except Exception:
                            new_values.append(float("nan"))
                    # Add new column with (metric_name, "value") as the column name
                    type_scores[(metric_name, "value")] = new_values

            # Sort by final_score_stat after applying stats_aggregated_metrics
            # This allows using custom aggregations as the final score
            if type_scores is not None and len(type_scores) > 0:
                if final_score_stat in type_scores.columns:
                    type_scores = type_scores.sort_values(
                        final_score_stat, ascending=False
                    )

            if final_scores is None:
                final = pd.DataFrame()
                if type_scores is not None and len(type_scores) > 0:
                    final = type_scores[final_score_stat]
                    final.name = "score"
                    final = final.reset_index()
            else:
                final = final_scores
            if not isinstance(results, pd.DataFrame):
                details_df = pd.DataFrame.from_records(results)
            else:
                details_df = results
        else:
            details_df = type_scores = final = pd.DataFrame()
        return SimpleTournamentResults(
            config=config,
            scores=scores_df,
            details=details_df,
            scores_summary=type_scores,  # type: ignore
            final_scores=final,
            path=path,
        )

    @classmethod
    def from_result_records(
        cls,
        path: Path,
        verbosity: int = 1,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
    ) -> "SimpleTournamentResults":
        """From result records.

        Args:
            path: Path.
            verbosity: Verbosity.
            final_score_stat: Final score stat.

        Returns:
            'SimpleTournamentResults': The result.
        """
        return cls.combine(
            [Path(path)],
            recursive=False,
            recalc_details=True,
            recalc_scores=True,
            must_have_details=True,
            verbosity=verbosity,
            final_score_stat=final_score_stat,
        )[0]

    @classmethod
    def combine(
        cls,
        paths: Path | Iterable[Path],
        recursive: bool = True,
        recalc_details: bool = True,
        recalc_scores: bool = False,
        must_have_details: bool = False,
        verbosity: int = 1,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
        add_tournament_column: bool = True,
        complete_only: bool = True,
    ) -> tuple["SimpleTournamentResults", list[Path]]:
        """Combines the results of multiple tournaments stored on disk.

        This method can combine tournaments saved with different storage formats
        (csv, gzip, parquet) and different optimization levels. It auto-detects
        the format of each tournament's files.

        Args:
            paths: Paths to look for results within
            recursive: Check children of given paths recursively
            recalc_details: Recalculate detailed results from the `results/` folder
            recalc_scores: Recalculate scores from detailed negotiation results
            must_have_details: Raise an exception if detailed negotiation results cannot be found
            verbosity: Verbosity level (0=silent, 1=basic, 2+=detailed)
            final_score_stat: Tuple of (measure, statistic) for final score calculation.
                See `cartesian_tournament` for details.
            add_tournament_column: Add a column called "tournament" with tournament name
                in details and scores DataFrames.
            complete_only: If True, only include tournaments that completed successfully
                (have all required files). Incomplete tournaments are ignored.

        Returns:
            A tuple of:
                - SimpleTournamentResults with the combined results of all tournaments
                - List of Paths that were successfully loaded

        Raises:
            FileNotFoundError: If no valid tournament paths are found or if a
                needed file is missing (when must_have_details=True)

        Notes:
            **File Format Handling**:
                - Tournaments with different storage formats can be combined
                - Each tournament's format is auto-detected independently
                - The combined result uses the default format (csv)

            **Required Files** (when complete_only=True):
                - details.csv/.csv.gz/.parquet (any format)
                - all_scores.csv/.csv.gz/.parquet (any format)
                - scores.csv (final scores, always CSV)

            **Data Reconstruction**:
                - If recalc_details=True, details are rebuilt from results/ folder
                - If recalc_scores=True, scores are rebuilt from details
                - This allows combining tournaments with different storage_optimization levels
        """
        """Loads results from the given paths (recursively if given)"""
        if isinstance(paths, Path):
            paths = [paths]
        assert isinstance(paths, Iterable)
        if complete_only:
            recalc_details = False
            recalc_scores = False
            must_have_details = True

        needed_files: list[tuple[str, str] | str] = []
        if complete_only:
            needed_files += [
                ALL_RESULTS_FILE_NAME,
                ALL_SCORES_FILE_NAME,
                FINAL_SCORES_FILE_NAME,
            ]
        else:
            if recalc_details:
                needed_files.append(RESULTS_DIR_NAME)
            elif must_have_details:
                needed_files.append((ALL_RESULTS_FILE_NAME, RESULTS_DIR_NAME))
            if recalc_scores:
                needed_files.append((ALL_RESULTS_FILE_NAME, RESULTS_DIR_NAME))

        if recursive:
            known_dirs = set(TOURNAMENT_DIRS)
            found_dirs = set()
            for path in track(paths, "Walking Tree") if verbosity else paths:
                for base, dirs, _ in os.walk(path):
                    p = Path(base).absolute()
                    if has_needed_files(p, needed_files):
                        found_dirs.add(p)
                    remaining_dirs = []
                    for d in dirs:
                        if d in known_dirs:
                            continue
                        p = (Path(base) / d).absolute()
                        if has_needed_files(p, needed_files):
                            found_dirs.add(p)
                        else:
                            remaining_dirs.append(d)
                    dirs = remaining_dirs
            paths = list(found_dirs)
        else:
            paths = [_.absolute() for _ in paths if has_needed_files(_, needed_files)]
        if not paths:
            raise FileNotFoundError(
                "None of the given paths has the needed files to reconstruct the results of a tournament"
            )
        results, scores = [], []
        loaded_paths = set()
        path_names = shortest_unique_names([str(_) for _ in paths], sep=os.sep)
        for path, pname in (
            track(zip(paths, path_names), "Reading ... ", total=len(paths))
            if verbosity
            else zip(paths, path_names)
        ):
            if verbosity > 1:
                print(f"Reading {path}")
            if recalc_details or not (path / ALL_RESULTS_FILE_NAME).exists():
                src = path / RESULTS_DIR_NAME
                d = pd.DataFrame.from_records([load(_) for _ in src.glob("*.json")])
            else:
                d = pd.read_csv(path / ALL_RESULTS_FILE_NAME, index_col=0)
            if add_tournament_column:
                d[TOURNAMENT_COL_NAME] = pname
            if must_have_details and len(d) < 1:
                print(
                    f"Cannot find detailed results in {path / ALL_RESULTS_FILE_NAME} and you specified `must_have_details` ... Will ignore it"
                )
                continue
            if recalc_scores or not (path / ALL_SCORES_FILE_NAME).exists():
                if len(d) <= 0:
                    if verbosity:
                        print(
                            f"Failed to calculate scores for {path / ALL_SCORES_FILE_NAME} ... Will ignore it"
                        )
                    continue
                s = pd.DataFrame.from_records(
                    [
                        make_scores(_, scored_indices=_.get("scored_indices"))
                        for _ in d.to_dict("records")
                    ]
                )
            else:
                s = pd.read_csv(path / ALL_SCORES_FILE_NAME, index_col=0)
            if add_tournament_column:
                s[TOURNAMENT_COL_NAME] = pname
            if len(d) > 0:
                loaded_paths.add(path)
                results.append(d)
            if len(s) > 0:
                loaded_paths.add(path)
                scores.append(s)
        if len(scores) < 1:
            raise FileNotFoundError("Cannot find any records or details to use")

        # Load and combine configs from all paths
        configs = []
        for path in loaded_paths:
            config_path = path / CONFIG_FILE_NAME
            if config_path.exists():
                configs.append(load(config_path))

        # Combine results to count unique scenarios
        combined_results = pd.concat(results, ignore_index=True)
        combined_scores = pd.concat(scores, ignore_index=True)

        # Count unique scenarios from the combined results
        n_unique_scenarios = int(
            combined_results["scenario"].nunique()
            if "scenario" in combined_results.columns
            else 0
        )

        # Combine configs (validates competitors/opponents match)
        config = _combine_configs(configs, n_unique_scenarios) if configs else None

        return cls.from_records(
            config=config,
            scores=combined_scores,
            results=combined_results,
            final_score_stat=final_score_stat,
        ), list(loaded_paths)

    @classmethod
    def load(
        cls,
        path: Path,
        must_have_details: bool = False,
        memory_optimization: OptimizationLevel = "balanced",
        storage_optimization: OptimizationLevel = "space",
        storage_format: StorageFormat | None = None,
    ) -> "SimpleTournamentResults":
        """Loads tournament results from the given path.

        This method auto-detects the storage format and reconstructs scores
        if needed. It supports loading from tournaments saved with any
        storage_optimization level.

        Args:
            path: Path to load results from
            must_have_details: If True, raise an error if details are not found
            memory_optimization: Memory optimization level for the loaded results:
                - "speed": Keep all data in memory
                - "balanced": Keep details in memory, compute scores on demand
                - "space": Load data from disk on demand
            storage_optimization: Storage optimization level (for metadata only,
                does not affect loading behavior)
            storage_format: Storage format hint (auto-detected if None from
                existing files)

        Returns:
            SimpleTournamentResults loaded from disk

        Raises:
            FileNotFoundError: If required files are not found

        Notes:
            **File Format Detection Priority** (for details and all_scores):
                1. Parquet (.parquet) - Best compression, preserves types
                2. Gzip (.csv.gz) - Good compression
                3. CSV (.csv) - Plain text

            **Data Reconstruction Priority** (if scores/details not found):
                1. Load from data files (details.parquet/csv.gz/csv, all_scores.parquet/csv.gz/csv)
                2. Reconstruct from results/ folder JSON files
                3. Reconstruct scores from details DataFrame

            Small files (scores.csv, type_scores.csv) are always stored as plain CSV.
        """
        kwargs: dict[str, Any] = dict(
            path=path,
            memory_optimization=memory_optimization,
            storage_optimization=storage_optimization,
            storage_format=storage_format,
        )

        # Load config if exists
        config_path = path / CONFIG_FILE_NAME
        if config_path.exists():
            kwargs["config"] = load(config_path)

        # Load scores_summary and final_scores (always CSV format, small files)
        for k, base_name, required, header, index_col in (
            ("scores_summary", "type_scores", must_have_details, [0, 1], 0),
            ("final_scores", "scores", True, 0, 0),
        ):
            # These are always CSV
            csv_path = path / f"{base_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, header=header, index_col=index_col)
                kwargs[k] = df
            elif required:
                raise FileNotFoundError(f"{base_name}.csv not found in {path}")

        # Load details and all_scores (auto-detect format)
        for k, base_name in (("scores", "all_scores"), ("details", "details")):
            df = _load_dataframe(path, base_name, header=0, index_col=0)
            if df is not None:
                kwargs[k] = df

        # If scores or details not found, try to reconstruct from results/ folder
        if "scores" not in kwargs or "details" not in kwargs:
            results_dir = path / RESULTS_DIR_NAME
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                if json_files:
                    records = [load(_) for _ in json_files]
                    if "details" not in kwargs:
                        kwargs["details"] = pd.DataFrame.from_records(records)
                    if "scores" not in kwargs:
                        # make_scores returns a list of score dicts per record
                        # We need to flatten all these lists into one list
                        all_scores = []
                        for record in records:
                            all_scores += make_scores(
                                record, scored_indices=record.get("scored_indices")
                            )
                        kwargs["scores"] = pd.DataFrame.from_records(all_scores)

        # If scores still not found but details exist, reconstruct from details
        # This handles the case where results/ folder was deleted (balanced/space optimization)
        if "scores" not in kwargs and "details" in kwargs:
            details_df = kwargs["details"]
            if len(details_df) > 0:
                # For parquet, types are preserved; for CSV, we need to parse
                if _find_dataframe_file(path, "details") and str(
                    _find_dataframe_file(path, "details")
                ).endswith(".parquet"):
                    records = details_df.to_dict("records")
                else:
                    records = _details_df_to_records(details_df)
                all_scores = []
                for record in records:
                    all_scores += make_scores(
                        record, scored_indices=record.get("scored_indices")
                    )
                if all_scores:
                    kwargs["scores"] = pd.DataFrame.from_records(all_scores)

        if must_have_details and "details" not in kwargs:
            raise FileNotFoundError(
                f"Cannot find details in {path} and results/ folder not available"
            )

        return SimpleTournamentResults(**kwargs)

    def save(
        self,
        path: Path | None = None,
        exist_ok: bool = True,
        storage_optimization: OptimizationLevel | None = None,
        storage_format: StorageFormat | None = None,
    ) -> None:
        """Save all results to the given path.

        Args:
            path: Path to save results to. If None, uses self.path
            exist_ok: If True, don't raise an error if the directory exists
            storage_optimization: Override the instance's storage_optimization setting.
                - "speed": Save all files (scores, details, scores_summary, final_scores)
                - "balanced": Same as speed (cleanup happens in cartesian_tournament)
                - "space": Skip saving all_scores (can be reconstructed from details)
            storage_format: Override the instance's storage_format setting.
                - "csv": Plain CSV files
                - "gzip": Gzip-compressed CSV files
                - "parquet": Parquet binary format
        """
        if path is None:
            path = self.path
        if path is None:
            raise FileNotFoundError(
                "You must pass path to save or have a path in the tournament to save it"
            )
        path = Path(path).absolute()
        path.mkdir(exist_ok=exist_ok, parents=True)

        # Use provided options or fall back to instance settings
        effective_storage_opt = (
            _normalize_optimization_level(storage_optimization)
            if storage_optimization is not None
            else self._storage_optimization
        )
        effective_format = (
            storage_format if storage_format is not None else self._storage_format
        )

        # Save files that use the storage format (all_scores, details)
        for df, base_name, skip_on_space in (
            (self.scores, "all_scores", True),  # Skip scores in "space" mode
            (self.details, "details", False),
        ):
            # Skip all_scores in "space" mode
            if skip_on_space and effective_storage_opt == "space":
                continue
            if df is not None and len(df) > 0:
                _save_dataframe(df, path, base_name, effective_format)

        # Save files that are always CSV (small files: scores_summary, final_scores)
        for df, base_name in (
            (self.scores_summary, "type_scores"),
            (self.final_scores, "scores"),
        ):
            if df is not None and len(df) > 0:
                df.to_csv(path / f"{base_name}.csv", index_label="index")


def combine_tournaments(
    srcs: Path | Iterable[Path],
    dst: Path | None = None,
    *,
    recursive: bool = True,
    recalc_details: bool = True,
    recalc_scores: bool = False,
    must_have_details: bool = False,
    verbosity: int = 1,
    final_score_stat: tuple[str, str] = ("advantage", "mean"),
    copy: bool = False,
    rename_scenarios: bool = True,
    rename_short: bool = True,
    add_tournament_folders: bool = True,
    override_existing: bool = False,
    add_tournament_column: bool = True,
    complete_only: bool = False,
) -> SimpleTournamentResults:
    """Combine results from multiple tournament runs into a single unified result set.

    Useful for merging tournaments that were run in parallel or sequentially,
    allowing comprehensive analysis across all runs. Can optionally copy logs
    and other artifacts to a destination directory.

    This function supports combining tournaments saved with different storage formats
    (csv, gzip, parquet) and different optimization levels.

    Args:
        srcs: Source path(s) containing tournament results. Can be a single Path
            or an iterable of Paths to multiple tournament directories.
        dst: Destination path to save combined results. If None, results are not saved.
        recursive: If True, recursively search for tournaments in subdirectories.
        recalc_details: If True, recalculate negotiation details from saved records.
        recalc_scores: If True, recalculate scores from negotiation details.
        must_have_details: If True, only include tournaments with complete detail records.
        verbosity: Logging verbosity level (0=silent, 1=basic, 2+=detailed).
        final_score_stat: Tuple of (measure, statistic) for computing final scores.
            Default is ("advantage", "mean") for mean advantage across negotiations.
        copy: If True, copy all tournament artifacts (logs, plots, etc.) to dst.
        rename_scenarios: If True, rename scenarios to avoid collisions across tournaments.
        rename_short: If True, use shortened names when renaming scenarios.
        add_tournament_folders: If True, preserve tournament folder structure in dst.
        override_existing: If True, overwrite existing files in dst.
        add_tournament_column: If True, add a column identifying source tournament.
        complete_only: If True, only include tournaments that completed successfully.

    Returns:
        SimpleTournamentResults containing the merged data from all source tournaments.

    Notes:
        **Storage Format Handling**:
            - Tournaments with different storage formats (csv, gzip, parquet) can be combined
            - Each tournament's format is auto-detected independently
            - Combined results are saved in CSV format (when dst is provided)

        **File Format Detection Priority** (for each source tournament):
            1. Parquet (.parquet) - Best compression, preserves types
            2. Gzip (.csv.gz) - Good compression
            3. CSV (.csv) - Plain text

        **Combining Tournaments with Different Optimization Levels**:
            - Tournaments saved with storage_optimization="space" (no all_scores file)
              can be combined with tournaments saved with "speed" or "balanced"
            - Scores are reconstructed from details when needed

    Examples:
        Combine two tournament runs::

            results = combine_tournaments(
                srcs=[Path("run1"), Path("run2")], dst=Path("combined"), copy=True
            )
            print(results.final_scores)

        Merge all tournaments in a directory::

            results = combine_tournaments(
                srcs=Path("tournaments"), recursive=True, must_have_details=True
            )
    """
    results, paths = SimpleTournamentResults.combine(
        srcs,
        recursive,
        recalc_details,
        recalc_scores,
        must_have_details,
        verbosity,
        final_score_stat,
        add_tournament_column,
        complete_only=complete_only,
    )
    if results and dst is not None:
        results.save(dst)
    if verbosity:
        print("[green]Done Combining Results[/green]")
    if copy and dst is not None:
        dst = Path(dst).absolute()
        if paths:
            for current in TOURNAMENT_DIRS:
                (dst / current).mkdir(exist_ok=True, parents=True)
        for i, path in (
            enumerate(track(paths, "Copying ... ")) if verbosity else enumerate(paths)
        ):
            path = Path(path).absolute()
            tname = path.name
            prefix = f"{tname}_" if not rename_short else f"N{i}"
            for current in TOURNAMENT_DIRS:
                if not (path / current).exists():
                    if verbosity:
                        print(f"[yellow]{current}[/yellow] not found in {path}")
                        continue
                if add_tournament_folders:
                    this_dst = dst / current / tname
                    this_dst.mkdir(exist_ok=True, parents=True)
                else:
                    this_dst = dst / current
                if verbosity > 1:
                    print(f"Copying {this_dst.relative_to(dst)} from {path} ")
                for x in (path / current).glob("*"):
                    try:
                        if x.is_dir():
                            shutil.copytree(
                                x, this_dst / x.name, dirs_exist_ok=override_existing
                            )
                        else:
                            shutil.copy(x, this_dst / x.name)
                    except Exception as e:
                        if verbosity:
                            print(
                                f"[red]Copy Error:[/red] {x} -> {this_dst} Failed ({e})"
                            )

                if rename_scenarios:
                    files = list(this_dst.glob("*"))
                    for p in files:
                        p = p.absolute()
                        pnew = p.parent / (f"{prefix}{p.name}")
                        try:
                            os.rename(p, pnew)
                        except Exception:
                            if verbosity:
                                print(
                                    f"[red]Rename Failed:[/red]{p.parent}: {p.name} -> {pnew.name}"
                                )
                        # shutil.move(p, p.parent / f"{tname}{p.name}")
                    for df in (results.scores, results.details):
                        for col in df.columns:
                            if isinstance(col, int) or "scenario" not in col:
                                continue
                            df[col] = prefix + df[col].astype(str)
                    results.save(dst)

    return results


def oneinint(x: int | tuple[int, int] | None, log_uniform=None) -> int | None:
    """Returns x or a random sample within its values.

    Args:
        x: The value or 2-valued tuple to sample from
        log_uniform: If true samples using a log-uniform distribution instead of a uniform distribution.
                     If `None`, uses a log-uniform distribution if min > 0 and max/min >= 10

    """
    if isinstance(x, tuple):
        if log_uniform is None:
            log_uniform = x[0] > 0 and x[1] / x[0] >= LOG_UNIFORM_LIMIT
        if x[0] == x[-1]:
            return x[0]
        if log_uniform:
            L = [log(_) for _ in x]
            return min(x[1], max(x[0], int(exp(random() * (L[1] - L[0]) + L[0]))))
        return randint(*x)
    return x


def oneinfloat(x: float | tuple[float, float] | None) -> float | None:
    """Returns x or a random sample within its values"""
    if isinstance(x, tuple):
        if x[0] == x[-1]:
            return x[0]
        return x[0] + random() * (x[1] - x[0])
    return x


def _make_mechanism(
    s: Scenario,
    partners: tuple[type[Negotiator], ...],
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    verbosity: int = 0,
    run_id: int | str | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None, ...] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
) -> tuple[Mechanism, dict, Scenario, str | None]:
    """Create and configure a mechanism with negotiators for a single negotiation.

    Instantiates negotiator objects, adds them to a mechanism, and prepares everything
    for running a negotiation session. Handles name generation, parameter passing,
    and error handling during negotiator initialization.

    Returns:
        Tuple of (mechanism, param_dump, scenario, real_scenario_name) where param_dump
        contains the parameters used to create each negotiator.
    """
    if path:
        path = Path(path)
        for name in (NEGOTIATIONS_DIR_NAME, PLOTS_DIR_NAME, RESULTS_DIR_NAME):
            (path / name).mkdir(exist_ok=True, parents=True)
    s = copy.deepcopy(s)
    assert s.outcome_space is not None
    real_scenario_name = s.outcome_space.name
    if not run_id:
        run_id = unique_name("run", add_time=False, sep=".")
    run_id = str(run_id)
    effective_scenario_name = real_scenario_name
    if mask_scenario_name:
        effective_scenario_name = run_id
        new_os = type(s.outcome_space)(
            issues=s.outcome_space.issues, name=effective_scenario_name
        )
        for u in s.ufuns:
            s.outcome_space = new_os
        s = Scenario(outcome_space=new_os, ufuns=s.ufuns)

    if mechanism_params is None:
        mechanism_params = dict()
    if annotation is None:
        annotation = dict(rep=rep)
    else:
        annotation["rep"] = rep
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    if private_infos is None:
        private_infos = tuple(dict() for _ in partners)  # type: ignore
    assert mechanism_params is not None
    assert all(_ is not None for _ in partner_params)  # type: ignore

    def _name(a: Negotiator) -> str:
        name = a.short_type_name if not full_names else a.type_name
        if name is None:
            name = get_full_type_name(type(a))
        return name

    mechanism_params["name"] = effective_scenario_name
    mechanism_params["verbosity"] = verbosity - 1
    mechanism_params["annotation"] = annotation

    m = mechanism_type(outcome_space=s.outcome_space, **mechanism_params)
    complete_names, negotiators, failures = [], [], dict()
    for type_, p, pinfo in zip(partners, partner_params, private_infos):  # type: ignore
        try:
            negotiator = type_(**p, private_info=copy.deepcopy(pinfo))
            name = _name(negotiator)
            complete_names.append(name)
            negotiators.append(negotiator)
            if p:
                name += hash_to_base64(hash(str(p)))
        except Exception as e:
            if ignore_exceptions:
                failures = dict(
                    erred_negotiator=get_full_type_name(type_),
                    error_details=str(e),
                    has_error=True,
                )
                break
            else:
                raise (e)
    if failures:
        return m, failures, s, real_scenario_name

    if not partner_names:
        partner_names = tuple(complete_names)  # type: ignore
    for L_, (negotiator, name, u) in enumerate(
        zip(negotiators, partner_names, s.ufuns)
    ):
        # Use same value for both id and name to ensure consistency in traces
        # (trace uses id in 'negotiator' field but names for utility columns)
        if id_reveals_type or name_reveals_type:
            # If either reveals type, use the type-revealing format for both
            negotiator.id = f"{name}@{L_}"
            negotiator.name = f"{name}@{L_}"
        else:
            # Generate a single random name and use for both id and name
            random_name = unique_name("n", add_time=False, sep="")
            negotiator.id = random_name
            negotiator.name = random_name
        complete_names.append(name)
        m.add(negotiator, ufun=copy.deepcopy(u))

    return (m, failures, s, real_scenario_name)


def _make_failure_record(
    s: Scenario,
    state: MechanismState,
    partners: tuple[type[Negotiator], ...],
    run_id: str,
    execution_time: float,
    mechanism_params: dict,
    partner_names: tuple[str, ...] | None = None,
    param_dump: tuple[dict[str, Any], ...] | None = None,
    real_scenario_name: str | None = None,
    mechanism_type: type[Mechanism] | None = None,
    stats: ScenarioStats | None = None,
) -> dict:
    """Create a record for a failed negotiation with null/error values."""
    if not partner_names:
        partner_names = tuple(get_full_type_name(_) for _ in partners)
    if param_dump is not None and all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    min_max_utils = [_.minmax() for _ in s.ufuns]
    min_utils = [_[0] for _ in min_max_utils]
    max_utils = [_[1] for _ in min_max_utils]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["min_utils"] = min_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = partner_names
    run_record["negotiator_ids"] = partner_names
    run_record["negotiator_types"] = partner_names
    run_record["negotiator_times"] = [float("nan") for _ in partners]
    run_record["n_steps"] = mechanism_params.get("n_steps", float("inf"))
    run_record["time_limit"] = mechanism_params.get("time_limit", float("inf"))
    run_record["pend"] = mechanism_params.get("pend", float("inf"))
    run_record["pend_per_second"] = mechanism_params.get(
        "pend_per_second", float("inf")
    )
    run_record["step_time_limit"] = mechanism_params.get(
        "step_time_limit", float("inf")
    )
    run_record["negotiator_time_limit"] = mechanism_params.get(
        "negotiator_time_limit", float("inf")
    )
    run_record["annotation"] = mechanism_params.get("annotation", dict())
    run_record["scenario"] = real_scenario_name
    run_record["mechanism_name"] = "Unknown"
    run_record["mechanism_type"] = get_full_type_name(mechanism_type)
    run_record["effective_scenario_name"] = s.outcome_space.name
    run_record["running"] = state.running
    run_record["waiting"] = state.waiting
    run_record["started"] = state.started
    run_record["last_step"] = state.step
    run_record["last_time"] = state.time
    run_record["relative_time"] = state.relative_time
    run_record["broken"] = state.broken
    run_record["timedout"] = state.timedout
    run_record["agreement"] = state.agreement
    run_record[RESULTS_DIR_NAME] = state.results
    run_record["n_negotiators"] = state.n_negotiators
    run_record["has_error"] = state.has_error
    run_record["erred_negotiator"] = state.erred_negotiator
    run_record["error_details"] = state.error_details

    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )
        # Calculate fairness as max of nash, kalai, and ks optimality
        fairness_values = []
        for key in ("nash_optimality", "kalai_optimality", "ks_optimality"):
            if key in run_record and not isnan(run_record[key]):
                fairness_values.append(run_record[key])
        run_record["fairness"] = (
            max(fairness_values) if fairness_values else float("nan")
        )

    return run_record


def _calc_opponent_model_scores(
    negotiators: list[Negotiator],
    actual_ufuns: list["BaseUtilityFunction"],
    metrics: tuple[COMPARE_UFUN_METHOD_TYPE, ...],
    outcome_space: OutcomeSpace | None = None,
) -> dict[str, list[float]]:
    """Calculate opponent modeling scores for each negotiator.

    For each negotiator, compare their estimated opponent utility function(s) with the
    actual opponent utility function(s) using specified comparison metrics.

    Args:
        negotiators: List of negotiators from the completed mechanism
        actual_ufuns: List of actual utility functions (from scenario), ordered by negotiator index
        metrics: Tuple of comparison methods to use (e.g., 'kendall', 'kendall_optimality', 'ndcg')
        outcome_space: The outcome space for comparison (if None, inferred from ufuns)

    Returns:
        Dictionary mapping metric names to lists of scores (one score per negotiator).
        For bilateral negotiations, compares negotiator.opponent_ufun with actual opponent ufun.
        For multilateral, averages scores across all opponent_ufuns in private_info["opponent_ufuns"].

    Notes:
        - If a negotiator doesn't have an opponent model, returns NaN for that negotiator
        - Metric names are prefixed with "opponent_" (e.g., "opponent_kendall_optimality")
    """

    n = len(negotiators)
    bilateral = n == 2

    # Initialize result dict with empty lists for each metric
    result: dict[str, list[float]] = {}
    for metric in metrics:
        metric_name = metric if isinstance(metric, str) else metric.__name__
        result[f"opp_{metric_name}"] = []

    for i, negotiator in enumerate(negotiators):
        # Get opponent indices (all others)
        opponent_indices = [j for j in range(n) if j != i]

        for metric in metrics:
            min_value = 0.0 if metric != "kendall" else -1.0
            metric_name = metric if isinstance(metric, str) else metric.__name__
            key = f"opp_{metric_name}"

            if bilateral:
                # For bilateral: use negotiator.opponent_ufun
                estimated_ufun = negotiator.opponent_ufun
                actual_ufun = actual_ufuns[opponent_indices[0]]

                try:
                    score = compare_ufuns(
                        estimated_ufun,
                        actual_ufun,
                        method=metric,
                        outcome_space=outcome_space,
                    )
                    result[key].append(score)
                except Exception:
                    # Opponent model may not be a fully functional ufun
                    result[key].append(min_value)
            else:
                # For multilateral: average across all opponent_ufuns
                opponent_ufuns_dict = negotiator.private_info.get("opponent_ufuns", {})

                if not opponent_ufuns_dict:
                    # No opponent models - check single opponent_ufun as fallback
                    estimated_ufun = negotiator.opponent_ufun
                    # Use single opponent_ufun, average over all opponents
                    scores = []
                    for opp_idx in opponent_indices:
                        try:
                            s = compare_ufuns(
                                estimated_ufun,
                                actual_ufuns[opp_idx],
                                method=metric,
                                outcome_space=outcome_space,
                            )
                            scores.append(s)
                        except Exception:
                            # Opponent model may not be a fully functional ufun
                            scores.append(min_value)
                        result[key].append(
                            sum(scores) / len(scores) if scores else min_value
                        )
                else:
                    # Have opponent_ufuns dict - use specific models for each opponent
                    scores = []
                    for opp_idx in opponent_indices:
                        # Try to find opponent model by negotiator id or index
                        opp_negotiator = negotiators[opp_idx]
                        opp_id = opp_negotiator.id
                        estimated_ufun = opponent_ufuns_dict.get(
                            opp_id, opponent_ufuns_dict.get(str(opp_idx))
                        )
                        if estimated_ufun is not None:
                            try:
                                s = compare_ufuns(
                                    estimated_ufun,
                                    actual_ufuns[opp_idx],
                                    method=metric,
                                    outcome_space=outcome_space,
                                )
                                scores.append(s)
                            except Exception:
                                # Opponent model may not be a fully functional ufun
                                pass
                    result[key].append(
                        sum(scores) / len(scores) if scores else min_value
                    )

    return result


def _make_record(
    m: Mechanism,
    s: Scenario,
    param_dump,
    partner_names,
    run_id,
    execution_time,
    real_scenario_name,
    stats,
    scored_indices: list[int] | None = None,
    opponent_modeling_metrics: tuple[COMPARE_UFUN_METHOD_TYPE, ...] = (),
    distribute_opponent_modeling_scores: bool = True,
) -> dict[str, Any]:
    """Create a detailed record dictionary from a completed negotiation.

    Extracts information from the mechanism state and scenario, calculates utilities
    and optimality measures, and bundles everything into a comprehensive record dict.
    """
    state = m.state
    if all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    min_max_utils = [_.minmax() for _ in s.ufuns]
    min_utils = [_[0] for _ in min_max_utils]
    max_utils = [_[1] for _ in min_max_utils]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["min_utils"] = min_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = m.negotiator_names
    run_record["negotiator_ids"] = m.negotiator_ids
    run_record["negotiator_types"] = [_.type_name for _ in m.negotiators]
    run_record["negotiator_times"] = [m.negotiator_times[_] for _ in m.negotiator_ids]
    run_record["n_steps"] = m._internal_nmi.n_steps
    run_record["time_limit"] = m._internal_nmi.time_limit
    run_record["shared_n_steps"] = m._shared_nmi.n_steps
    run_record["shared_time_limit"] = m._shared_nmi.time_limit
    run_record["pend"] = m._internal_nmi.pend
    run_record["pend_per_second"] = m._internal_nmi.pend_per_second
    run_record["step_time_limit"] = m._internal_nmi.step_time_limit
    run_record["negotiator_time_limit"] = m._internal_nmi.negotiator_time_limit
    run_record["annotation"] = m._internal_nmi.annotation
    run_record["scenario"] = real_scenario_name
    run_record["mechanism_name"] = m.name
    run_record["mechanism_type"] = m.type_name
    run_record["effective_scenario_name"] = s.outcome_space.name
    run_record["running"] = state.running
    run_record["waiting"] = state.waiting
    run_record["started"] = state.started
    run_record["last_step"] = state.step
    run_record["last_time"] = state.time
    run_record["relative_time"] = state.relative_time
    run_record["broken"] = state.broken
    run_record["timedout"] = state.timedout
    run_record["agreement"] = state.agreement
    run_record[RESULTS_DIR_NAME] = state.results
    run_record["n_negotiators"] = state.n_negotiators
    run_record["has_error"] = state.has_error
    run_record["erred_negotiator"] = state.erred_negotiator
    run_record["error_details"] = state.error_details
    run_record["scored_indices"] = scored_indices

    if m._internal_nmi.annotation:
        run_record.update(m._internal_nmi.annotation)
    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )
        # Calculate fairness as max of nash, kalai, and ks optimality
        fairness_values = []
        for key in ("nash_optimality", "kalai_optimality", "ks_optimality"):
            if key in run_record and not isnan(run_record[key]):
                fairness_values.append(run_record[key])
        run_record["fairness"] = (
            max(fairness_values) if fairness_values else float("nan")
        )

    # Calculate opponent modeling scores if metrics are specified
    if opponent_modeling_metrics:
        # Handle anl2026 special metric: it adds kendall_optimality internally
        # but is calculated specially as advantage + normalized kendall_optimality
        has_anl2026 = "anl2026" in opponent_modeling_metrics
        # Filter out anl2026 from metrics passed to _calc_opponent_model_scores
        # (it's not a valid compare_ufuns method)
        actual_metrics = tuple(m for m in opponent_modeling_metrics if m != "anl2026")
        # If anl2026 is requested, ensure kendall_optimality is included
        if has_anl2026 and "kendall_optimality" not in actual_metrics:
            actual_metrics = actual_metrics + ("kendall_optimality",)

        opp_scores = _calc_opponent_model_scores(
            negotiators=list(m.negotiators),
            actual_ufuns=list(s.ufuns),
            metrics=actual_metrics,
            outcome_space=s.outcome_space,
        )

        # Apply distribution (normalization) if requested
        if distribute_opponent_modeling_scores:
            for key, values in opp_scores.items():
                if isinstance(values, list) and len(values) > 0:
                    total = sum(v for v in values if not isnan(v))
                    if total > 0:
                        opp_scores[key] = [
                            v / total if not isnan(v) else float("nan") for v in values
                        ]
                    else:
                        # Total is zero or negative: set everyone to 0
                        # (0 means "no opponent model" for all metrics)
                        opp_scores[key] = [0.0 for _ in values]

        # Calculate anl2026 metric if requested
        # anl2026 = advantage + normalized_kendall_optimality
        # Note: kendall_optimality is ALWAYS distributed for anl2026, regardless of
        # the distribute_opponent_modeling_scores setting
        if has_anl2026:
            kendall_key = "opp_kendall_optimality"
            if kendall_key in opp_scores:
                # Get raw kendall values and normalize them for anl2026
                # (this is separate from the distribute_opponent_modeling_scores normalization)
                kendall_values = opp_scores[kendall_key]
                total = sum(v for v in kendall_values if not isnan(v))
                if total > 0:
                    normalized_kendall = [
                        v / total if not isnan(v) else float("nan")
                        for v in kendall_values
                    ]
                else:
                    # Total is zero or negative: set everyone to 0
                    # (0 means "no opponent model")
                    normalized_kendall = [0.0] * len(kendall_values)

                n_negotiators = len(kendall_values)

                # Calculate advantage for each negotiator
                advantages = []
                for i in range(n_negotiators):
                    u = agreement_utils[i]
                    r = reservations[i]
                    mx = max_utils[i]
                    mn = min_utils[i]
                    if (isinf(r) and r < 0) or isnan(r):
                        adv = u - mn if not isnan(mn) else 0.0
                    elif mx != r:
                        adv = (u - r) / (mx - r)
                    else:
                        adv = 0.0
                    advantages.append(adv)

                # anl2026 = advantage + normalized_kendall_optimality
                opp_scores["opp_anl2026"] = [
                    adv + k if not isnan(k) else adv
                    for adv, k in zip(advantages, normalized_kendall)
                ]

        run_record.update(opp_scores)

    return run_record


def _save_record(
    run_record,
    m: Mechanism,
    partner_names,
    real_scenario_name,
    rep,
    run_id,
    path,
    storage_optimization: OptimizationLevel = "space",
    storage_format: StorageFormat = DEFAULT_TABLE_STORAGE_FORMAT,
    scenario: Scenario | None = None,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
    single_file: bool = True,
    stats: ScenarioStats | None = None,
):
    """Save negotiation record to disk using mechanism.save() with source priority.

    Args:
        run_record: The run record dictionary to save
        m: The mechanism instance
        partner_names: Names of the negotiating partners
        real_scenario_name: Name of the scenario
        rep: Repetition number
        run_id: Unique run identifier
        path: Path to save to (or None to skip saving)
        storage_optimization: Optimization level ("speed", "balanced", "space")
        storage_format: Storage format ("csv", "gzip", "parquet")
        scenario: The scenario object (to include in metadata)
        python_class_identifier: Identifier for Python classes in serialization
        single_file: If True, save as single file; if False, save as folder with agreement stats
        stats: Scenario statistics for calculating agreement optimality (used when single_file=False)

    Remarks:
        - Tries sources in priority order: full_trace_with_utils, full_trace,
          extended_trace, trace, then other available sources, with history last
        - Does not save scenario or scenario stats (scenario already saved
          separately in scenarios/ directory)
        - Saves config only if not optimizing for space
        - Includes scenario name in metadata
        - Warns if file exists
        - Storage format: parquet for speed/space, gzip for balanced, csv otherwise
        - When single_file=False, agreement_stats are calculated and saved if stats is provided
    """
    if not path:
        return

    file_name = _get_run_file_name(real_scenario_name, partner_names, rep, run_id)

    # Determine source priority: full_trace_with_utils > full_trace > extended_trace > trace > others > history
    available_sources = m.available_save_sources()
    priority_sources = [
        "full_trace_with_utils",
        "full_trace",
        "extended_trace",
        "trace",
    ]

    # Build ordered list of sources to try
    sources_to_try = []
    for src in priority_sources:
        if src in available_sources:
            sources_to_try.append(src)

    # Add other sources except history
    for src in available_sources:
        if src not in sources_to_try and src != "history":
            sources_to_try.append(src)

    # Add history last
    if "history" in available_sources:
        sources_to_try.append("history")

    # Use the first available source
    source = sources_to_try[0] if sources_to_try else "history"

    # Prepare metadata with scenario information
    metadata = {
        "scenario_name": real_scenario_name,
        "repetition": rep,
        "run_id": str(run_id),
        "partner_names": partner_names,
    }

    # Add scenario object to metadata if provided
    if scenario:
        metadata["scenario"] = {
            "name": (
                scenario.outcome_space.name
                if scenario.outcome_space
                else real_scenario_name
            ),
            "n_outcomes": (
                scenario.outcome_space.cardinality if scenario.outcome_space else None
            ),
            "n_negotiators": len(scenario.ufuns) if scenario.ufuns else None,
        }

    # Save using mechanism.save()
    save_dir = path / NEGOTIATIONS_DIR_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract agreement_stats from run_record if available (for folder mode)
    agreement_stats = None
    if not single_file:
        # Try to construct OutcomeOptimality from run_record values
        try:
            if "pareto_optimality" in run_record:
                agreement_stats = OutcomeOptimality(
                    pareto_optimality=run_record.get("pareto_optimality", float("nan")),
                    nash_optimality=run_record.get("nash_optimality", float("nan")),
                    kalai_optimality=run_record.get("kalai_optimality", float("nan")),
                    modified_kalai_optimality=run_record.get(
                        "modified_kalai_optimality", float("nan")
                    ),
                    max_welfare_optimality=run_record.get(
                        "max_welfare_optimality", float("nan")
                    ),
                    ks_optimality=run_record.get("ks_optimality", float("nan")),
                    modified_ks_optimality=run_record.get(
                        "modified_ks_optimality", float("nan")
                    ),
                )
        except Exception:
            agreement_stats = None

    try:
        m.save(
            parent=save_dir,
            name=file_name,
            single_file=single_file,
            save_scenario=False,  # Don't save scenario (already saved in scenarios/)
            save_scenario_stats=False,  # Don't save scenario stats
            save_config=storage_optimization
            != "space",  # Save config unless optimizing for space
            source=source,
            metadata=metadata,
            agreement_stats=agreement_stats,
            overwrite=True,
            warn_if_existing=True,
            storage_format=storage_format,
        )
    except Exception as e:
        # Fallback to saving just the run record if mechanism.save() fails
        print(
            f"[yellow]Warning: Could not save negotiation using mechanism.save() "
            f"({e}). Falling back to saving run_record only.[/yellow]"
        )
        full_name = path / RESULTS_DIR_NAME / f"{file_name}.json"
        full_name.parent.mkdir(parents=True, exist_ok=True)
        if full_name.exists():
            print(f"[yellow]{full_name} already found[/yellow]")
            full_name = (
                path / RESULTS_DIR_NAME / unique_name(f"{file_name}.json", sep="")
            )
        dump(run_record, full_name)
        return

    # Always save the run_record separately as JSON for easy access
    full_name = path / RESULTS_DIR_NAME / f"{file_name}.json"
    full_name.parent.mkdir(parents=True, exist_ok=True)
    if full_name.exists():
        print(f"[yellow]{full_name} already found[/yellow]")
        full_name = path / RESULTS_DIR_NAME / unique_name(f"{file_name}.json", sep="")
    dump(run_record, full_name)


def _plot_run(
    m,
    partner_names,
    real_scenario_name,
    rep,
    run_id,
    path,
    plot,
    plot_params,
    image_format=DEFAULT_IMAGE_FORMAT,
):
    """Save a plot of the negotiation mechanism to disk.

    Args:
        image_format: Format for saving the figure. Used only if plot_params doesn't specify
                     a fig_name with an extension. If fig_name in plot_params has an extension,
                     that extension is respected.
    """
    file_name = _get_run_file_name(real_scenario_name, partner_names, rep, run_id)
    if not path or not plot:
        return
    if plot_params is None:
        plot_params = dict()
    plot_params["save_fig"] = (True,)

    # Check if user provided a fig_name with an extension
    fig_name = plot_params.get("fig_name")
    if fig_name and Path(str(fig_name)).suffix:
        # User provided a filename with extension, respect it
        full_name = path / PLOTS_DIR_NAME / fig_name
    else:
        # Auto-generate filename with image_format
        full_name = path / PLOTS_DIR_NAME / f"{file_name}.{image_format}"

    m.plot(path=path, fig_name=full_name, show=False, **plot_params)


def run_negotiation(
    s: Scenario,
    partners: tuple[type[Negotiator], ...],
    run_id: int | str,
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    verbosity: int = 0,
    plot=False,
    plot_params: dict[str, Any] | None = None,
    stats: ScenarioStats | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
    scored_indices: list[int] | None = None,
    n_repetitions: int = 1,
    scenario_index: int = 0,
    before_start_callback: BeforeStartCallback | None = None,
    after_construction_callback: AfterConstructionCallback | None = None,
    after_end_callback: AfterEndCallback | None = None,
    neg_start_callback: NegStartCallback | None = None,
    neg_end_callback: NegEndCallback | None = None,
    neg_progress_callback: NegProgressCallback | None = None,
    config: dict[str, Any] | None = None,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    storage_optimization: OptimizationLevel = "space",
    storage_format: StorageFormat | None = None,
    save_negotiations_as_folders: bool = False,
    opponent_modeling_metrics: tuple[COMPARE_UFUN_METHOD_TYPE, ...] = (),
    distribute_opponent_modeling_scores: bool = True,
) -> dict[str, Any]:
    """Run a single negotiation session and return comprehensive results.

    Creates negotiator instances, runs them through a negotiation mechanism, and returns
    detailed results including agreement, utilities, timing, and error information.

    Args:
        s: Scenario containing the outcome space and utility functions for all parties.
        partners: Negotiator types/classes to instantiate for this negotiation, in order.
        partner_names: Display names for negotiators. If None, generated from class names.
        partner_params: Initialization parameters for each negotiator. If None, use defaults.
        rep: Repetition number for this negotiation (for tracking in tournament context).
        path: Directory to save logs and plots. If None, nothing is saved to disk.
        mechanism_type: Negotiation protocol/mechanism class (default: SAOMechanism).
        mechanism_params: Parameters passed to mechanism constructor.
        full_names: If True and partner_names is None, use full class names instead of shortened.
        verbosity: Logging verbosity level (0 for silent).
        plot: If True and path is provided, save a plot of the negotiation.
        plot_params: Parameters passed to plotting function.
        run_id: Unique identifier for this run. If None, generated from timestamp.
        stats: Pre-calculated scenario statistics. If None, calculated if needed.
        annotation: Dictionary stored in mechanism.nmi.annotation (accessible to negotiators via self.nmi.annotation).
        private_infos: Tuple of private info dicts, one per negotiator (accessible via self.private_info).
        id_reveals_type: If True, negotiator IDs reveal their type.
        name_reveals_type: If True, negotiator names reveal their type.
        mask_scenario_name: If True, scenario name is masked from negotiators.
        ignore_exceptions: If True, catch and log exceptions instead of propagating.
        scored_indices: Positions of negotiators to score. None means score all (used internally by tournament).
        n_repetitions: Total number of repetitions for this scenario/partner combo (for RunInfo).
        scenario_index: Index of this scenario in the tournament (for RunInfo).
        before_start_callback: Optional callback invoked before negotiation starts. Receives RunInfo object.
        after_construction_callback: Optional callback invoked after mechanism construction. Receives ConstructedNegInfo object.
        after_end_callback: Optional callback invoked after negotiation ends. Receives the record dictionary
            and optionally the config dictionary. Supports both (record) and (record, config) signatures.
        config: Tournament configuration dictionary (same as saved to config.yaml). Passed to callbacks.
        save_negotiations_as_folders: If True, save each negotiation as a folder containing trace,
            agreement_stats, config, and metadata. If False (default), save as single trace files.

    Returns:
        Dictionary containing complete negotiation results:
        - agreement: Final agreed outcome or None
        - utilities: Utility of agreement for each negotiator
        - reserved_values: Reservation values for each negotiator
        - max_utils: Maximum possible utility for each negotiator
        - partners: Negotiator class names
        - negotiator_ids: Unique IDs of negotiator instances
        - negotiator_times: Time spent by each negotiator
        - scenario: Scenario name
        - timedout/broken/has_error: Status flags
        - step/time/relative_time: Negotiation progress metrics
        - Plus optimality statistics if stats is provided

    Examples:
        Basic usage:
        ```python
        record = run_negotiation(
            s=scenario,
            partners=(AspirationNegotiator, RandomNegotiator),
            mechanism_params=dict(n_steps=100),
        )
        print(f"Agreement: {record['agreement']}")
        print(f"Utilities: {record['utilities']}")
        ```
    """
    if before_start_callback:
        try:
            before_start_callback(
                RunInfo(
                    s=s,
                    partners=partners,
                    partner_names=partner_names,
                    partner_params=partner_params,
                    rep=rep,
                    path=path,
                    mechanism_type=mechanism_type,
                    mechanism_params=mechanism_params,
                    full_names=full_names,
                    verbosity=verbosity,
                    plot=plot,
                    plot_params=plot_params,
                    run_id=run_id,
                    stats=stats,
                    annotation=annotation,
                    private_infos=private_infos,
                    id_reveals_type=id_reveals_type,
                    name_reveals_type=name_reveals_type,
                    mask_scenario_name=mask_scenario_name,
                    ignore_exceptions=ignore_exceptions,
                    scored_indices=scored_indices,
                    n_repetitions=n_repetitions,
                    scenario_index=scenario_index,
                    config=config or {},
                )
            )
        except Exception as e:
            if verbosity > 0:
                print(f"Before start callback failed for {run_id} with exception: {e}")
    m, failures, s, real_scenario_name = _make_mechanism(
        s=s,
        partners=partners,
        partner_names=partner_names,
        partner_params=partner_params,
        rep=rep,
        path=path,
        mechanism_type=mechanism_type,
        mechanism_params=mechanism_params,
        full_names=full_names,
        run_id=run_id,
        annotation=annotation,
        private_infos=private_infos,
        id_reveals_type=id_reveals_type,
        name_reveals_type=name_reveals_type,
        mask_scenario_name=mask_scenario_name,
        ignore_exceptions=ignore_exceptions,
    )

    # Extract partner names from mechanism if not provided
    if partner_names is None:
        partner_names = tuple(
            n.short_type_name if not full_names else n.type_name for n in m.negotiators
        )

    if after_construction_callback:
        try:
            after_construction_callback(
                ConstructedNegInfo(
                    run_id, m, failures, s, real_scenario_name, config=config or {}
                )
            )
        except Exception as e:
            if verbosity > 0:
                print(
                    f"After construction callback failed for {run_id} with exception: {e}"
                )
    reservations = tuple(u.reserved_value for u in s.ufuns)
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    param_dump = tuple(str(to_flat_dict(_)) if _ else None for _ in partner_params)  # type: ignore
    if failures:
        agreement_utils = reservations
        execution_time = 0.0
        state = SAOState(
            has_error=True,
            error_details=failures["error_details"],
            erred_negotiator=failures["erred_negotiator"],
        )
    else:
        if verbosity > 0:
            print(
                f"{datetime.datetime.now()} {partner_names} on {real_scenario_name} (rep: {rep}): [magenta]started[/magenta]",
                flush=True,
            )
        strt = perf_counter()
        try:
            state = m.run(
                start_callback=_PicklableCallback(neg_start_callback, run_id)
                if neg_start_callback is not None
                else None,
                progress_callback=_PicklableCallback(neg_progress_callback, run_id)
                if neg_progress_callback is not None
                else None,
                completion_callback=_PicklableCallback(neg_end_callback, run_id)
                if neg_end_callback is not None
                else None,
            )
        except Exception as e:
            if not ignore_exceptions:
                raise e
            else:
                state = m.state
                state.has_error = True
                state.error_details = str(e)
        execution_time = perf_counter() - strt
        if verbosity > 0:
            agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
            advs = tuple(round(a - b, 3) for a, b in zip(agreement_utils, reservations))
            print(
                f"{datetime.datetime.now()} {partner_names} on {real_scenario_name} (rep: {rep}): {state.agreement} in "
                f"{state.relative_time:4.2%} of allowed steps/time with advantages: "
                f"{advs} "
                f"[green]done[/green] in {humanize_time(execution_time)}",
                flush=True,
            )

    run_record = _make_record(
        m=m,
        s=s,
        param_dump=param_dump,
        partner_names=partner_names,
        run_id=run_id,
        execution_time=execution_time,
        real_scenario_name=real_scenario_name,
        stats=stats,
        scored_indices=scored_indices,
        opponent_modeling_metrics=opponent_modeling_metrics,
        distribute_opponent_modeling_scores=distribute_opponent_modeling_scores,
    )
    if after_end_callback:
        try:
            _call_after_end_callback(after_end_callback, run_record, config)
        except Exception as e:
            if verbosity > 0:
                print(f"After end callback failed for {run_id}: {e}")

    # Determine effective storage format
    effective_storage_format = (
        storage_format if storage_format is not None else DEFAULT_TABLE_STORAGE_FORMAT
    )

    _save_record(
        run_record,
        m,
        partner_names,
        real_scenario_name,
        rep,
        run_id,
        path,
        storage_optimization=storage_optimization,
        storage_format=effective_storage_format,
        scenario=s,
        single_file=not save_negotiations_as_folders,
    )
    _plot_run(
        m,
        partner_names,
        real_scenario_name,
        rep,
        run_id,
        path,
        plot,
        plot_params,
        image_format,
    )
    return run_record


def failed_run_record(
    s: Scenario,
    partners: tuple[type[Negotiator], ...],
    timeout: float,
    run_id: str,
    partner_names: tuple[str, ...] = tuple(),
    partner_params: tuple[dict[str, Any], ...] | None = None,
    error: str | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None, ...] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
    stats: ScenarioStats | None = None,
    scored_indices: list[int] | None = None,
    storage_optimization: OptimizationLevel = "space",
    storage_format: StorageFormat | None = None,
):
    """Create a record for a negotiation that failed to complete (timeout or exception).

    Attempts to create a mechanism and extract as much state as possible, then generates
    a record with error flags set and null/zero values for results. Used when run_negotiation
    times out or encounters an unrecoverable error.

    Returns:
        Dictionary with same structure as successful negotiation records but with has_error=True
        and null/zero values for agreement, utilities, etc.
    """
    if partner_params is None:
        partner_params = tuple(dict() for _ in partners)  # type: ignore
    param_dump = tuple(str(to_flat_dict(_)) if _ else None for _ in partner_params)  # type: ignore
    execution_time = timeout
    try:
        m, _, s, real_scenario_name = _make_mechanism(
            s=s,
            partners=partners,
            partner_names=partner_names,
            partner_params=partner_params,
            rep=rep,
            path=path,
            mechanism_type=mechanism_type,
            mechanism_params=mechanism_params,
            full_names=full_names,
            run_id=run_id,
            annotation=annotation,
            private_infos=private_infos,
            id_reveals_type=id_reveals_type,
            name_reveals_type=name_reveals_type,
            mask_scenario_name=mask_scenario_name,
            ignore_exceptions=ignore_exceptions,
        )
        state = m.state
        state.has_error = True
        state.timedout = True
        state.started = True
        state.error_details = f"Timedout after {timeout} with error {error}"

        run_record = _make_record(
            m=m,
            s=s,
            param_dump=param_dump,
            partner_names=partner_names,
            run_id=run_id,
            execution_time=execution_time,
            real_scenario_name=real_scenario_name,
            stats=stats,
            scored_indices=scored_indices,
        )
    except Exception as e:
        real_scenario_name = s.outcome_space.name
        m = SAOMechanism()
        state = SAOState()
        state.has_error = True
        state.timedout = True
        state.started = True
        state.error_details = (
            f"Timedout after {timeout} with exception {error} then Raised {e}"
        )
        run_record = _make_failure_record(
            state=state,
            s=s,
            param_dump=param_dump,
            partner_names=partner_names,
            run_id=run_id,
            execution_time=execution_time,
            real_scenario_name=real_scenario_name,
            stats=stats,
            mechanism_type=mechanism_type,
            mechanism_params=mechanism_params if mechanism_params else dict(),
            partners=partners,
        )

    # Determine effective storage format
    effective_storage_format = (
        storage_format if storage_format is not None else DEFAULT_TABLE_STORAGE_FORMAT
    )

    _save_record(
        run_record,
        m,
        partner_names,
        real_scenario_name,
        rep,
        run_id,
        path,
        storage_optimization=storage_optimization,
        storage_format=effective_storage_format,
        scenario=s,
    )
    return run_record


# def _stop_process_pool(executor):
#     try:
#         if executor and executor._processes:
#             for _, process in executor._processes.items():
#                 process.terminate()
#         if executor:
#             executor.shutdown(wait=False)
#             executor.cancel_pending_futures()
#             executor.shutdown(wait=False)
#     except Exception:
#         pass


def make_scores(
    record: dict[str, Any],
    scored_indices: list[int] | None = None,
    raw_aggregated_metrics: dict[str, Callable[[dict[str, float]], float]]
    | None = None,
) -> list[dict[str, float]]:
    """Convert a negotiation run record into score dictionaries for each negotiator.

    Extracts utilities, reserved values, and other metrics from the negotiation record
    and creates a score entry for each negotiator (or only for negotiators at specified indices).

    Args:
        record: Dictionary containing complete negotiation results including utilities, agreement,
               partners, times, errors, and optimality statistics.
        scored_indices: If provided, only create scores for negotiators at these position indices.
                       If None, score all negotiators. Used in explicit opponent mode to score
                       only competitors (not opponents).
        raw_aggregated_metrics: Optional dict mapping new metric names to callables that receive
                              all metrics for a negotiator and return a combined score.

    Returns:
        List of dictionaries, one per scored negotiator, each containing:
        - strategy: Negotiator class name
        - utility: Utility achieved from the agreement
        - reserved_value: Negotiator's reservation value
        - advantage: Normalized utility gain (utility - reserved) / (max - reserved), 0.0 if max == reserved
        - partner_welfare: Average utility of other negotiators
        - welfare: Average utility of all negotiators
        - scenario: Scenario name
        - partners: Names of other negotiators
        - time: Time taken by this negotiator
        - negotiator_id: Unique ID of the negotiator instance
        - has_error: Whether any error occurred
        - self_error: Whether this negotiator caused an error
        - mechanism_error: Whether the mechanism caused an error
        - Plus any optional columns and optimality metrics (if save_stats was True):
            - nash_optimality: Closeness to Nash bargaining solution
            - kalai_optimality: Closeness to Kalai-Smorodinsky solution
            - ks_optimality: Closeness to KS solution
            - pareto_optimality: Closeness to Pareto frontier
            - max_welfare_optimality: Closeness to maximum welfare point
            - fairness: Maximum of nash_optimality, kalai_optimality, and ks_optimality
            - modified_kalai_optimality: Closeness to modified Kalai solution
            - modified_ks_optimality: Closeness to modified KS solution
        - Plus opponent modeling metrics (if opponent_modeling_metrics was provided):
            - opponent_<metric>: Opponent modeling score for each specified metric
        - Plus raw aggregated scores (if raw_aggregated_metrics was provided):
            - Custom combined metrics computed from other metrics
    """
    utils, partners = record["utilities"], record["partners"]
    reserved_values = record["reserved_values"]
    negids = record["negotiator_ids"]
    max_utils = record["max_utils"]
    min_utils = record.get("min_utils", [float("nan")] * len(utils))
    times = record.get("negotiator_times", [None] * len(utils))
    has_error = record["has_error"]
    erred_negotiator = record["erred_negotiator"]
    error_details = record["error_details"]
    mech_error = has_error and not erred_negotiator
    scores = []
    for i, (u, r, a, m, mn, t, nid) in enumerate(
        zip(utils, reserved_values, partners, max_utils, min_utils, times, negids)
    ):
        # Only score negotiators at specified indices
        if scored_indices is not None and i not in scored_indices:
            continue
        n_p = len(partners)
        bilateral = n_p == 2

        # Calculate advantage: handle -inf and NaN reserved values
        # When r is -inf or NaN, use u - min(utility) instead of (u - r) / (m - r)
        if (isinf(r) and r < 0) or isnan(r):
            # When reserved value is -inf or NaN, advantage is just u - minimum utility
            advantage = u - mn if not isnan(mn) else 0.0
        elif m != r:
            advantage = (u - r) / (m - r)
        else:
            advantage = 0.0

        basic = dict(
            strategy=a,
            utility=u,
            reserved_value=r,
            advantage=advantage,
            partner_welfare=sum(_ for j, _ in enumerate(utils) if j != i) / (n_p - 1),
            welfare=sum(_ for _ in utils) / n_p,
            scenario=record["scenario"],
            partners=(partners[_] for _ in range(len(partners)) if _ != i)
            if not bilateral
            else partners[1 - i],
            time=t,
            negotiator_id=nid,
            has_error=has_error,
            self_error=has_error and not mech_error and (erred_negotiator == nid),
            mechanism_error=mech_error,
            error_details=error_details,
            mechanism_name=record.get("mechanism_name", ""),
        )
        for col in OPTIONAL_COLS:
            if col in record:
                basic[col] = record[col]
        for c in OPTIMALITY_COLS:
            if c in record:
                basic[c] = record[c]

        # Extract per-negotiator opponent model scores from record
        # These are stored as lists with one value per negotiator
        for key, value in record.items():
            if key.startswith("opp_") and isinstance(value, (list, tuple)):
                if i < len(value):
                    basic[key] = value[i]

        # Apply raw_aggregated_metrics to compute custom combined metrics
        if raw_aggregated_metrics:
            # Build a dict of numeric metrics for this negotiator
            numeric_metrics = {
                k: v
                for k, v in basic.items()
                if isinstance(v, (int, float))
                and (not isnan(v) if isinstance(v, float) else True)
            }
            for metric_name, aggregator in raw_aggregated_metrics.items():
                try:
                    basic[metric_name] = aggregator(numeric_metrics)
                except Exception:
                    basic[metric_name] = float("nan")

        scores.append(basic)
    return scores


def _get_run_file_name(
    scenario_name: str,
    partner_names: tuple[str, ...] | list[str],
    rep: int,
    run_id: int | str,
) -> str:
    """Generate the file name for a run result.

    Args:
        scenario_name: Name of the scenario.
        partner_names: Names of the negotiators.
        rep: Repetition number.
        run_id: Unique run identifier.

    Returns:
        File name string (without extension).
    """
    return f"{scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"


def _load_existing_results(
    path: Path, python_class_identifier=PYTHON_CLASS_IDENTIFIER
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load existing results from a tournament results/ folder.

    Args:
        path: Path to the tournament directory.
        python_class_identifier: Function to convert classes to string identifiers.

    Returns:
        Tuple of (results_list, scores_list) where each is a list of dictionaries.
        Returns ([], []) if results/ folder doesn't exist or is empty.
    """
    results_dir = path / RESULTS_DIR_NAME
    if not results_dir.exists():
        return [], []

    results = []
    scores = []

    # Load all JSON files from results/ folder
    for result_file in results_dir.glob("*.json"):
        try:
            record = load(result_file)
            if record and isinstance(record, dict):
                results.append(record)
                # Generate scores from this record
                scored_indices = record.get("scored_indices")
                scores += make_scores(record, scored_indices=scored_indices)
        except Exception:
            # Skip corrupted or incomplete files
            continue

    return results, scores


def _check_completed_runs(
    path: Path, python_class_identifier=PYTHON_CLASS_IDENTIFIER
) -> set[str]:
    """Check which runs have been completed based on results/ folder.

    Args:
        path: Path to the tournament directory.
        python_class_identifier: Function to convert classes to string identifiers.

    Returns:
        Set of file names (without extension) for completed runs.
    """
    results_dir = path / RESULTS_DIR_NAME
    if not results_dir.exists():
        return set()

    completed = set()
    for result_file in results_dir.glob("*.json"):
        # Extract file name without extension
        file_name = result_file.stem
        try:
            # Verify the file is non-empty and valid JSON
            record = load(result_file)
            if record and isinstance(record, dict):
                completed.add(file_name)
        except Exception:
            # Skip corrupted or incomplete files
            continue

    return completed


def _is_run_completed(
    run_info: dict[str, Any],
    completed_runs: set[str],
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
) -> bool:
    """Check if a specific run has been completed.

    Args:
        run_info: Run information dictionary.
        completed_runs: Set of completed run file names.
        python_class_identifier: Function to convert classes to string identifiers.

    Returns:
        True if this run has been completed, False otherwise.
    """
    # Generate the run_id if not present
    if "run_id" not in run_info:
        run_info["run_id"] = hash_to_base64(
            hash(
                str(
                    serialize(run_info, python_class_identifier=python_class_identifier)
                )
            )
        )

    file_name = _get_run_file_name(
        scenario_name=run_info["s"].outcome_space.name,
        partner_names=run_info["partner_names"],
        rep=run_info["rep"],
        run_id=run_info["run_id"],
    )
    return file_name in completed_runs


def cartesian_tournament(
    competitors: list[type[Negotiator] | str] | tuple[type[Negotiator] | str, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    opponents: list[type[Negotiator] | str]
    | tuple[type[Negotiator] | str, ...]
    | None = tuple(),
    opponent_params: list[dict | None] | None = None,
    opponent_names: list[str] | None = None,
    private_infos: list[None | tuple[dict, ...]] | None = None,
    competitor_params: list[dict | None] | None = None,
    competitor_names: list[str] | None = None,
    rotate_ufuns: bool = True,
    rotate_private_infos: bool = True,
    n_repetitions: int = 1,
    path: Path | None = None,
    path_exists: Literal["continue", "overwrite", "fail"] = "continue",
    njobs: int = 0,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    n_steps: int | tuple[int, int] | None = 100,
    time_limit: float | tuple[float, float] | None = None,
    pend: float | tuple[float, float] = 0.0,
    pend_per_second: float | tuple[float, float] = 0.0,
    step_time_limit: float | tuple[float, float] | None = None,
    negotiator_time_limit: float | tuple[float, float] | None = None,
    hidden_time_limit: float | tuple[float, float] | None = None,
    external_timeout: int | None = None,
    # full_names: bool = True,
    plot_fraction: float = 0.0,
    plot_params: dict[str, Any] | None = None,
    verbosity: int = 1,
    self_play: bool = True,
    randomize_runs: bool = True,
    sort_runs: bool = False,
    save_every: int = 0,
    save_stats: bool = True,
    save_scenario_figs: bool = True,
    recalculate_stats: bool = False,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    opponent_modeling_metrics: tuple[
        COMPARE_UFUN_METHOD_TYPE, ...
    ] = (),  # valid values are
    distribute_opponent_modeling_scores: bool = True,
    raw_aggregated_metrics: dict[str, Callable[[dict[str, float]], float]]
    | None = None,
    stats_aggregated_metrics: dict[str, Callable[[dict[tuple[str, str], float]], float]]
    | None = None,
    final_score: tuple[str, str] = ("advantage", "mean"),
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    shorten_names: bool | None = None,
    raise_exceptions: bool = True,
    mask_scenario_names: bool = True,
    only_failures_on_self_play: bool = False,
    ignore_discount: bool = False,
    ignore_reserved: bool = False,
    normalize_ufuns: bool = True,
    reserved_value_eps: float = 0.0,
    storage_optimization: OptimizationLevel = "space",
    memory_optimization: OptimizationLevel = "balanced",
    storage_format: StorageFormat | None = None,
    save_negotiations_as_folders: bool = False,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
    before_start_callback: BeforeStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    neg_start_callback: NegStartCallback | None = None,
    after_construction_callback: AfterConstructionCallback | None = None,
    neg_progress_callback: NegProgressCallback | None = None,
    neg_end_callback: NegEndCallback | None = None,
    after_end_callback: AfterEndCallback | None = None,
    metadata: dict[str, Any] | None = None,
) -> SimpleTournamentResults:
    """Run a Cartesian tournament where negotiators compete across multiple scenarios.

    This function runs negotiations between all combinations of competitors across all scenarios,
    optionally with rotated utility functions. When opponents are provided, competitors only play
    against opponents (not each other) and only competitor scores are recorded.

    Args:
        competitors: Negotiator types or class names to compete in the tournament.
        scenarios: Negotiation scenarios, each with an outcome space and utility functions.
        opponents: Optional negotiator types to use as opponents. If provided, competitors only
                  play against opponents (not each other) and only competitors are scored.
                  Competitors will be tested in both first and last positions to evaluate
                  different roles (e.g., buyer vs seller).
        opponent_params: Parameters for initializing opponents (one dict per opponent type).
        opponent_names: Optional list of custom names for opponents. If provided, must have
                       the same length as opponents and contain unique names. If not provided,
                       names are generated from class names using shortest_unique_names().

        private_infos: Private information passed to negotiators via their `private_info` attribute.
                      Must be a list of tuples, one tuple per scenario.
        competitor_params: Parameters for initializing competitors (one dict per competitor type).
        competitor_names: Optional list of custom names for competitors. If provided, must have
                         the same length as competitors and contain unique names. If not provided,
                         names are generated from class names using shortest_unique_names().

        rotate_ufuns: If True, utility functions are rotated across negotiator positions.
                     For bilateral negotiations, this creates scenarios with reversed preferences.
                     Not recommended when using explicit opponents as roles become ambiguous.
        rotate_private_infos: If True and rotate_ufuns is True, rotate private information with ufuns.
        n_repetitions: Number of times to repeat each scenario/partner combination.
        path: Directory path to save tournament results. If None, results are not saved to disk.
        path_exists: Controls behavior when path already exists (default: "continue"):
                 - "continue": Resume incomplete tournament by skipping completed negotiations
                 - "overwrite": Delete existing tournament and start fresh
                 - "fail": Raise FileExistsError if tournament directory exists
        njobs: Parallelization level. -1 for serial execution (good for debugging),
              0 for all available cores, positive integer for specific number of processes.
        mechanism_type: The negotiation protocol/mechanism class to use (default: SAOMechanism).
        mechanism_params: Additional parameters passed to the mechanism constructor.
        n_steps: Maximum rounds per negotiation. Can be int, (min, max) tuple for random sampling, or None for unlimited.
        time_limit: Maximum seconds per negotiation. Can be float, (min, max) tuple, or None.
        pend: Probability of ending negotiation each step. Can be float, (min, max) tuple, or 0.0.
        pend_per_second: Probability of ending negotiation each second. Can be float, (min, max) tuple, or 0.0.
        step_time_limit: Maximum seconds per negotiation step. Can be float, (min, max) tuple, or None.
        negotiator_time_limit: Maximum total seconds for all actions by each negotiator.
        hidden_time_limit: Time limit not revealed to negotiators.
        external_timeout: Timeout in seconds for receiving results from parallel negotiations.
        plot_fraction: Fraction of negotiations to plot (0.0 to 1.0). Only used if path is provided.
        plot_params: Parameters passed to plotting functions.
        verbosity: Logging level (0 for silent, higher for more verbose).
        self_play: If True, allow negotiations where all parties are the same type.
        only_failures_on_self_play: If True, only record self-play negotiations that fail to reach agreement.
        randomize_runs: If True, run negotiations in random order instead of sequentially.
        sort_runs: If True, sort runs by scenario size before execution.
        save_every: Save results to disk after this many negotiations (0 to disable periodic saving).
        save_stats: If True, calculate optimality statistics (Pareto, Nash, Kalai-Smorodinsky, etc.).
        save_scenario_figs: If True, save visualizations of scenarios in utility space.
        recalculate_stats: If True, always recalculate stats even if loaded from disk (old behavior).
                          If False (default), load stats/info/plots from disk when available and use
                          Scenario.rotate_ufuns() to create rotated versions efficiently. This dramatically
                          speeds up tournaments by avoiding redundant stat calculations. When rotate_ufuns=False,
                          stats are never recalculated if available in the scenario folder.
        image_format: Format for saving figures. Supported formats: 'webp', 'png', 'jpg', 'jpeg', 'svg', 'pdf'.
                     Default is 'webp'. Applies to both scenario figures and run plots.
        opponent_modeling_metrics: Tuple of utility function comparison methods for evaluating
                                  how well each negotiator models their opponent's preferences.
                                  For each metric specified, compares the negotiator's `opponent_ufun`
                                  attribute (if available) with the actual opponent utility function.
                                  Results are added as `opp_<metric>` columns in the scores DataFrame.

                                  Valid values include:
                                  - 'kendall_optimality': Kendall tau correlation (-1 to 1)
                                  - 'ordinal_optimality': Ordinal ranking similarity (0 to 1)
                                  - 'cardinal_optimality': Cardinal value similarity (0 to 1)
                                  - 'utility_optimality': Direct utility comparison (0 to 1)
                                  - 'pareto_optimality': Pareto efficiency measure (0 to 1)
                                  - 'nash_optimality': Nash bargaining optimality (0 to 1)
                                  - 'kalai_optimality': Kalai-Smorodinsky optimality (0 to 1)
                                  - 'max_welfare_optimality': Maximum welfare optimality (0 to 1)

                                  These can be used as final scores: final_score=('opp_kendall_optimality', 'mean')

                                  Example::

                                      results = cartesian_tournament(
                                          competitors=[MyNegotiator, RandomNegotiator],
                                          scenarios=scenarios,
                                          opponent_modeling_metrics=(
                                              "kendall_optimality",
                                              "euclidean_optimality",
                                          ),
                                      )
                                      # Access opponent modeling scores
                                      print(
                                          results.scores[
                                              [
                                                  "strategy",
                                                  "opp_kendall_optimality",
                                                  "opp_euclidean_optimality",
                                              ]
                                          ]
                                      )

        raw_aggregated_metrics: Optional dict mapping custom metric names to aggregation functions.
                              Each function receives a dict of {metric_name: value} containing all
                              per-negotiation metrics for a single negotiator (e.g., 'advantage',
                              'utility', 'nash_optimality', etc.) and returns a combined score.
                              Results are added as new columns in the scores DataFrame.

                              Useful for creating weighted combinations of existing metrics.

                              Example::

                                  results = cartesian_tournament(
                                      competitors=[...],
                                      scenarios=[...],
                                      raw_aggregated_metrics={
                                          "combined": lambda d: d.get("advantage", 0)
                                          * 0.5
                                          + d.get("utility", 0) * 0.5,
                                          "risk_adjusted": lambda d: d.get("utility", 0)
                                          - 0.1 * d.get("partner_welfare", 0),
                                      },
                                  )
                                  # Use in final score
                                  # final_score=('combined', 'mean')

        stats_aggregated_metrics: Optional dict mapping custom metric names to aggregation functions
                                that operate on summary statistics across all negotiations.
                                Each function receives a dict of {(metric_name, stat_name): value}
                                where stat_name can be 'mean', 'std', 'min', 'max', '25%', '50%', '75%', 'count'.
                                Results are added as (metric_name, 'value') columns in scores_summary.

                                This is particularly useful for creating custom final scores that combine
                                multiple statistics in ways not possible with standard aggregations.

                                Example::

                                      results = cartesian_tournament(
                                          competitors=[...],
                                          scenarios=[...],
                                          stats_aggregated_metrics={
                                              'risk_adjusted_score': lambda d: (
                                                  d.get(('advantage', 'mean'), 0) - 0.5 * d.get(('advantage', 'std'), 0)
                                              ),
                                              'weighted_final': lambda d: (
                                                  d.get(('advantage', 'mean'), 0) * 0.7 +
                                                  d.get(('utility', 'mean'), 0) * 0.3
                                              ),
                                          },
                                          # Use custom aggregation as final score
                                          final_score=('weighted_final', 'value'),
                                      )
        final_score: Tuple of (metric, statistic) for ranking. Metric can be 'advantage', 'utility',
                    'partner_welfare', 'welfare', or any calculated statistic. Statistic can be
                    'mean', 'median', 'min', 'max', or 'std'. Default: ('advantage', 'mean').
        id_reveals_type: If True, negotiator IDs reveal their type (for analysis).
        name_reveals_type: If True, negotiator names reveal their type.
        shorten_names: Deprecated. Use competitor_names and opponent_names instead.
                      This parameter is ignored and will be removed in a future version.
        raise_exceptions: If True, exceptions from negotiators/mechanisms stop the tournament.
                         If False, exceptions are logged but tournament continues.
        mask_scenario_names: If True, mask scenario names from negotiators.
        only_failures_on_self_play: If True, only record self-play runs that fail to reach agreement.
        ignore_discount: If True, ignore discounting in utility functions (use base ufun).
        ignore_reserved: If True, ignore reserved values in utility functions.
        reserved_value_eps: Epsilon value used to correct problematic reserved values (default: 0.0).
                           When a utility function has None, inf, -inf, or NaN reserved value, it will be
                           corrected to ufun.min() - reserved_value_eps. A warning is emitted for each
                           corrected utility function.
        normalize_ufuns: If True (default), all utility functions are normalized to [0, 1] range before
                        negotiation. Normalization is applied independently to each ufun, guaranteeing
                        that the best outcome has utility 1.0 and the worst has utility 0.0. This
                        normalization happens BEFORE scenarios are saved to disk, so all saved scenarios,
                        statistics, and figures reflect the normalized utilities.
        storage_optimization: Controls disk space usage for tournament results (default: "space"):
                            - "speed"/"time"/"none": Keep all files (results/, all_scores.csv, details.csv, etc.)
                            - "balanced": Remove results/ folder after details.csv is created
                            - "space"/"max": Remove both results/ folder AND all_scores.csv (default)
                              (scores can be reconstructed from details.csv)
        memory_optimization: Controls RAM usage for returned SimpleTournamentResults (default: "balanced"):
                           - "speed"/"time"/"none": Keep all DataFrames in memory
                           - "balanced": Keep details + final_scores + scores_summary in memory,
                                        compute scores on demand then cache (default)
                           - "space"/"max": Keep only final_scores + scores_summary in memory,
                                     load details/scores from disk when needed
                           Note: If path is None, memory_optimization is ignored (everything kept in memory)
        storage_format: Storage format for large data files (all_scores, details):
                       - "csv": Plain CSV files (human-readable, larger size)
                       - "gzip": Gzip-compressed CSV files (good compression, human-readable when decompressed)
                       - "parquet": Parquet binary format (best compression, preserves types, fastest)
                       - None: Auto-select based on storage_optimization (default; csv for speed, gzip for balanced, parquet for space)
                       Note: Small files (scores.csv, type_scores.csv) are always CSV regardless of this setting.
        save_negotiations_as_folders: If True, save each negotiation as a folder containing trace,
                                     agreement_stats (optimality measures), config, and metadata files.
                                     If False (default), save as single trace files for compactness.
        python_class_identifier: Function to convert classes to string identifiers.
        before_start_callback: Optional callback invoked before each negotiation starts.
                              Receives a RunInfo object with all negotiation parameters.
                              Useful for logging, monitoring, or custom setup. Exceptions are caught
                              and logged (if verbosity > 0) but don't stop the tournament.
        after_construction_callback: Optional callback invoked after mechanism construction but before
                                    negotiation starts. Receives a ConstructedNegInfo object with the
                                    constructed mechanism and scenario details. Useful for inspecting
                                    or modifying the mechanism before negotiation. Exceptions are caught
                                    and logged (if verbosity > 0) but don't stop the tournament.
        after_end_callback: Optional callback invoked after each negotiation completes.
                           Receives the complete negotiation record dictionary with agreement,
                           utilities, and all result metadata. Useful for custom analysis or
                           logging. Exceptions are caught and logged (if verbosity > 0) but
                           don't stop the tournament.
        progress_callback: Optional callback invoked during tournament setup to report progress.
                          Receives (message: str, current: int, total: int) where message describes
                          the current phase, current is the progress index, and total is the
                          expected count. Useful for showing setup progress in UIs before
                          negotiations start. Called during competitor validation, scenario
                          processing, and run configuration building.
        neg_start_callback: Optional callback invoked at the start of each negotiation.
                           Receives (run_id: int | str, state: SAOState) where run_id uniquely
                           identifies the negotiation and state is the initial mechanism state.
                           Useful for monitoring negotiation progress. Exceptions are caught
                           and logged but don't stop the tournament.

                           **Parallel Execution Note:** Callbacks do NOT need to be defined at module
                           level. You can use local functions, lambdas, or closures. However, when
                           running in parallel mode (njobs > 0), callbacks are serialized using
                           cloudpickle and executed in separate worker processes. This means:

                           - Callbacks can capture local variables from enclosing scopes (closures work)
                           - **IMPORTANT:** Modifications to captured variables (lists, dicts, etc.) will
                             NOT be visible in the parent process. Each worker gets a copy of the closure.
                           - To collect results from parallel callbacks, use side effects that persist
                             across processes (e.g., write to files, database, use multiprocessing.Manager)
                           - Callbacks must be picklable (avoid unpicklable objects like file handles, locks)

        neg_progress_callback: Optional callback invoked after each step of each negotiation.
                              Receives (run_id: int | str, state: SAOState) where state contains
                              current step number, offers, and agreement status. Useful for real-time
                              monitoring of negotiation progress. Exceptions are caught and logged
                              but don't stop the tournament.

                              See neg_start_callback documentation for parallel execution requirements.

        neg_end_callback: Optional callback invoked at the completion of each negotiation.
                         Receives (run_id: int | str, state: SAOState) where state contains
                         the final agreement, step count, and termination reason. Useful for
                         analyzing negotiation outcomes in real-time. Exceptions are caught
                         and logged but don't stop the tournament.

                         See neg_start_callback documentation for parallel execution requirements.
        metadata: Optional dictionary of metadata to include in tournament results.

    Returns:
        SimpleTournamentResults containing scores, detailed results, score summaries, and final rankings.

    Notes:
        - In explicit opponent mode (opponents provided), competitors appear at the first
          position. Use rotate_ufuns=True to test performance in different roles.
        - Use njobs=-1 for debugging to run serially and see full tracebacks.

    Examples:
        Normal tournament between two negotiators:
        ```python
        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=[scenario1, scenario2],
            n_steps=100,
            path=Path("results/"),
        )
        ```

        Testing a negotiator against fixed opponents:
        ```python
        results = cartesian_tournament(
            competitors=[MyNegotiator],
            opponents=[RandomNegotiator, AspirationNegotiator],
            scenarios=[scenario1],
            rotate_ufuns=False,  # Keep roles fixed
        )
        ```

        Using callbacks for monitoring:
        ```python
        def log_start(info: RunInfo):
            print(f"Starting negotiation {info.rep} with {info.partners}")


        def log_end(record: dict):
            print(f"Ended with agreement: {record['agreement']}")


        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=[scenario1],
            before_start_callback=log_start,
            after_end_callback=log_end,
        )
        ```

        Using per-negotiation callbacks (local closures work in both serial and parallel):
        ```python
        # Serial mode - can modify local variables (closure copy)
        start_times = []


        def track_start(run_id, state):
            print(f"Negotiation {run_id} started at step {state.step}")
            start_times.append(run_id)  # Won't work in parallel mode!


        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=[scenario1],
            neg_start_callback=track_start,
            njobs=-1,  # Serial mode
        )

        # Parallel mode - use files or database for side effects
        from pathlib import Path

        log_dir = Path("negotiation_logs")
        log_dir.mkdir(exist_ok=True)


        def track_start_parallel(run_id, state):
            # Write to file - works across processes
            msg = f"Started at step {state.step}"
            (log_dir / f"start_{run_id}.log").write_text(msg)


        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=[scenario1],
            neg_start_callback=track_start_parallel,
            njobs=4,  # Parallel mode - callbacks still work!
        )
        ```

        Resuming an interrupted tournament:
        ```python
        # Start a tournament
        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=scenarios,
            n_repetitions=100,
            path=Path("results/"),
            path_exists="continue",  # Resume if interrupted (default)
        )

        # If interrupted and restarted, only remaining negotiations will run
        # Use path_exists="overwrite" to delete and restart from scratch
        # Use path_exists="fail" to raise error if directory exists
        ```
    """

    # Handle deprecated shorten_names parameter
    if shorten_names is not None:
        deprecated(
            "The 'shorten_names' parameter is deprecated and will be removed in a future version. "
            "Use 'competitor_names' and 'opponent_names' to customize names instead. "
            "Names are now always generated using shortest_unique_names()."
        )

    if mechanism_params is None:
        mechanism_params = dict()
    mechanism_params["ignore_negotiator_exceptions"] = not raise_exceptions

    # Validate image_format
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"image_format must be one of {SUPPORTED_IMAGE_FORMATS}, got '{image_format}'"
        )

    # Create deep copies of scenarios to avoid modifying the originals
    import copy

    scenarios = [copy.deepcopy(s) for s in scenarios]

    if ignore_discount:
        for s in scenarios:
            s.remove_discounting(recalculate_stats=False)

    if normalize_ufuns:
        scenarios = [
            s.normalize(
                guarantee_max=True,
                guarantee_min=True,
                common_range=False,
                recalculate_stats=True,
            )
            if not s.is_normalized(common_range=False)
            else s
            for s in scenarios
        ]
    if ignore_reserved:
        for s in scenarios:
            s.remove_reserved_values(recalculate_stats=True)

    # Normalize optimization levels (treat "time"/"none" as "speed", "max" as "space")
    storage_optimization = _normalize_optimization_level(storage_optimization)
    memory_optimization = _normalize_optimization_level(memory_optimization)

    # Set default storage format if not provided
    effective_storage_format: StorageFormat = (
        storage_format
        if storage_format is not None
        else _get_default_storage_format(storage_optimization)
    )

    competitors = [get_class(_) for _ in competitors]
    opponents = [get_class(_) for _ in opponents] if opponents else []
    if competitor_params is None:
        competitor_params = [dict() for _ in competitors]
    if opponent_params is None:
        opponent_params = [dict() for _ in opponents]
    if private_infos is None:
        private_infos = [tuple(dict() for _ in s.ufuns) for s in scenarios]

    def check_generate_names_and_map(
        negotiators: list[type[Negotiator] | str],
        params: list[dict | None] | None,
        names: list[str] | None,
        name_type="competitor",
    ) -> tuple[list[str], dict[str, str]]:
        if params is None:
            params = [dict() for _ in negotiators]
        original_names = [get_full_type_name(_) for _ in negotiators]
        if names is None:
            names = shortest_unique_names(
                [
                    get_full_type_name(_) + encode_params(p)
                    for _, p in zip(negotiators, params)
                ]
            )

        if len(names) != len(negotiators):
            raise ValueError(
                f"names length ({len(names)}) must match "
                f"{name_type}s length ({len(negotiators)})"
            )
        if len(set(names)) != len(names):
            duplicates = [name for name in set(names) if names.count(name) > 1]
            raise ValueError(
                f"names must be unique. Duplicate names found: {duplicates}"
            )
        return names, dict(zip(names, original_names))

    # Generate names if not provided by user
    competitor_names, competitor_map_ = check_generate_names_and_map(
        competitors, competitor_params, competitor_names, "competitor"
    )
    opponent_names, opponent_map_ = (
        check_generate_names_and_map(
            opponents, opponent_params, opponent_names, "opponent"
        )
        if opponents
        else ([], dict())
    )

    config = dict(
        n_scenarios=len(scenarios),
        n_competitors=len(competitors),
        competitors=[get_full_type_name(_) for _ in competitors],
        competitor_names=competitor_names,
        competitor_params=serialize(
            competitor_params, python_class_identifier=python_class_identifier
        ),
        competitor_type_map=competitor_map_,
        opponent_type_map=opponent_map_ if opponents else {},
        opponents=[get_full_type_name(_) for _ in opponents] if opponents else None,
        n_opponents=len(opponents) if opponents else 0,
        opponent_names=opponent_names,
        opponent_params=serialize(
            opponent_params, python_class_identifier=python_class_identifier
        ),
        private_infos=serialize(
            private_infos, python_class_identifier=python_class_identifier
        ),
        opponent_modeling_metrics=serialize(opponent_modeling_metrics),
        raw_aggregated_metrics=serialize(raw_aggregated_metrics),
        stats_aggregated_metrics=serialize(stats_aggregated_metrics),
        rotate_ufuns=rotate_ufuns,
        rotate_private_infos=rotate_private_infos,
        n_repetitions=n_repetitions,
        path=str(path) if path else None,
        njobs=njobs,
        mechanism_type=get_full_type_name(mechanism_type),
        mechanism_params=serialize(
            mechanism_params, python_class_identifier=python_class_identifier
        ),
        n_steps=n_steps,
        time_limit=time_limit,
        pend=pend,
        pend_per_second=pend_per_second,
        step_time_limit=step_time_limit,
        negotiator_time_limit=negotiator_time_limit,
        hidden_time_limit=hidden_time_limit,
        external_timeout=external_timeout,
        plot_fraction=plot_fraction,
        plot_params=serialize(
            plot_params, python_class_identifier=python_class_identifier
        ),
        verbosity=verbosity,
        self_play=self_play,
        randomize_runs=randomize_runs,
        sort_runs=sort_runs,
        save_every=save_every,
        save_stats=save_stats,
        save_scenario_figs=save_scenario_figs,
        recalculate_stats=recalculate_stats,
        image_format=image_format,
        final_score=final_score,
        id_reveals_type=id_reveals_type,
        name_reveals_type=name_reveals_type,
        shorten_names=shorten_names,
        raise_exceptions=raise_exceptions,
        mask_scenario_names=mask_scenario_names,
        only_failures_on_self_play=only_failures_on_self_play,
        ignore_discount=ignore_discount,
        ignore_reserved=ignore_reserved,
        normalize_ufuns=normalize_ufuns,
        reserved_value_eps=reserved_value_eps,
        distribute_opponent_modeling_scores=distribute_opponent_modeling_scores,
        storage_optimization=storage_optimization,
        memory_optimization=memory_optimization,
        storage_format=storage_format,
        save_negotiations_as_folders=save_negotiations_as_folders,
        python_class_identifier=python_class_identifier,
        has_before_start_callback=before_start_callback is not None,
        has_after_construction_callback=after_construction_callback is not None,
        has_after_end_callback=after_end_callback is not None,
        has_progress_callback=progress_callback is not None,
        has_neg_progress_callback=neg_progress_callback is not None,
        has_start_callback=neg_start_callback is not None,
        has_end_callback=neg_end_callback is not None,
        metadata=serialize(metadata),
    )

    # Handle existing tournament directory based on path_exists mode
    existing_results: list[dict[str, Any]] = []
    existing_scores: list[dict[str, Any]] = []
    completed_runs: set[str] = set()

    if path:
        path_obj = Path(path)

        # Check if tournament directory already exists
        tournament_exists = path_obj.exists() and any(path_obj.iterdir())

        if tournament_exists:
            if path_exists == "fail":
                raise FileExistsError(
                    f"Tournament directory already exists at {path}. "
                    "Use path_exists='continue' to resume or path_exists='overwrite' to delete and restart."
                )
            elif path_exists == "overwrite":
                if verbosity > 0:
                    print(f"[yellow]Deleting existing tournament at {path}...[/yellow]")
                shutil.rmtree(path_obj)
                path_obj.mkdir(parents=True, exist_ok=True)
            elif path_exists == "continue":
                # Load existing results from results/ folder for partial completion
                # We'll check if it's complete after we know what runs we need
                existing_results, existing_scores = _load_existing_results(
                    path_obj, python_class_identifier
                )
                completed_runs = _check_completed_runs(
                    path_obj, python_class_identifier
                )

                if completed_runs and verbosity > 0:
                    print(
                        f"[yellow]Found {len(completed_runs)} completed negotiations.[/yellow]"
                    )

        # Create directory if it doesn't exist
        path_obj.mkdir(parents=True, exist_ok=True)

    # Save config to disk if path is provided
    if path:
        dump(config, Path(path) / CONFIG_FILE_NAME)

    # Report progress: competitors loaded
    if progress_callback:
        try:
            _call_progress_callback(
                progress_callback,
                f"Loaded {len(competitors)} competitors",
                1,
                4,
                config,
            )
        except Exception:
            pass

    # Determine if we're in explicit opponent mode
    explicit_opponents = len(opponents) > 0

    runs = []
    scenarios_path = path if path is None else Path(path) / SCENARIOS_DIR_NAME
    if scenarios_path is not None:
        scenarios_path.mkdir(exist_ok=True, parents=True)

    competitor_info = list(
        zip(competitors, competitor_params, competitor_names, strict=True)
    )
    opponent_info = (
        list(zip(opponents, opponent_params, opponent_names, strict=True))
        if opponents
        else []
    )
    n_scenarios = len(scenarios)
    for scenario_idx, (s, pinfo) in enumerate(zip(scenarios, private_infos)):
        # Report progress: processing scenarios
        if progress_callback:
            try:
                _call_progress_callback(
                    progress_callback,
                    f"Processing scenario {scenario_idx + 1}/{n_scenarios}",
                    2,
                    4,
                    config,
                )
            except Exception:
                pass

        pinfolst = list(pinfo) if pinfo else [dict() for _ in s.ufuns]
        n = len(s.ufuns)

        # Generate partners_list based on whether we have explicit opponents
        if explicit_opponents:
            # In explicit opponent mode, create combinations where:
            # - Competitor at position 0, opponents fill remaining positions
            # When rotate_ufuns=True, ufun rotation naturally tests different roles
            if n < 2:
                continue
            partners_list = []
            # Competitor at position 0 (first position)
            for comp_info in competitor_info:
                for opp_combo in product(*[opponent_info] * (n - 1)):
                    partners_list.append((comp_info,) + opp_combo)
        else:
            # Normal mode: all competitors play all positions
            partners_list = list(product(*tuple([competitor_info] * n)))

        if not self_play:
            partners_list = [
                _
                for _ in partners_list
                if len(
                    {
                        str(
                            serialize(
                                p, python_class_identifier=python_class_identifier
                            )
                        )
                        for p in _
                    }
                )
                > 1
            ]

        # Helper function to process ufuns based on ignore_discount and ignore_reserved
        def process_ufun(ufun):
            """Process a utility function, optionally removing discounting and/or reserved value."""
            result = ufun
            # Strip discounting if requested
            if ignore_discount:
                while isinstance(result, DiscountedUtilityFunction):
                    inner = result.extract_base_ufun()
                    result = inner
            # Clear reserved value if requested
            if ignore_reserved and hasattr(result, "reserved_value"):
                result = copy.deepcopy(result)
                result.reserved_value = float("-inf")
            return result

        # NEW OPTIMIZED APPROACH (when recalculate_stats=False, the default):
        # - Load scenarios WITH stats/info/plots from disk
        # - Use Scenario.rotate_ufuns() to create rotated versions (stats are rotated, not recalculated)
        # - Only calculate stats if not available in the original scenario folder
        # - When rotate_ufuns=False, never recalculate if stats exist

        # OLD APPROACH (when recalculate_stats=True):
        # - Load scenarios WITHOUT stats
        # - Manually rotate ufuns and create new scenarios
        # - Recalculate stats for every rotation

        # Generate scenarios list based on recalculate_stats flag
        # scenarios_to_process is a list of (scenario, scenario_name, pinfo_tuple)
        scenarios_to_process = []
        original_name = s.outcome_space.name

        if recalculate_stats:
            # OLD BEHAVIOR: Process ufuns manually and recalculate stats for each rotation
            processed_ufuns = [process_ufun(u) for u in s.ufuns]
            ufun_sets = [[copy.deepcopy(_) for _ in processed_ufuns]]
            pinfo_sets = [pinfo]
            for i, u in enumerate(processed_ufuns):
                u.name = f"{i}_{u.name}"
            if rotate_ufuns:
                for _ in range(len(ufun_sets)):
                    ufuns = ufun_sets[-1]
                    ufun_sets.append([ufuns[-1]] + ufuns[:-1])
                    if rotate_private_infos and pinfolst:
                        pinfo_sets.append(tuple([pinfolst[-1]] + pinfolst[:-1]))
                    else:
                        pinfo_sets.append(pinfo)

            for i, (ufuns, pinfo_tuple) in enumerate(zip(ufun_sets, pinfo_sets)):
                if len(ufun_sets) > 1:
                    for j, u in enumerate(ufuns):
                        n = "_".join(u.name.split("_")[1:])
                        u.name = f"{j}_{n}"
                    scenario = Scenario(
                        type(s.outcome_space)(
                            issues=s.outcome_space.issues,
                            name=f"{original_name}-{i}" if i else original_name,
                        ),
                        tuple(ufuns),  # type: ignore
                    )
                    scenario_name = f"{original_name}-{i}" if i else original_name
                else:
                    scenario = s
                    scenario_name = original_name

                # Calculate stats if needed (before saving to ensure stats are available)
                if save_stats:
                    scenario.calc_stats()

                scenarios_to_process.append((scenario, scenario_name, pinfo_tuple))
        else:
            # NEW OPTIMIZED BEHAVIOR: Use rotate_ufuns() to avoid recalculating stats
            # Process scenario with ignore_discount/ignore_reserved
            base_scenario = copy.deepcopy(s)
            base_scenario.ufuns = tuple([process_ufun(u) for u in base_scenario.ufuns])  # type: ignore

            # Rename ufuns with indices
            for i, u in enumerate(base_scenario.ufuns):
                u.name = f"{i}_{u.name}"

            # Calculate stats only if not available and save_stats=True
            if save_stats and base_scenario.stats is None:
                base_scenario.calc_stats()

            # Determine number of rotations
            n_rotations = len(base_scenario.ufuns) if rotate_ufuns else 1

            # Generate all scenarios (base + rotations)
            for rotation_idx in range(n_rotations):
                if rotation_idx == 0:
                    # Base scenario (no rotation)
                    scenario = base_scenario
                    scenario_name = original_name
                    pinfo_tuple = pinfo
                else:
                    # Rotated scenario - use rotate_ufuns() to efficiently create variant
                    scenario = base_scenario.rotate_ufuns(
                        n=rotation_idx, rotate_info=rotate_private_infos
                    )
                    scenario_name = f"{original_name}-{rotation_idx}"

                    # Update outcome space name
                    scenario.outcome_space = type(s.outcome_space)(
                        issues=s.outcome_space.issues, name=scenario_name
                    )

                    # Update ufun names to reflect rotation
                    for j, u in enumerate(scenario.ufuns):
                        n = "_".join(u.name.split("_")[1:])
                        u.name = f"{j}_{n}"

                    # Handle private info rotation
                    if rotate_private_infos and pinfolst:
                        # Rotate private info the same way
                        rotated_pinfolst = (
                            pinfolst[-rotation_idx:] + pinfolst[:-rotation_idx]
                        )
                        pinfo_tuple = tuple(rotated_pinfolst)
                    else:
                        pinfo_tuple = pinfo

                scenarios_to_process.append((scenario, scenario_name, pinfo_tuple))

        # COMMON CODE: Process all scenarios (from either path above)
        for scenario, scenario_name, pinfo_tuple in scenarios_to_process:
            # Check for problematic reserved values, correct them, and warn
            _check_and_correct_reserved_values(
                scenario, scenario_name, reserved_value_eps
            )

            this_path = None

            # Save scenario to disk if scenarios_path is provided
            if scenarios_path:
                this_path = scenarios_path / scenario_name
                # Exclude pareto frontier when optimizing for space
                include_pareto = storage_optimization != "space"
                # Use dumpas to save scenario with stats
                scenario.dumpas(
                    this_path,
                    type="yml",
                    compact=False,
                    save_stats=save_stats,
                    save_info=True,
                    include_pareto_frontier=include_pareto,
                )
                if save_scenario_figs:
                    plot_offline_run(
                        trace=[],
                        ids=["First", "Second"],
                        ufuns=scenario.ufuns[:2],  # type: ignore
                        agreement=None,
                        timedout=False,
                        broken=False,
                        has_error=False,
                        names=["First", "Second"],
                        save_fig=True,
                        path=str(this_path),
                        fig_name=f"fig.{image_format}",
                        only2d=True,
                        show_annotations=False,
                        show_agreement=False,
                        show_pareto_distance=False,
                        show_nash_distance=False,
                        show_kalai_distance=False,
                        show_ks_distance=False,
                        show_max_welfare_distance=False,
                        show_max_relative_welfare_distance=False,
                        show_end_reason=False,
                        show_reserved=True,
                        show_total_time=False,
                        show_relative_time=False,
                        show_n_steps=False,
                        show=False,
                    )

            # Prepare mechanism parameters
            mparams = copy.deepcopy(mechanism_params)
            mparams.update(
                dict(
                    n_steps=oneinint(n_steps),
                    time_limit=oneinfloat(time_limit),
                    pend=oneinfloat(pend),
                    pend_per_second=oneinfloat(pend_per_second),
                    negotiator_time_limit=oneinfloat(negotiator_time_limit),
                    step_time_limit=oneinfloat(step_time_limit),
                    hidden_time_limit=oneinfloat(hidden_time_limit),
                )
            )
            if scenarios_path:
                params_path = scenarios_path / scenario_name / MECHANISM_FILE_NAME
                pdict = dict(type=get_full_type_name(mechanism_type)) | mparams
                dump(pdict, params_path)

            # Generate runs for all partner combinations
            for partners in partners_list:
                # Determine which positions should be scored
                if explicit_opponents:
                    # Find which position(s) contain competitors
                    scored_indices = [
                        i for i, p in enumerate(partners) if p in competitor_info
                    ]
                else:
                    # Score all positions in normal mode
                    scored_indices = None

                for i in range(n_repetitions):
                    run_dict = dict(
                        s=scenario,
                        partners=[_[0] for _ in partners],
                        partner_names=[_[2] for _ in partners],
                        partner_params=[_[1] for _ in partners],
                        rep=i,
                        annotation=dict(rep=i, n_repetitions=n_repetitions),
                        path=path if path else None,
                        mechanism_type=mechanism_type,
                        mechanism_params=mparams,
                        full_names=True,
                        verbosity=verbosity - 1,
                        plot=random() < plot_fraction,
                        stats=scenario.stats if save_stats else None,
                        id_reveals_type=id_reveals_type,
                        name_reveals_type=name_reveals_type,
                        plot_params=plot_params,
                        mask_scenario_name=mask_scenario_names,
                        private_infos=pinfo_tuple,
                        scored_indices=scored_indices,
                        config=config,
                        image_format=image_format,
                        storage_optimization=storage_optimization,
                        storage_format=effective_storage_format,
                        save_negotiations_as_folders=save_negotiations_as_folders,
                        opponent_modeling_metrics=opponent_modeling_metrics,
                        distribute_opponent_modeling_scores=distribute_opponent_modeling_scores,
                    )
                    # Add run_id for identifying completed runs
                    run_dict["run_id"] = hash_to_base64(
                        hash(
                            str(
                                serialize(
                                    run_dict,
                                    python_class_identifier=python_class_identifier,
                                )
                            )
                        )
                    )
                    runs.append(run_dict)
    if randomize_runs:
        shuffle(runs)
    if sort_runs:
        runs = sorted(runs, key=lambda x: scenario_size(x["s"]))

    # Filter out already completed runs if continuing an existing tournament
    if completed_runs:
        initial_run_count = len(runs)
        runs = [
            r
            for r in runs
            if not _is_run_completed(r, completed_runs, python_class_identifier)
        ]
        skipped_count = initial_run_count - len(runs)
        if verbosity > 0 and skipped_count > 0:
            print(
                f"[yellow]Skipping {skipped_count} already completed negotiations[/yellow]"
            )

        # If all runs are complete, load and return existing results
        if len(runs) == 0 and path:
            if verbosity > 0:
                print(
                    f"[green]All {initial_run_count} negotiations are already complete. Loading existing results...[/green]"
                )
            # Try to load the complete tournament
            try:
                tresults = SimpleTournamentResults.load(
                    Path(path),
                    memory_optimization=memory_optimization,
                    storage_optimization=storage_optimization,
                    storage_format=effective_storage_format,
                )
                if verbosity > 0:
                    print(
                        "[green]Successfully loaded existing tournament results.[/green]"
                    )
                return tresults
            except Exception as e:
                # If loading fails, reconstruct from existing results
                if verbosity > 0:
                    print(
                        f"[yellow]Could not load saved results ({e}). Reconstructing from individual negotiations...[/yellow]"
                    )
                # Continue to reconstruct results from existing_results

    # Report progress: setup complete, ready to start negotiations
    if progress_callback:
        try:
            _call_progress_callback(
                progress_callback, f"Starting {len(runs)} negotiations", 3, 4, config
            )
        except Exception:
            pass

    if verbosity > 0:
        print(
            f"Will run {len(runs)} negotiations on {len(scenarios)} scenarios between {len(competitors)} competitors",
            flush=True,
        )
    results, scores = [], []

    def process_record(record, results=results, scores=scores):
        """Process record.

        Args:
            record: Record.
            results: Results.
            scores: Scores.
        """
        if self_play and only_failures_on_self_play:
            is_self_play = len(set(record["partners"])) == 1
            if is_self_play and record["agreement"] is not None:
                return results, scores
        results.append(record)
        scores += make_scores(
            record,
            scored_indices=record.get("scored_indices"),
            raw_aggregated_metrics=raw_aggregated_metrics,
        )
        if path and save_every and i % save_every == 0:
            _save_dataframe(
                pd.DataFrame.from_records(results),
                path,
                "details",
                effective_storage_format,
            )
            _save_dataframe(
                pd.DataFrame.from_records(scores),
                path,
                "all_scores",
                effective_storage_format,
            )
        return results, scores

    def get_run_id(info) -> str:
        """Get run id.

        Args:
            info: Info.
        """
        return hash_to_base64(
            hash(str(serialize(info, python_class_identifier=python_class_identifier)))
        )

    if njobs < 0:
        for i, info in enumerate(
            track(runs, total=len(runs), description=NEGOTIATIONS_DIR_NAME)
        ):
            # Remove run_id from info before unpacking to avoid duplicate argument error
            info_copy = {k: v for k, v in info.items() if k != "run_id"}
            process_record(
                run_negotiation(
                    **info_copy,
                    before_start_callback=before_start_callback,
                    after_construction_callback=after_construction_callback,
                    after_end_callback=after_end_callback,
                    neg_start_callback=neg_start_callback,
                    neg_end_callback=neg_end_callback,
                    neg_progress_callback=neg_progress_callback,
                    run_id=get_run_id(info),
                )
            )

    else:
        timeout = external_timeout if external_timeout else float("inf")

        def _safe_max(x) -> float:
            if x is None:
                return float("inf")
            if isinstance(x, tuple):
                return x[-1]
            return x

        tparams = dict(
            time_limit=mechanism_params.get("time_limit", float("inf")),
            negotiator_time_limit=mechanism_params.get(
                "negotiator_time_limit", float("inf")
            ),
            step_time_limit=mechanism_params.get("step_time_limit", float("inf")),
            hidden_time_limit=mechanism_params.get("hidden_time_limit", float("inf")),
        ) | dict(
            time_limit=_safe_max(time_limit),
            negotiator_time_limit=_safe_max(negotiator_time_limit),
            step_time_limit=_safe_max(step_time_limit),
            hidden_time_limit=_safe_max(hidden_time_limit),
        )
        touts = [_ * 1.05 for _ in tparams.values() if _ is not None and not isinf(_)]
        timeout = min(max(touts) if touts else float("inf"), timeout)
        if isinf(timeout):
            timeout = None
        if timeout is not None and verbosity > 0:
            print(
                f"[magenta]Will use {timeout} as a timeout when receiving results[/magenta]"
            )

        futures = dict()
        n_cores = cpu_count()
        if n_cores is None:
            n_cores = 4
        cpus = min(n_cores, njobs) if njobs else cpu_count()
        kwargs_ = dict(max_workers=cpus)
        version = sys.version_info
        # Only use max_tasks_per_child if we have multiple workers to avoid
        # worker restart deadlocks with single-worker pools on Python 3.14+
        if (version.major > 3 or version.minor > 10) and cpus is not None and cpus > 1:
            kwargs_.update(max_tasks_per_child=MAX_TASKS_PER_CHILD)

        with ProcessPoolExecutor(**kwargs_) as pool:  # type: ignore
            for info in runs:
                # Remove run_id from info before unpacking to avoid duplicate argument error
                info_copy = {k: v for k, v in info.items() if k != "run_id"}
                futures[
                    pool.submit(
                        run_negotiation,
                        **info_copy,
                        before_start_callback=before_start_callback,
                        after_construction_callback=after_construction_callback,
                        after_end_callback=after_end_callback,
                        neg_start_callback=neg_start_callback,
                        neg_progress_callback=neg_progress_callback,
                        neg_end_callback=neg_end_callback,
                        run_id=info["run_id"],  # Pass the pre-computed run_id
                    )
                ] = info
            for i, f in enumerate(
                track(
                    as_completed(futures),
                    total=len(futures),
                    description=NEGOTIATIONS_DIR_NAME,
                )
            ):
                try:
                    result = f.result(timeout=timeout)
                    process_record(result)
                except TimeoutError:
                    info = futures.get(f, dict(partners=["Unknown", "Unknown"]))
                    print(
                        f"[red]Negotiation between {info['partners']} [bold]timedout[/bold] [red] after {timeout} seconds ...\n\tKilling the process",
                        end="",
                    )
                    if len(info) > 1:
                        result = failed_run_record(**info)
                        if after_end_callback:
                            try:
                                _call_after_end_callback(
                                    after_end_callback, result, config
                                )
                            except Exception as e:
                                if verbosity > 0:
                                    print(
                                        f"After end callback failed on a failed run {get_run_id(info)}: {e}"
                                    )
                        process_record(result)

                    f.cancel()
                    try:
                        if os.name == "nt":  # Check if running on Windows
                            pool._processes[f._process_ident].terminate()
                        else:
                            os.kill(
                                f._process_ident,  # type: ignore
                                signal.SIGTERM,
                            )  # Default to SIGTERM
                            time.sleep(
                                TERMINATION_WAIT_TIME
                            )  # Allow brief time for termination
                            if not pool._processes[f._process_ident].is_alive():  # type: ignore
                                os.kill(
                                    f._process_ident,  # type: ignore
                                    signal.SIGKILL,
                                )  # Forceful if needed
                        print("[yellow]SUCCEEDED[/yellow]")
                    except Exception as e:
                        print(f"[red]FAILED[/red] with exception {e}")

                except BrokenProcessPool as e:
                    if verbosity > 1:
                        print("[red]Broken Pool[/red]")
                        print(e)
                    break
                except Exception as e:
                    if verbosity > 1:
                        print("[red]Exception[/red]")
                        if verbosity > 2:
                            print(traceback.format_exc())
                        print(e)
            pool.shutdown(wait=False)
            # _stop_process_pool(pool)

    # Merge with existing results if continuing a tournament
    if existing_results:
        results = existing_results + results
        scores = existing_scores + scores
        if verbosity > 0:
            print(
                f"[green]Merged {len(existing_results)} existing results with {len(results) - len(existing_results)} new results[/green]"
            )

    tresults = SimpleTournamentResults.from_records(
        config,
        scores,
        results,
        final_score_stat=final_score,
        path=path,
        stats_aggregated_metrics=stats_aggregated_metrics,
    )
    if verbosity > 0:
        print(tresults.final_scores)
    if path:
        # Save results with storage optimization and format
        tresults.save(
            path,
            storage_optimization=storage_optimization,
            storage_format=effective_storage_format,
        )

        # Apply storage cleanup based on storage_optimization level
        if storage_optimization in ("balanced", "space"):
            # Remove results/ folder (JSON files) - data is now in details file
            results_dir = path / RESULTS_DIR_NAME
            if results_dir.exists():
                try:
                    shutil.rmtree(results_dir)
                    if verbosity > 0:
                        print(
                            f"[yellow]Removed {RESULTS_DIR_NAME}/ folder (storage_optimization={storage_optimization})[/yellow]"
                        )
                except Exception as e:
                    if verbosity > 0:
                        print(
                            f"[red]Failed to remove {RESULTS_DIR_NAME}/ folder: {e}[/red]"
                        )

        if storage_optimization == "space":
            # Also remove all_scores file - can be reconstructed from details
            # Try all possible formats
            for ext in (".csv", ".csv.gz", ".parquet"):
                scores_file = path / f"all_scores{ext}"
                if scores_file.exists():
                    try:
                        scores_file.unlink()
                        if verbosity > 0:
                            print(
                                f"[yellow]Removed all_scores{ext} (storage_optimization=space)[/yellow]"
                            )
                    except Exception as e:
                        if verbosity > 0:
                            print(f"[red]Failed to remove all_scores{ext}: {e}[/red]")

    # Create a new SimpleTournamentResults with the appropriate memory optimization
    # This allows lazy loading when memory_optimization != "speed" and path is provided
    if path and memory_optimization != "speed":
        tresults = SimpleTournamentResults(
            config=config,
            scores=tresults._scores if memory_optimization == "balanced" else None,
            details=tresults._details,
            scores_summary=tresults._scores_summary,
            final_scores=tresults._final_scores,
            path=path,
            memory_optimization=memory_optimization,
            storage_optimization=storage_optimization,
            storage_format=effective_storage_format,
            final_score_stat=final_score,
        )

    if progress_callback:
        try:
            _call_progress_callback(
                progress_callback, "Completed all negotiations", 4, 4, config
            )
        except Exception:
            pass
    return tresults


def continue_cartesian_tournament(
    path: Path | str,
    verbosity: int | None = None,
    njobs: int | None = None,
    before_start_callback: BeforeStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    neg_start_callback: NegStartCallback | None = None,
    after_construction_callback: AfterConstructionCallback | None = None,
    neg_progress_callback: NegProgressCallback | None = None,
    neg_end_callback: NegEndCallback | None = None,
    after_end_callback: AfterEndCallback | None = None,
) -> SimpleTournamentResults | None:
    """
    Continue or load a cartesian tournament from a saved path.

    This is a convenience function that:
    1. Checks if the path contains a valid tournament (config.yaml and scenarios/)
    2. If incomplete, continues the tournament by running remaining negotiations
    3. If complete, loads and returns the existing results
    4. If invalid, returns None

    Args:
        path: Directory path containing the tournament (must have config.yaml and scenarios/)
        verbosity: Optional verbosity level to override the one in config.yaml
        njobs: Optional parallelization level to override the one in config.yaml
        before_start_callback: Called before each negotiation run starts with RunInfo.
        progress_callback: Called periodically during tournament execution with progress info.
        neg_start_callback: Called when a negotiation starts with (run_id, initial_state).
        after_construction_callback: Called after negotiation is constructed with ConstructedNegInfo.
        neg_progress_callback: Called during negotiation with (run_id, current_state).
        neg_end_callback: Called when a negotiation ends with (run_id, final_state).
        after_end_callback: Called after each negotiation run completes with (RunInfo, Mechanism, results_dict).

    Returns:
        SimpleTournamentResults if tournament is valid, None otherwise

    Examples:
        ```python
        # Start a tournament
        results = cartesian_tournament(
            competitors=[MyNegotiator, TheirNegotiator],
            scenarios=scenarios,
            n_repetitions=100,
            path=Path("my_tournament/"),
        )

        # Later, continue or load it
        results = continue_cartesian_tournament(Path("my_tournament/"))
        if results is None:
            print("Invalid tournament path")
        ```

        ```python
        # Continue with different verbosity/parallelization
        results = continue_cartesian_tournament(
            Path("my_tournament/"),
            verbosity=2,
            njobs=-1,  # Serial execution for debugging
        )
        ```

        ```python
        # Continue with callbacks
        def on_neg_end(run_id, state):
            print(f"Negotiation {run_id} ended: agreement={state.agreement}")


        results = continue_cartesian_tournament(
            Path("my_tournament/"), neg_end_callback=on_neg_end
        )
        ```
    """
    path = Path(path)

    # Check if path exists
    if not path.exists() or not path.is_dir():
        return None

    # Check if config.yaml exists
    config_path = path / CONFIG_FILE_NAME
    if not config_path.exists():
        return None

    # Check if scenarios/ directory exists
    scenarios_path = path / SCENARIOS_DIR_NAME
    if not scenarios_path.exists() or not scenarios_path.is_dir():
        return None

    # Load config
    try:
        config = load(config_path)
    except Exception:
        return None

    # Validate config has required fields
    required_fields = ["n_scenarios", "competitors"]
    if not all(field in config for field in required_fields):
        return None

    # Load scenarios from scenarios/ directory
    try:
        scenario_dirs = sorted(
            [d for d in scenarios_path.iterdir() if d.is_dir()], key=lambda x: x.name
        )

        # Note: n_scenarios in config is the number of BASE scenarios (before rotation)
        # The scenarios/ directory might contain rotated versions (e.g., S0, S0-1, S0-2)
        # We only load the base scenarios (those without "-" suffix for rotation > 0)
        base_scenario_dirs = [
            d
            for d in scenario_dirs
            if "-" not in d.name or d.name.split("-")[-1] == "0"
        ]

        # If no base scenarios found, try loading all (backwards compatibility)
        if not base_scenario_dirs:
            base_scenario_dirs = scenario_dirs

        scenarios = []
        for scenario_dir in base_scenario_dirs:
            scenario = Scenario.load(scenario_dir)
            if scenario is None:
                # Failed to load a scenario
                return None
            scenarios.append(scenario)

    except Exception:
        return None

        # scenarios = []
        # for scenario_dir in scenario_dirs:
        #     scenario = Scenario.load(scenario_dir)
        #     if scenario is None:
        #         # Failed to load a scenario
        #         return None
        #     scenarios.append(scenario)

    # Extract parameters from config
    try:
        # First, try to load pre-computed results if they exist
        # This is faster and avoids potential reconstruction issues
        try:
            existing_results = SimpleTournamentResults.load(
                path,
                memory_optimization=config.get("memory_optimization", "balanced"),
                storage_optimization=config.get("storage_optimization", "balanced"),
                storage_format=config.get("storage_format"),
            )
            if verbosity and verbosity > 0:
                print("[green]Loaded existing tournament results.[/green]")
            return existing_results
        except Exception:
            # Pre-computed results not available or incomplete, continue to reconstruct
            pass

        # Get negotiator classes
        competitors = [get_class(name) for name in config["competitors"]]
        opponents = (
            [get_class(name) for name in config["opponents"]]
            if config.get("opponents")
            else []
        )

        # Override verbosity and njobs if provided
        final_verbosity = (
            verbosity if verbosity is not None else config.get("verbosity", 1)
        )
        final_njobs = njobs if njobs is not None else config.get("njobs", 0)

        # Call cartesian_tournament with path_exists="continue"
        # Use config values directly; only provide defaults for truly optional parameters
        return cartesian_tournament(
            competitors=competitors,  # type: ignore
            scenarios=scenarios,
            opponents=opponents if opponents else None,  # type: ignore
            competitor_params=config.get("competitor_params"),
            opponent_params=config.get("opponent_params"),
            competitor_names=config.get("competitor_names"),
            opponent_names=config.get("opponent_names"),
            private_infos=config.get("private_infos"),
            rotate_ufuns=config["rotate_ufuns"],
            rotate_private_infos=config["rotate_private_infos"],
            n_repetitions=config["n_repetitions"],
            path=path,
            path_exists="continue",
            njobs=final_njobs,
            mechanism_type=get_class(config["mechanism_type"]),
            mechanism_params=config.get("mechanism_params"),
            n_steps=config["n_steps"],
            time_limit=config.get("time_limit"),
            pend=config["pend"],
            pend_per_second=config["pend_per_second"],
            step_time_limit=config.get("step_time_limit"),
            negotiator_time_limit=config.get("negotiator_time_limit"),
            hidden_time_limit=config.get("hidden_time_limit"),
            external_timeout=config.get("external_timeout"),
            plot_fraction=config["plot_fraction"],
            plot_params=config.get("plot_params"),
            verbosity=final_verbosity,
            self_play=config["self_play"],
            randomize_runs=config["randomize_runs"],
            sort_runs=config["sort_runs"],
            save_every=config["save_every"],
            save_stats=config["save_stats"],
            save_scenario_figs=config["save_scenario_figs"],
            recalculate_stats=config.get("recalculate_stats", False),
            image_format=config.get("image_format", DEFAULT_IMAGE_FORMAT),
            opponent_modeling_metrics=config.get("opponent_modeling_metrics", ()),
            distribute_opponent_modeling_scores=config.get(
                "distribute_opponent_modeling_scores", True
            ),
            raw_aggregated_metrics=config.get("raw_aggregated_metrics"),
            stats_aggregated_metrics=config.get("stats_aggregated_metrics"),
            final_score=config["final_score"],
            id_reveals_type=config["id_reveals_type"],
            name_reveals_type=config["name_reveals_type"],
            raise_exceptions=config["raise_exceptions"],
            mask_scenario_names=config["mask_scenario_names"],
            only_failures_on_self_play=config.get("only_failures_on_self_play", False),
            ignore_discount=config["ignore_discount"],
            ignore_reserved=config["ignore_reserved"],
            normalize_ufuns=config.get("normalize_ufuns", True),
            reserved_value_eps=config.get("reserved_value_eps", 0.0),
            storage_optimization=config["storage_optimization"],
            memory_optimization=config["memory_optimization"],
            storage_format=config.get("storage_format"),
            save_negotiations_as_folders=config.get(
                "save_negotiations_as_folders", False
            ),
            python_class_identifier=config.get(
                "python_class_identifier", PYTHON_CLASS_IDENTIFIER
            ),
            metadata=config.get("metadata"),
            before_start_callback=before_start_callback,
            progress_callback=progress_callback,
            neg_start_callback=neg_start_callback,
            after_construction_callback=after_construction_callback,
            neg_progress_callback=neg_progress_callback,
            neg_end_callback=neg_end_callback,
            after_end_callback=after_end_callback,
        )
    except Exception as e:
        # Failed to reconstruct tournament parameters
        # Debug: print exception in verbose mode
        if verbosity and verbosity > 0:
            import traceback

            print(f"[yellow]Warning: Failed to continue tournament: {e}[/yellow]")
            traceback.print_exc()
        return None


if __name__ == "__main__":
    from random import randint, random

    from negmas.helpers.misc import intin
    from negmas.outcomes import make_issue, make_os
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction as U
    from negmas.preferences.generators import generate_utility_values
    from negmas.preferences.value_fun import TableFun
    from negmas.sao.negotiators import (
        AspirationNegotiator,
        MiCRONegotiator,
        NaiveTitForTatNegotiator,
    )

    n_scenarios, n_outcomes = 5, (10, 100)
    ufun_sets = []

    for i in range(n_scenarios):
        r = random()
        n = intin(n_outcomes, log_uniform=True)
        name = "S"
        if r < 0.3:
            n_pareto = n
            name = "DivideThePieGen"
        else:
            n_pareto = randint(min(5, n // 2), n // 2)
        if r < 0.05:
            vals = generate_utility_values(
                n_pareto, n, n_ufuns=2, pareto_first=False, pareto_generator="zero_sum"
            )
            name = "DivideThePie"
        else:
            vals = generate_utility_values(
                n_pareto,
                n,
                n_ufuns=2,
                pareto_first=False,
                pareto_generator="curve" if random() < 0.5 else "piecewise_linear",
            )

        issues = (make_issue([f"{i}_{n - 1 - i}" for i in range(n)], "portions"),)
        ufuns = tuple(
            U(
                values=(
                    TableFun(
                        {_: float(vals[i][k]) for i, _ in enumerate(issues[0].all)}
                    ),
                ),
                name=f"{uname}{i}",
                reserved_value=0.0,
                outcome_space=make_os(issues, name=f"{name}{i}"),
            )
            for k, uname in enumerate(("First", "Second"))
        )
        ufun_sets.append(ufuns)

    scenarios = [
        Scenario(
            outcome_space=ufuns[0].outcome_space,  # type: ignore We are sure this is not None
            ufuns=ufuns,
        )
        for ufuns in ufun_sets
    ]

    cartesian_tournament(
        competitors=(AspirationNegotiator, NaiveTitForTatNegotiator, MiCRONegotiator),
        scenarios=scenarios,
        n_repetitions=1,
    )
