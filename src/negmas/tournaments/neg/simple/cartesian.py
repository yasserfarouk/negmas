"""
Negotiation tournaments module.
"""

from __future__ import annotations

import ast
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
from math import exp, isinf, log
from os import cpu_count
from pathlib import Path
from random import randint, random, shuffle
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Sequence

import pandas as pd
from attr import asdict, define
from rich import print
from rich.progress import track

from negmas.common import TraceElement, TRACE_ELEMENT_MEMBERS
from negmas.helpers import unique_name
from negmas.helpers.inout import dump, has_needed_files, load
from negmas.helpers.strings import humanize_time, shortest_unique_names
from negmas.helpers.types import get_class, get_full_type_name
from negmas.inout import Scenario, scenario_size
from negmas.mechanisms import Mechanism, Traceable
from negmas.negotiators import Negotiator
from negmas.plots.util import plot_offline_run
from negmas.preferences.discounted import DiscountedUtilityFunction
from negmas.preferences.ops import (
    ScenarioStats,
    calc_outcome_distances,
    calc_outcome_optimality,
    estimate_max_dist,
)
from negmas.sao.common import SAOState
from negmas.sao.mechanism import SAOMechanism
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, serialize, to_flat_dict

OptimizationLevel = Literal["speed", "time", "none", "balanced", "space", "max"]
StorageFormat = Literal["csv", "gzip", "parquet"]


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
    "SimpleTournamentResults",
    "combine_tournaments",
    "RunInfo",
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

EXTENSION = ".csv"
ALL_SCORES_FILE_NAME = f"all_scores{EXTENSION}"
ALL_RESULTS_FILE_NAME = f"details{EXTENSION}"
TYPE_SCORES_FILE_NAME = f"type_scores{EXTENSION}"
FINAL_SCORES_FILE_NAME = f"scores{EXTENSION}"
NEGOTIATOR_BEHAVIOR_DIR_NAME = "negotiator_behavior"
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
)

TOURNAMENT_DIRS = [
    SCENARIOS_DIR_NAME,
    NEGOTIATOR_BEHAVIOR_DIR_NAME,
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
    """

    s: Scenario
    partners: tuple[type[Negotiator]]
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
    run_id: int | str | None = None
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


@define
class ConstructedNegInfo:
    run_id: int | str | None
    mechanism: Mechanism
    failures: dict
    scenario: Scenario
    real_scenario_name: str | None


BeforeStartCallback = Callable[[RunInfo], None]
AfterConstructionCallback = Callable[[ConstructedNegInfo], None]
AfterEndCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[[str, int, int], None]  # (message, current, total)


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
        scores: list[dict[str, Any]] | pd.DataFrame | None = None,
        results: list[dict[str, Any]] | pd.DataFrame | None = None,
        type_scores: pd.DataFrame | None = None,
        final_scores: pd.DataFrame | None = None,
        final_score_stat: tuple[str, str] = ("advantage", "mean"),
        path: Path | None = None,
    ) -> "SimpleTournamentResults":
        """Creates SimpleTournamentResults from records of results

        Args:
            scores: The scores of negotiators in all negotiations (If not given, `results` can be used to calculate it).
            results: Results of all negotiations (If not given, the resulting SimpleTournamentResults object will lack details)
            type_scores: Optionally, type-scores. If not given, it will be calculated from scores
            final_scores: Optionally, final scores. If not given, `final_scoer_stat` will be used to calculate them
            final_score_stat: A tuple of the measure used and the statistic applied to it for calculating final score. See `cartesian_tournament` for more details
            path: The path in which the data for this tournament is stored.

        Raises:
            ValueError: If no scores or results are given

        Returns:
            A new SimpleTournamentResults with the given data
        """
        if scores is None and results is None:
            raise ValueError("Cannot pass both scoers and results as None")
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
                    type_scores = (
                        scores_df.loc[:, cols]
                        .groupby("strategy")
                        .describe()
                        .sort_values(final_score_stat, ascending=False)
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
        return cls.from_records(
            scores=pd.concat(scores, ignore_index=True),
            results=pd.concat(results, ignore_index=True),
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
    partners: tuple[type[Negotiator]],
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
    private_infos: tuple[dict[str, Any] | None] | None = None,
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
                name += str(hash(str(p)))
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
        if id_reveals_type:
            negotiator.id = f"{name}@{L_}"
        else:
            negotiator.id = unique_name("n", add_time=False, sep="")
        if name_reveals_type:
            negotiator.name = f"{name}@{L_}"
        else:
            negotiator.name = unique_name("n", add_time=False, sep="")
        complete_names.append(name)
        m.add(negotiator, ufun=copy.deepcopy(u))

    return (m, failures, s, real_scenario_name)


def _make_failure_record(
    state: SAOState,
    s: Scenario,
    param_dump,
    partner_names,
    run_id,
    execution_time,
    real_scenario_name,
    stats,
    mechanism_type,
    mechanism_params,
    partners,
):
    """Create a record for a failed negotiation with null/error values."""
    if not partner_names:
        partner_names = [get_full_type_name(_) for _ in partners]
    if all(_ is None for _ in param_dump):
        param_dump = None
    agreement_utils = tuple(u(state.agreement) for u in s.ufuns)
    reservations = tuple(u.reserved_value for u in s.ufuns)
    max_utils = [_.max() for _ in s.ufuns]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
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

    return run_record


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
    max_utils = [_.max() for _ in s.ufuns]
    run_record = asdict(state)
    run_record["utilities"] = agreement_utils
    run_record["max_utils"] = max_utils
    run_record["reserved_values"] = reservations
    run_record["partners"] = partner_names
    run_record["params"] = param_dump
    run_record["run_id"] = run_id
    run_record["execution_time"] = execution_time
    run_record["negotiator_names"] = m.negotiator_names
    run_record["negotiator_ids"] = m.negotiator_ids
    run_record["negotiator_types"] = [_.type_name for _ in m.negotiators]
    run_record["negotiator_times"] = [m.negotiator_times[_] for _ in m.negotiator_ids]
    run_record["n_steps"] = m.nmi.n_steps
    run_record["time_limit"] = m.nmi.time_limit
    run_record["pend"] = m.nmi.pend
    run_record["pend_per_second"] = m.nmi.pend_per_second
    run_record["step_time_limit"] = m.nmi.step_time_limit
    run_record["negotiator_time_limit"] = m.nmi.negotiator_time_limit
    run_record["annotation"] = m.nmi.annotation
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

    if m.nmi.annotation:
        run_record.update(m.nmi.annotation)
    if stats is not None:
        dists = calc_outcome_distances(agreement_utils, stats)
        run_record.update(
            to_flat_dict(
                calc_outcome_optimality(dists, stats, estimate_max_dist(s.ufuns))
            )
        )

    return run_record


def _save_record(
    run_record,
    m: Mechanism,
    partner_names,
    real_scenario_name,
    rep,
    run_id,
    path,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
):
    """Save negotiation record to disk as JSON and/or pickle."""
    file_name = f"{real_scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"
    if not path:
        return

    def save_as_df(data: list[TraceElement] | list[tuple], names, file_name):
        """Save as df.

        Args:
            data: Data.
            names: Names.
            file_name: File name.
        """
        pd.DataFrame(data=data, columns=names).to_csv(file_name, index=False)

    for k, v in m._negotiator_logs.items():
        if not v:
            continue
        if k in m.negotiator_ids:
            k = m._negotiator_map[k].name
        neg_name = path / "logs" / file_name / f"{k}.csv"
        if neg_name.exists():
            print(f"[yellow]{neg_name} already found[/yellow]")
            neg_name = (
                path
                / "logs"
                / file_name
                / unique_name("{k}.csv", sep="", add_time=True)
            )
        neg_name.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(v).to_csv(neg_name, index=True, index_label="index")

    full_name = path / NEGOTIATIONS_DIR_NAME / f"{file_name}.csv"
    if full_name.exists():
        print(f"[yellow]{full_name} already found[/yellow]")
        full_name = (
            path / unique_name(NEGOTIATIONS_DIR_NAME, sep="") / f"{file_name}.csv"
        )

    if isinstance(m, Traceable):
        assert hasattr(m, "full_trace")
        save_as_df(m.full_trace, TRACE_ELEMENT_MEMBERS, full_name)  # type: ignore
        for i, negotiator in enumerate(m.negotiators):
            neg_name = (
                path
                / NEGOTIATOR_BEHAVIOR_DIR_NAME
                / file_name
                / f"{negotiator.name}_at{i}.csv"
            )
            if neg_name.exists():
                print(f"[yellow]{neg_name} already found[/yellow]")
                neg_name = (
                    path
                    / NEGOTIATOR_BEHAVIOR_DIR_NAME
                    / file_name
                    / unique_name(f"{negotiator.name}_at{i}.csv", sep="")
                )
            neg_name.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(m, Traceable):
                save_as_df(
                    m.negotiator_full_trace(negotiator.id),
                    (
                        "time",
                        "relative_time",
                        "step",
                        "offer",
                        "response",
                        "text",
                        "data",
                    ),
                    neg_name,
                )  # type: ignore
    else:
        pd.DataFrame.from_records(
            serialize(m.history, python_class_identifier=python_class_identifier)
        ).to_csv(full_name, index=False)
    full_name = path / RESULTS_DIR_NAME / f"{file_name}.json"
    if full_name.exists():
        print(f"[yellow]{full_name} already found[/yellow]")
        full_name = path / RESULTS_DIR_NAME / unique_name(f"{file_name}.json", sep="")
    dump(run_record, full_name)


def _plot_run(
    m, partner_names, real_scenario_name, rep, run_id, path, plot, plot_params
):
    """Save a plot of the negotiation mechanism to disk."""
    file_name = f"{real_scenario_name}_{'_'.join(partner_names)}_{rep}_{run_id}"
    if not path or not plot:
        return
    if plot_params is None:
        plot_params = dict()
    plot_params["save_fig"] = (True,)
    full_name = path / PLOTS_DIR_NAME / f"{file_name}.png"
    m.plot(path=path, fig_name=full_name, show=False, **plot_params)


def run_negotiation(
    s: Scenario,
    partners: tuple[type[Negotiator]],
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
    run_id: int | str | None = None,
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
        after_end_callback: Optional callback invoked after negotiation ends. Receives the record dictionary.

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
    if after_construction_callback:
        try:
            after_construction_callback(
                ConstructedNegInfo(run_id, m, failures, s, real_scenario_name)
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
            state = m.run()
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
    )
    if after_end_callback:
        try:
            after_end_callback(run_record)
        except Exception as e:
            if verbosity > 0:
                print(f"After end callback failed for {run_id}: {e}")
    _save_record(run_record, m, partner_names, real_scenario_name, rep, run_id, path)
    _plot_run(
        m, partner_names, real_scenario_name, rep, run_id, path, plot, plot_params
    )
    return run_record


def failed_run_record(
    s: Scenario,
    partners: tuple[type[Negotiator]],
    timeout: float,
    partner_names: tuple[str] | None = None,
    partner_params: tuple[dict[str, Any]] | None = None,
    error: str | None = None,
    rep: int = 0,
    path: Path | None = None,
    mechanism_type: type[Mechanism] = SAOMechanism,
    mechanism_params: dict[str, Any] | None = None,
    full_names: bool = True,
    run_id: int | str | None = None,
    annotation: dict[str, Any] | None = None,
    private_infos: tuple[dict[str, Any] | None] | None = None,
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    mask_scenario_name: bool = True,
    ignore_exceptions: bool = False,
    stats: ScenarioStats | None = None,
    scored_indices: list[int] | None = None,
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
            mechanism_params=mechanism_params,
            partners=partners,
        )
    _save_record(run_record, m, partner_names, real_scenario_name, rep, run_id, path)
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
    record: dict[str, Any], scored_indices: list[int] | None = None
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
        - Plus any optional columns and optimality metrics (if save_stats was True)
    """
    utils, partners = record["utilities"], record["partners"]
    reserved_values = record["reserved_values"]
    negids = record["negotiator_ids"]
    max_utils, times = (
        record["max_utils"],
        record.get("negotiator_times", [None] * len(utils)),
    )
    has_error = record["has_error"]
    erred_negotiator = record["erred_negotiator"]
    error_details = record["error_details"]
    mech_error = has_error and not erred_negotiator
    scores = []
    for i, (u, r, a, m, t, nid) in enumerate(
        zip(utils, reserved_values, partners, max_utils, times, negids)
    ):
        # Only score negotiators at specified indices
        if scored_indices is not None and i not in scored_indices:
            continue
        n_p = len(partners)
        bilateral = n_p == 2
        basic = dict(
            strategy=a,
            utility=u,
            reserved_value=r,
            advantage=(u - r) / (m - r) if m != r else 0.0,
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
        scores.append(basic)
    return scores


def cartesian_tournament(
    competitors: list[type[Negotiator] | str] | tuple[type[Negotiator] | str, ...],
    scenarios: list[Scenario] | tuple[Scenario, ...],
    opponents: list[type[Negotiator] | str]
    | tuple[type[Negotiator] | str, ...] = tuple(),
    opponent_params: Sequence[dict | None] | None = None,
    private_infos: list[None | tuple[dict, ...]] | None = None,
    competitor_params: Sequence[dict | None] | None = None,
    rotate_ufuns: bool = True,
    rotate_private_infos: bool = True,
    n_repetitions: int = 1,
    path: Path | None = None,
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
    final_score: tuple[str, str] = ("advantage", "mean"),
    id_reveals_type: bool = False,
    name_reveals_type: bool = True,
    shorten_names: bool = True,
    raise_exceptions: bool = True,
    mask_scenario_names: bool = True,
    only_failures_on_self_play: bool = False,
    ignore_discount: bool = False,
    ignore_reserved: bool = False,
    storage_optimization: OptimizationLevel = "space",
    memory_optimization: OptimizationLevel = "balanced",
    storage_format: StorageFormat | None = None,
    python_class_identifier=PYTHON_CLASS_IDENTIFIER,
    before_start_callback: BeforeStartCallback | None = None,
    after_construction_callback: AfterConstructionCallback | None = None,
    after_end_callback: AfterEndCallback | None = None,
    progress_callback: ProgressCallback | None = None,
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
        private_infos: Private information passed to negotiators via their `private_info` attribute.
                      Must be a list of tuples, one tuple per scenario.
        competitor_params: Parameters for initializing competitors (one dict per competitor type).
        rotate_ufuns: If True, utility functions are rotated across negotiator positions.
                     For bilateral negotiations, this creates scenarios with reversed preferences.
                     Not recommended when using explicit opponents as roles become ambiguous.
        rotate_private_infos: If True and rotate_ufuns is True, rotate private information with ufuns.
        n_repetitions: Number of times to repeat each scenario/partner combination.
        path: Directory path to save tournament results. If None, results are not saved to disk.
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
        save_scenario_figs: If True, save PNG visualizations of scenarios in utility space.
        final_score: Tuple of (metric, statistic) for ranking. Metric can be 'advantage', 'utility',
                    'partner_welfare', 'welfare', or any calculated statistic. Statistic can be
                    'mean', 'median', 'min', 'max', or 'std'. Default: ('advantage', 'mean').
        id_reveals_type: If True, negotiator IDs reveal their type (for analysis).
        name_reveals_type: If True, negotiator names reveal their type.
        shorten_names: If True, use shortened class names in results.
        raise_exceptions: If True, exceptions from negotiators/mechanisms stop the tournament.
                         If False, exceptions are logged but tournament continues.
        mask_scenario_names: If True, mask scenario names from negotiators.
        only_failures_on_self_play: If True, only record self-play runs that fail to reach agreement.
        ignore_discount: If True, ignore discounting in utility functions (use base ufun).
        ignore_reserved: If True, ignore reserved values in utility functions.
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
    """
    if mechanism_params is None:
        mechanism_params = dict()
    mechanism_params["ignore_negotiator_exceptions"] = not raise_exceptions

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

    # Report progress: competitors loaded
    if progress_callback:
        try:
            progress_callback(f"Loaded {len(competitors)} competitors", 1, 3)
        except Exception:
            pass

    # Determine if we're in explicit opponent mode
    explicit_opponents = len(opponents) > 0

    runs = []
    scenarios_path = path if path is None else Path(path) / SCENARIOS_DIR_NAME
    if scenarios_path is not None:
        scenarios_path.mkdir(exist_ok=True, parents=True)
    stats = None

    def shorten(name):
        #        for s in ("Negotiator", "Agent"):
        #            x = name.replace(s, "")
        #            if not x:
        #                return name
        #            name = x
        """Shorten.

        Args:
            name: Name.
        """
        return name

    if shorten_names:
        competitor_names = [
            shorten(_)
            for _ in shortest_unique_names([get_full_type_name(_) for _ in competitors])
        ]
        opponent_names = (
            [
                shorten(_)
                for _ in shortest_unique_names(
                    [get_full_type_name(_) for _ in opponents]
                )
            ]
            if opponents
            else []
        )
    else:
        competitor_names = [get_full_type_name(_) for _ in competitors]
        opponent_names = [get_full_type_name(_) for _ in opponents] if opponents else []
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
                progress_callback(
                    f"Processing scenario {scenario_idx + 1}/{n_scenarios}", 2, 3
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
                    result = result.ufun
            # Clear reserved value if requested
            if ignore_reserved and hasattr(result, "reserved_value"):
                result = copy.deepcopy(result)
                result.reserved_value = float("-inf")
            return result

        # Process ufuns before deepcopy
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

        original_name = s.outcome_space.name
        # original_ufun_names = [_.name for _ in s.ufuns]
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
                    tuple(ufuns),
                )
            else:
                scenario = s
            this_path = None
            # Calculate stats if needed (before saving to ensure stats are available)
            if save_stats:
                scenario.calc_stats()

            if scenarios_path:
                this_path = scenarios_path / str(scenario.outcome_space.name)
                # Exclude pareto frontier when optimizing for space
                include_pareto = storage_optimization != "space"
                # Use dumpas to save scenario with stats in _stats.yaml (not stats.json)
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
                        ufuns=s.ufuns,  # type: ignore
                        agreement=None,
                        timedout=False,
                        broken=False,
                        has_error=False,
                        names=["First", "Second"],
                        save_fig=True,
                        path=str(this_path),
                        fig_name="fig.png",
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
                params_path = (
                    scenarios_path
                    / str(scenario.outcome_space.name)
                    / MECHANISM_FILE_NAME
                )
                pdict = dict(type=get_full_type_name(mechanism_type)) | mparams
                dump(pdict, params_path)
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

                runs += [
                    dict(
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
                        stats=stats,
                        id_reveals_type=id_reveals_type,
                        name_reveals_type=name_reveals_type,
                        plot_params=plot_params,
                        mask_scenario_name=mask_scenario_names,
                        private_infos=pinfo_tuple,
                        scored_indices=scored_indices,
                    )
                    for i in range(n_repetitions)
                ]
    if randomize_runs:
        shuffle(runs)
    if sort_runs:
        runs = sorted(runs, key=lambda x: scenario_size(x["s"]))

    # Report progress: setup complete, ready to start negotiations
    if progress_callback:
        try:
            progress_callback(f"Starting {len(runs)} negotiations", 3, 3)
        except Exception:
            pass

    if verbosity > 0:
        print(
            f"Will run {len(runs)} negotiations on {len(scenarios)} scenarios between {len(competitors)} competitors",
            flush=True,
        )
    results, scores = [], []
    results_path = path if not path else path / ALL_RESULTS_FILE_NAME
    scores_path = path if not path else path / ALL_SCORES_FILE_NAME

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
        scores += make_scores(record, scored_indices=record.get("scored_indices"))
        if results_path and save_every and i % save_every == 0:
            pd.DataFrame.from_records(results).to_csv(results_path, index_label="index")
            pd.DataFrame.from_records(scores).to_csv(scores_path, index_label="index")
        return results, scores

    def get_run_id(info):
        """Get run id.

        Args:
            info: Info.
        """
        return hash(
            str(serialize(info, python_class_identifier=python_class_identifier))
        )

    if njobs < 0:
        for i, info in enumerate(
            track(runs, total=len(runs), description=NEGOTIATIONS_DIR_NAME)
        ):
            process_record(
                run_negotiation(
                    **info,
                    before_start_callback=before_start_callback,
                    after_construction_callback=after_construction_callback,
                    after_end_callback=after_end_callback,
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
        if version.major > 3 or version.minor > 10:
            kwargs_.update(max_tasks_per_child=MAX_TASKS_PER_CHILD)

        with ProcessPoolExecutor(**kwargs_) as pool:  # type: ignore
            for info in runs:
                futures[
                    pool.submit(
                        run_negotiation,
                        **info,
                        before_start_callback=before_start_callback,
                        after_construction_callback=after_construction_callback,
                        after_end_callback=after_end_callback,
                        run_id=get_run_id(info),
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
                                after_end_callback(result)
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

    tresults = SimpleTournamentResults.from_records(
        scores, results, final_score_stat=final_score, path=path
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

    return tresults


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
