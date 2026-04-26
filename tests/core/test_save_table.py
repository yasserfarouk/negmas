"""Tests for save_table, load_table, and safe_write_file functions in negmas.helpers.inout."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from negmas.helpers.inout import safe_write_file, load_table, save_table
from negmas import warnings as negmas_warnings


class TestSaveTableBasic:
    """Basic functionality tests for save_table."""

    def test_save_table_csv_from_list_of_dicts(self, tmp_path):
        """Test saving a list of dicts as CSV."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        path = save_table(data, tmp_path / "test.csv", storage_format="csv")
        assert path.exists()
        assert path.suffix == ".csv"
        df = pd.read_csv(path)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_save_table_parquet_from_dataframe(self, tmp_path):
        """Test saving a DataFrame as parquet."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        path = save_table(df, tmp_path / "test.parquet", storage_format="parquet")
        assert path.exists()
        assert path.suffix == ".parquet"
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == ["x", "y"]
        assert len(loaded) == 3

    def test_save_table_gzip_from_list(self, tmp_path):
        """Test saving as gzip-compressed CSV."""
        data = [{"col1": 10, "col2": 20}]
        path = save_table(data, tmp_path / "test.csv", storage_format="gzip")
        assert path.exists()
        assert str(path).endswith(".csv.gz")
        df = pd.read_csv(path, compression="gzip")
        assert len(df) == 1

    def test_save_table_with_tuples_converts_to_strings(self, tmp_path):
        """Test that tuple values are converted to strings for parquet."""
        data = [{"name": "test", "values": (1, 2, 3)}]
        path = save_table(data, tmp_path / "test.parquet", storage_format="parquet")
        assert path.exists()
        loaded = pd.read_parquet(path)
        # Tuples should be converted to string representation
        assert loaded["values"].iloc[0] == "(1, 2, 3)"

    def test_save_table_empty_dataframe(self, tmp_path):
        """Test saving an empty DataFrame."""
        df = pd.DataFrame()
        path = save_table(df, tmp_path / "empty.csv", storage_format="csv")
        assert path.exists()

    def test_save_table_with_index(self, tmp_path):
        """Test saving with index included."""
        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        path = save_table(
            df, tmp_path / "indexed.csv", storage_format="csv", index=True
        )
        loaded = pd.read_csv(path, index_col=0)
        assert list(loaded.index) == ["x", "y"]


class TestSafeWriteFile:
    """Tests for the safe_write_file helper function."""

    def testsafe_write_file_basic(self, tmp_path):
        """Test basic file write succeeds."""
        path = tmp_path / "test.txt"

        def write_func(p: Path) -> None:
            p.write_text("hello world")

        result = safe_write_file(write_func, path)
        assert result == path
        assert path.exists()
        assert path.read_text() == "hello world"

    def testsafe_write_file_cleans_up_temp_files(self, tmp_path):
        """Test that temporary files are cleaned up on success."""
        path = tmp_path / "test.txt"

        def write_func(p: Path) -> None:
            p.write_text("test content")

        safe_write_file(write_func, path)
        # Check no .tmp files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def testsafe_write_file_retry_on_permission_error(self, tmp_path):
        """Test that retries happen on PermissionError."""
        path = tmp_path / "test.txt"
        call_count = 0

        def write_func(p: Path) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("Simulated permission error")
            p.write_text("success after retries")

        result = safe_write_file(write_func, path, base_delay=0.01)

        assert call_count == 3  # Two failures + one success
        assert result == path
        assert path.read_text() == "success after retries"

    def testsafe_write_file_fallback(self, tmp_path):
        """Test fallback when primary write consistently fails."""
        path = tmp_path / "test.parquet"
        fallback_path = tmp_path / "test.csv"

        def write_func(p: Path) -> None:
            raise PermissionError("Persistent permission error")

        def fallback_func(p: Path) -> None:
            p.write_text("fallback content")

        with pytest.warns(negmas_warnings.NegmasIOWarning, match="Falling back to"):
            result = safe_write_file(
                write_func,
                path,
                max_retries=2,
                base_delay=0.01,
                fallback_func=fallback_func,
                fallback_path=fallback_path,
            )

        assert result == fallback_path
        assert fallback_path.exists()
        assert fallback_path.read_text() == "fallback content"
        assert not path.exists()

    def testsafe_write_file_no_fallback_raises(self, tmp_path):
        """Test that error is raised when no fallback is provided."""
        path = tmp_path / "test.txt"

        def write_func(p: Path) -> None:
            raise PermissionError("Persistent permission error")

        with pytest.raises(PermissionError):
            safe_write_file(write_func, path, max_retries=2, base_delay=0.01)

    def testsafe_write_file_exponential_backoff_with_jitter(self, tmp_path):
        """Test that exponential backoff with jitter is applied between retries."""
        path = tmp_path / "test.txt"
        sleep_times: list[float] = []
        call_count = 0

        def write_func(p: Path) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("Simulated error")
            p.write_text("success")

        def mock_sleep(seconds: float) -> None:
            sleep_times.append(seconds)
            # Don't actually sleep in tests

        with patch("time.sleep", mock_sleep):
            safe_write_file(write_func, path, max_retries=3, base_delay=0.1)

        # Should have slept twice (after 1st and 2nd failure)
        assert len(sleep_times) == 2
        # Exponential backoff with jitter: base * 2^attempt + jitter (0 to 50% of base delay)
        # First retry: 0.1 + jitter (0 to 0.05), so between 0.1 and 0.15
        # Second retry: 0.2 + jitter (0 to 0.1), so between 0.2 and 0.3
        assert 0.1 <= sleep_times[0] <= 0.15
        assert 0.2 <= sleep_times[1] <= 0.3

    def testsafe_write_file_atomic_rename(self, tmp_path):
        """Test that os.replace is used for atomic rename."""
        path = tmp_path / "test.txt"

        def write_func(p: Path) -> None:
            p.write_text("atomic test")

        with patch("os.replace", wraps=os.replace) as mock_replace:
            safe_write_file(write_func, path)
            mock_replace.assert_called_once()

    def testsafe_write_file_cleans_up_on_failure(self, tmp_path):
        """Test that temp files are cleaned up even when all retries fail."""
        path = tmp_path / "test.txt"

        def write_func(p: Path) -> None:
            # Write something to the temp file before failing
            p.write_text("partial")
            raise PermissionError("Simulated error")

        with pytest.raises(PermissionError):
            safe_write_file(write_func, path, max_retries=2, base_delay=0.01)

        # Verify no temp files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestSaveTableRobustness:
    """Integration tests for save_table with robustness features."""

    def test_save_table_csv_with_transient_error(self, tmp_path):
        """Test that save_table handles transient CSV write errors."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        call_count = 0
        original_to_csv = pd.DataFrame.to_csv

        def mock_to_csv(self, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("Transient error")
            return original_to_csv(self, path, **kwargs)

        with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
            path = save_table(df, tmp_path / "test.csv", storage_format="csv")

        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_table_parquet_with_transient_error(self, tmp_path):
        """Test that save_table handles transient parquet errors."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        call_count = 0
        original_to_parquet = pd.DataFrame.to_parquet

        def mock_to_parquet(self, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("Transient error")
            return original_to_parquet(self, path, **kwargs)

        with patch.object(pd.DataFrame, "to_parquet", mock_to_parquet):
            path = save_table(df, tmp_path / "test.parquet", storage_format="parquet")

        assert path.exists()
        assert path.suffix == ".parquet"

    def test_save_table_parquet_fallback_to_csv(self, tmp_path):
        """Test end-to-end CSV fallback through save_table for parquet."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        def mock_to_parquet(self, path, **kwargs):
            raise PermissionError("Persistent error")

        with patch.object(pd.DataFrame, "to_parquet", mock_to_parquet):
            with pytest.warns(negmas_warnings.NegmasIOWarning):
                path = save_table(
                    df, tmp_path / "test.parquet", storage_format="parquet"
                )

        # Should have fallen back to CSV
        assert path.suffix == ".csv"
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 3

    def test_save_table_gzip_with_transient_error(self, tmp_path):
        """Test that save_table handles transient gzip write errors."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        call_count = 0
        original_to_csv = pd.DataFrame.to_csv

        def mock_to_csv(self, path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("Transient error")
            return original_to_csv(self, path, **kwargs)

        with patch.object(pd.DataFrame, "to_csv", mock_to_csv):
            path = save_table(df, tmp_path / "test.csv", storage_format="gzip")

        assert path.exists()
        assert str(path).endswith(".gz")


class TestLoadTable:
    """Tests for load_table function."""

    def test_load_table_csv(self, tmp_path):
        """Test loading a CSV file."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loaded = load_table(path)
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["a", "b"]

    def test_load_table_parquet(self, tmp_path):
        """Test loading a parquet file."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_table(path)
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["a", "b"]

    def test_load_table_as_list_of_dicts(self, tmp_path):
        """Test loading as list of dicts."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loaded = load_table(path, as_dataframe=False)
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["a"] == 1


class TestRoundTrip:
    """Round-trip tests for save_table -> load_table."""

    @pytest.mark.parametrize("storage_format", ["csv", "gzip", "parquet"])
    def test_round_trip_all_formats(self, tmp_path, storage_format):
        """Test round-trip for all storage formats."""
        data = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        path = save_table(data, tmp_path / "test.csv", storage_format=storage_format)
        loaded = load_table(path)
        assert len(loaded) == 2
        assert list(loaded["x"]) == [1, 2]
        assert list(loaded["y"]) == ["a", "b"]

    def test_round_trip_with_special_characters(self, tmp_path):
        """Test round-trip with special characters in data."""
        data = [{"text": "hello, world"}, {"text": 'quote "test"'}]
        path = save_table(data, tmp_path / "special.csv", storage_format="csv")
        loaded = load_table(path)
        assert loaded["text"].iloc[0] == "hello, world"
        assert loaded["text"].iloc[1] == 'quote "test"'

    def test_round_trip_with_none_values(self, tmp_path):
        """Test round-trip with None values."""
        data = [{"a": 1, "b": None}, {"a": None, "b": 2}]
        path = save_table(data, tmp_path / "nulls.parquet", storage_format="parquet")
        loaded = load_table(path)
        assert pd.isna(loaded["b"].iloc[0])
        assert pd.isna(loaded["a"].iloc[1])


class TestEdgeCases:
    """Edge case tests."""

    def test_save_table_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.csv"
        data = [{"x": 1}]
        path = save_table(data, nested_path, storage_format="csv")
        assert path.exists()

    def test_save_table_overwrites_existing_file(self, tmp_path):
        """Test that existing files are overwritten."""
        path = tmp_path / "test.csv"
        save_table([{"x": 1}], path, storage_format="csv")
        save_table([{"x": 2, "y": 3}], path, storage_format="csv")
        loaded = load_table(path)
        assert list(loaded.columns) == ["x", "y"]
        assert loaded["x"].iloc[0] == 2

    def test_save_table_with_list_of_tuples(self, tmp_path):
        """Test saving list of tuples with column names."""
        data = [(1, "a"), (2, "b")]
        path = save_table(
            data,
            tmp_path / "tuples.csv",
            columns=["num", "letter"],
            storage_format="csv",
        )
        loaded = load_table(path)
        assert list(loaded.columns) == ["num", "letter"]
        assert len(loaded) == 2

    def test_save_table_invalid_format_raises(self, tmp_path):
        """Test that invalid storage format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage_format"):
            save_table([{"x": 1}], tmp_path / "test.txt", storage_format="invalid")  # type: ignore


class TestLoadTableAlternatives:
    """Tests for load_table's alternative format fallback behavior."""

    def test_load_table_finds_csv_when_parquet_requested(self, tmp_path):
        """Test that load_table finds CSV file when parquet is requested but doesn't exist."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        csv_path = tmp_path / "data.csv"
        save_table(data, csv_path, storage_format="csv")

        # Request parquet but CSV exists
        with pytest.warns(
            negmas_warnings.NegmasIOWarning, match="Loading from alternative"
        ):
            loaded = load_table(tmp_path / "data.parquet")

        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 2

    def test_load_table_finds_parquet_when_csv_requested(self, tmp_path):
        """Test that load_table finds parquet file when CSV is requested but doesn't exist."""
        data = [{"x": 10, "y": 20}]
        parquet_path = tmp_path / "data.parquet"
        save_table(data, parquet_path, storage_format="parquet")

        # Request CSV but parquet exists
        with pytest.warns(
            negmas_warnings.NegmasIOWarning, match="Loading from alternative"
        ):
            loaded = load_table(tmp_path / "data.csv")

        assert list(loaded.columns) == ["x", "y"]
        assert len(loaded) == 1

    def test_load_table_finds_gzip_when_csv_requested(self, tmp_path):
        """Test that load_table finds gzip file when CSV is requested but doesn't exist."""
        data = [{"col": "value"}]
        gzip_path = tmp_path / "data.csv"
        save_table(data, gzip_path, storage_format="gzip")
        assert (tmp_path / "data.csv.gz").exists()

        # Request CSV but gzip exists (save_table creates .csv.gz)
        with pytest.warns(
            negmas_warnings.NegmasIOWarning, match="Loading from alternative"
        ):
            loaded = load_table(tmp_path / "data.csv")

        assert list(loaded.columns) == ["col"]
        assert len(loaded) == 1

    def test_load_table_finds_csv_when_gzip_requested(self, tmp_path):
        """Test that load_table finds CSV file when gzip is requested but doesn't exist."""
        data = [{"a": 1}]
        csv_path = tmp_path / "data.csv"
        save_table(data, csv_path, storage_format="csv")

        # Request gzip but CSV exists
        with pytest.warns(
            negmas_warnings.NegmasIOWarning, match="Loading from alternative"
        ):
            loaded = load_table(tmp_path / "data.csv.gz")

        assert list(loaded.columns) == ["a"]
        assert len(loaded) == 1

    def test_load_table_no_alternative_raises_file_not_found(self, tmp_path):
        """Test that load_table raises FileNotFoundError when no alternatives exist."""
        with pytest.raises(FileNotFoundError):
            load_table(tmp_path / "nonexistent.parquet")

    def test_load_table_try_alternatives_false_does_not_search(self, tmp_path):
        """Test that try_alternatives=False disables alternative format search."""
        data = [{"a": 1}]
        csv_path = tmp_path / "data.csv"
        save_table(data, csv_path, storage_format="csv")

        # Request parquet with try_alternatives=False - should not find CSV
        with pytest.raises(FileNotFoundError):
            load_table(tmp_path / "data.parquet", try_alternatives=False)

    def test_load_table_prefers_existing_file_over_alternatives(self, tmp_path):
        """Test that load_table uses the requested file if it exists, not alternatives."""
        # Create both CSV and parquet with different data
        csv_data = [{"source": "csv"}]
        parquet_data = [{"source": "parquet"}]

        save_table(csv_data, tmp_path / "data.csv", storage_format="csv")
        save_table(parquet_data, tmp_path / "data.parquet", storage_format="parquet")

        # Request CSV - should load CSV, not parquet
        loaded = load_table(tmp_path / "data.csv")
        assert loaded.iloc[0]["source"] == "csv"

        # Request parquet - should load parquet, not CSV
        loaded = load_table(tmp_path / "data.parquet")
        assert loaded.iloc[0]["source"] == "parquet"

    def test_load_table_alternative_with_as_dataframe_false(self, tmp_path):
        """Test that alternative format loading works with as_dataframe=False."""
        data = [{"x": 1, "y": 2}]
        save_table(data, tmp_path / "data.csv", storage_format="csv")

        # Request parquet as dict, but CSV exists
        with pytest.warns(negmas_warnings.NegmasIOWarning):
            loaded = load_table(tmp_path / "data.parquet", as_dataframe=False)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["x"] == 1
        assert loaded[0]["y"] == 2
