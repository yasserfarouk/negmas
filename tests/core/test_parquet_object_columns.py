"""Regression tests for parquet saving of DataFrames that contain non-native
Python objects in object-dtype columns.

pyarrow cannot store arbitrary objects (agents, ufuns, outcome spaces, dicts
holding such objects, ...). Both parquet writers -- ``save_table`` (used by the
world/Tournament runner) and ``_save_dataframe`` /
``_convert_complex_columns_to_json`` (used by the Cartesian runner) -- must
stringify such values instead of raising ``ArrowInvalid``. These tests would
fail (raise) before the sanitizers were hardened to handle raw objects and
mixed columns, not just list/dict/tuple.
"""

from __future__ import annotations

import pandas as pd

from negmas.helpers.inout import save_table
from negmas.tournaments.neg.simple.cartesian import _convert_complex_columns_to_json


class _Agentish:
    """Stands in for an Agent/Negotiator object stored in a column by mistake."""

    def __init__(self, id_):
        self.id = id_

    def __str__(self):
        return f"Agentish({self.id})"


def _object_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scalar": [1, 2, 3],
            "text": ["a", "b", None],
            "obj": [_Agentish("x"), _Agentish("y"), None],  # raw objects
            "complex": [[1, 2], {"k": 1}, (3, 4)],  # list/dict/tuple
            "nested_obj": [
                {"caller": _Agentish("z")},
                {"caller": None},
                {},
            ],  # object inside dict
            "mixed": [5, _Agentish("late"), "str"],  # scalar first, object later
        }
    )


def test_convert_complex_columns_handles_raw_objects():
    df = _object_frame()
    out, _ = _convert_complex_columns_to_json(df)
    # the raw-object column is stringified, not left as objects
    assert all(isinstance(v, str) for v in out["obj"].dropna())
    # the mixed column (scalar then object) is fully handled value-by-value
    assert all(
        v is None or isinstance(v, (str, int, float, bool)) for v in out["mixed"]
    )
    # objects nested inside a dict don't blow up json serialization
    assert all(isinstance(v, str) for v in out["nested_obj"])


def test_save_table_parquet_with_object_columns_roundtrips(tmp_path):
    df = _object_frame()
    path = tmp_path / "t.parquet"
    # Must not raise pyarrow ArrowInvalid.
    save_table(df, path, storage_format="parquet", index=False)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3
    assert "obj" in loaded.columns
    # scalar columns survive as numbers
    assert list(loaded["scalar"]) == [1, 2, 3]


def test_save_dataframe_parquet_with_object_columns_roundtrips(tmp_path):
    from negmas.tournaments.neg.simple.cartesian import _load_dataframe, _save_dataframe

    df = _object_frame()
    # Must not raise.
    _save_dataframe(df, tmp_path, "details", "parquet")
    loaded = _load_dataframe(tmp_path, "details")
    assert loaded is not None
    assert len(loaded) == 3
