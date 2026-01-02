from __future__ import annotations

import pytest

from quant_engine.exceptions.core import FatalError
from quant_engine.utils.guards import assert_schema_subset


def test_assert_schema_subset_allows_equal() -> None:
    assert_schema_subset({"a", "b"}, {"a", "b"}, label="x")


def test_assert_schema_subset_allows_subset() -> None:
    assert_schema_subset({"a"}, {"a", "b"}, label="x")


def test_assert_schema_subset_rejects_extra() -> None:
    with pytest.raises(FatalError):
        assert_schema_subset({"a", "b", "c"}, {"a", "b"}, label="x")
