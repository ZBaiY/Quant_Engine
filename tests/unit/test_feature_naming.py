from __future__ import annotations

import pytest

from quant_engine.contracts.model import parse_feature_name


def test_parse_feature_name_basic() -> None:
    assert parse_feature_name("RSI_MODEL_BTCUSDT") == ("RSI", "MODEL", "BTCUSDT", None)


def test_parse_feature_name_with_ref() -> None:
    assert parse_feature_name("SPREAD_MODEL_BTCUSDT^ETHUSDT") == ("SPREAD", "MODEL", "BTCUSDT", "ETHUSDT")


def test_parse_feature_name_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        parse_feature_name("INVALID")
    with pytest.raises(ValueError):
        parse_feature_name("TYPE__SYMBOL")
