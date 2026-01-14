from __future__ import annotations

import asyncio

from quant_engine.utils.paths import data_root_from_file
from apps.run_code.backtest_app import run_backtest_app

STRATEGY_NAME = "EXAMPLE"
BIND_SYMBOLS = {"A": "BTCUSDT", "B": "ETHUSDT"}

# STRATEGY_NAME = "RSI-ADX-SIDEWAYS-FRACTIONAL" # "RSI-ADX-SIDEWAYS-FRACTIONAL" to turn on the fractional trading 
# BIND_SYMBOLS = {"A": "BTCUSDT", "window_RSI" : '14', "window_ADX": '14', "window_RSI_rolling": '5'}


START_TS = 1766966400000 - 30 * 24 * 60 * 60 * 1000  # 2025-11-29 00:00:00 UTC (epoch ms)
END_TS = 1767052800000 + 3 * 60 * 60 * 1000     # 2025-12-30 00:00:00 UTC (epoch ms) + 3 hours buffer


DATA_ROOT = data_root_from_file(__file__, levels_up=1) ## default to src/quant_engine/data/
async def main() -> None:
    await run_backtest_app(
        strategy_name=STRATEGY_NAME,
        bind_symbols=BIND_SYMBOLS,
        start_ts=START_TS,
        end_ts=END_TS,
        data_root=DATA_ROOT,
    )


if __name__ == "__main__":
    asyncio.run(main())
