<h1 align="center">
  <strong>
    The Quant Engine (TradeBot v4)
  </strong>
</h1>

<p align="center" style="font-size:26px; font-weight:600; line-height:1.35; padding:10px 0;">
  A modular, extensible, execution-realistic research & trading framework —
  designed for professional-grade systematic trading.
</p>

---

# Overview
Quant Engine (TradeBot v4) is a **contract-driven quant research & execution framework** with **one unified runtime semantics** across:
- **Backtest**
- **Mock (paper) trading**
- **Live trading**

Core idea: components communicate through explicit contracts (Protocols), while the runtime enforces **time/lifecycle correctness** and **execution realism**.

**Design rules (non-negotiable):**
- **Strategy** = static specification (what to run). No mode, no time, no side effects.
- **Engine** = runtime semantics (time, lifecycle, legality).
- **Driver** (BacktestEngine / RealtimeEngine) = time pusher (calls `engine.step()`), strategy-agnostic.

## Event-driven → Contract-driven
Earlier versions chained logic directly (Data → Features → Model → Decision → Risk → Execution), which became fragile with multi-source data and execution realism.

v4 keeps the runtime event-driven, but **logic boundaries are enforced by contracts**:
- `FeatureChannel` → features
- `ModelProto` → scores
- `DecisionProto` → intents
- `RiskProto` → target positions
- `ExecutionPolicy/Router/Slippage/Matching` → fills

## Strategy loading and runtime control-flow

For a zoomable version of this diagram on GitHub, open **[loading-and-runtime-control-flow.pdf](loading-and-runtime-control-flow.pdf)**.

```mermaid
sequenceDiagram
    autonumber
    participant U as User / Entry
    participant S as Strategy (static spec)
    participant L as StrategyLoader
    participant D as Driver (BacktestEngine / RealtimeEngine)
    participant E as StrategyEngine (runtime semantics)
    participant H as DataHandlers (OHLCV/Orderbook/Options/IV/Sentiment)
    participant F as FeatureExtractor
    participant M as Model
    participant R as Risk
    participant X as ExecutionEngine
    participant P as Portfolio

    U->>S: strategy = ExampleStrategy()
    U->>L: from_config(strategy, mode, overrides?)

    Note over L: DATA is the *only* symbol declaration source
    L->>H: build_multi_symbol_handlers(data_spec, backtest?, primary_symbol)
    Note over H: handlers are created as shells (no history loaded yet)

    L->>F: FeatureLoader.from_config(features_user, handlers...)
    L->>M: build_model(type, symbol, **params)
    L->>R: RiskLoader.from_config(risk_cfg, symbol?)
    L->>E: assemble StrategyEngine(mode, handlers, F, M/R/Decision, execution, portfolio)

    Note over E: Engine is assembled but not running yet
    D->>E: (BACKTEST only) load_history(start_ts, end_ts)
    E->>H: handler.load_history(start_ts, end_ts)

    D->>E: warmup(anchor_ts, warmup_steps)
    E->>H: warmup_to(anchor_ts) / align cursors
    loop warmup steps
        E->>F: update(anchor_ts)
    end

    loop main loop (Driver-controlled)
        D->>E: step()
        E->>H: pull market snapshot (primary clock)
        E->>F: update(timestamp)
        E->>M: predict(features)
        E->>R: adjust(intent, context)
        E->>X: execute(target, market_data, timestamp)
        E->>P: apply fills / update state
        E-->>D: snapshot{timestamp, features, scores, target, fills, portfolio}
    end
```

---

# How a Market Bar Flows Through the Quant Engine (v4)
At runtime, each new market bar triggers a clean, contract-driven pipeline:

1. Handlers provide the current market snapshot (multi-source)
2. Features are computed and merged into a single feature dict
3. Models output scores
4. Decision + Risk convert scores into a target position
5. Execution layer produces fills (same semantics across backtest/mock/live)
6. Portfolio + reporting update P&L / accounting / traces

Each layer depends **only on contracts**, not implementations.

---

# Minimal Strategy Configuration Example (v4 JSON)
This is the *runtime assembly config* consumed by `StrategyLoader.from_config(...)`. In practice your real strategies will have more features and data sources; the important part is the **shape** (and the naming convention).

```json
{
  "data": {
    "primary": {
      "ohlcv": { "symbol": "BTCUSDT", "tf": "15m" },
      "orderbook": { "symbol": "BTCUSDT", "depth": 20 }
    },
    "secondary": {
      "ETHUSDT": {
        "ohlcv": { "tf": "15m" }
      }
    }
  },
  "features_user": [
    { "name": "RSI_MODEL_BTCUSDT", "type": "RSI", "symbol": "BTCUSDT", "params": { "window": 14 } },
    { "name": "ATR_RISK_BTCUSDT", "type": "ATR", "symbol": "BTCUSDT", "params": { "window": 14 } }
  ],
  "model": {
    "type": "RSI_MODEL",
    "params": { "rsi_feature": "RSI_MODEL_BTCUSDT" }
  },
  "decision": {
    "type": "THRESHOLD",
    "params": { "threshold": 0.0 }
  },
  "risk": {
    "type": "ATR_SIZER",
    "params": { "risk_fraction": 0.02 }
  },
  "execution": {
    "type": "TWAP",
    "params": { "segments": 5 }
  }
}
```

Notes:
- **Symbols are declared only in `data`** (primary + secondary). Features/models may reference symbols but must not introduce new ones.
- Feature names follow: `TYPE_PURPOSE_SYMBOL` (and if there is a ref: `TYPE_PURPOSE_REF^SYMBOL`).

---

# Minimal Working Example (Python)
```python
from quant_engine.strategy.engine import EngineMode
from quant_engine.strategy.loader import StrategyLoader
from quant_engine.backtest.engine import BacktestEngine

# user-defined strategy: static spec only (no mode/time/side effects)
from strategies.example_strategy import ExampleStrategy

strategy = ExampleStrategy()

# assembly: Strategy + mode -> StrategyEngine (handlers are shells; no history loaded yet)
engine = StrategyLoader.from_config(strategy=strategy, mode=EngineMode.BACKTEST)

# driver: time pusher (strategy-agnostic)
BacktestEngine(
    engine,
    start_ts=1640995200.0,   # 2022-01-01 UTC
    end_ts=1672531200.0,     # 2023-01-01 UTC
    warmup_steps=200,
).run()
```

---

# Why This Architectural Shift Matters
It enables the Quant Engine to gracefully support:
- ML-based sentiment regimes
- microstructure-aware execution
- IV-surface-derived features (SABR / SSVI)
- volatility forecasting
- multi-asset & cross-asset strategies
- execution-realistic mock trading
- reproducible backtests with live parity
- research & execution decoupled but interoperable

---

# Full System Architecture Diagram
```mermaid
flowchart TD

subgraph L0[Layer 0 — Data Sources]
    OBD[Orderbook L1 L2<br>Trades]
    MKT[Market Data<br>Binance Klines<br>]
    OPT[Derivatives Data<br>Option Chains<br>raw bid/ask/strike/expiry]
    ALT[Alternative Data<br>News<br>Twitter X<br>Reddit] 
end

subgraph L1[Layer 1 — Data Ingestion]
    ROBD[RealTimeOrderbookHandler<br>stream bars<br>update windows]
    RTDH[RealTimeDataHandler<br>stream bars<br>update windows]
    OCDH[OptionChainDataHandler<br>group by expiry<br>cache chains]
    SLOAD[SentimentLoader<br>fetch news tweets<br>cache dedupe]
end

OBD --> ROBD
MKT --> RTDH
OPT --> OCDH
ALT --> SLOAD

subgraph L2[Layer 2 — Feature Layer]
    FE[FeatureExtractor<br>TA indicators<br>Microstructure<br>Vol indicators<br>IV factors]
    IVFEAT[IVSurfaceFeature<br>ATM IV<br>Skew/Smile<br>Term Structure<br>Vol-of-vol<br>Roll-down]
    SENTPIPE[SentimentPipeline<br>text cleaning<br>FinBERT VADER fusion<br>sentiment score vol velocity]
    MERGE[Merge Features<br>TA + microstructure + vol + IV + sentiment<br>kept as a dict handled by strat]
end

ROBD --> FE
RTDH --> FE
RTDH --> IVFEAT
OCDH --> IVFEAT
SLOAD --> SENTPIPE

FE --> MERGE
IVFEAT --> MERGE
SENTPIPE --> MERGE

subgraph L3[Layer 3 — Modeling Layer ModelProto]
    MODEL[Model Library<br>Statistical<br>ML models<br>Regime classifier<br>Physics OU models]
end

MERGE --> MODEL

subgraph L4[Layer 4 — Decision Layer DecisionProto]
    DECIDE[Decision Engine<br>Signal + sentiment regime fusion<br>Threshold gating]
end

MODEL --> DECIDE
MERGE --> DECIDE

subgraph L5[Layer 5 — Risk Layer RiskProto]
    RISK[Risk Engine<br>SL TP<br>ATR volatility<br>Sentiment scaled size<br>Portfolio exposure]
end

MERGE --> RISK

subgraph L6[Layer 6 — Execution Layer]
    direction LR
    POLICY[ExecutionPolicy<br>Immediate<br>TWAP<br>Maker first]
    ROUTER[Router<br>L1 L2 aware<br>timeout rules]
    SLIP[SlippageModel<br>Linear impact<br>Depth model]
    MATCH[MatchingEngine<br>Backtest = Live]
end

RISK --> POLICY
POLICY --> ROUTER
ROUTER --> SLIP
SLIP --> MATCH

subgraph L7[Portfolio and Accounting]
    PORT[Portfolio Manager<br>positions<br>PnL<br>leverage<br>exposures]
end

MATCH --> PORT

subgraph L8[Reporting Engine]
    REPORT[Reporting<br>Backtest metrics<br>IS Slippage<br>Factor exposure<br>Sentiment regime attribution]
end

PORT --> REPORT
DECIDE --> REPORT
DECIDE --> RISK
RISK --> REPORT
SENTPIPE --> REPORT
IVFEAT --> REPORT
```
