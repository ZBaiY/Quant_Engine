# Quant_Engine
This is my personal Quant Engine


```mermaid
flowchart TD

%% ==========================================================
%% LAYER 0 — DATA SOURCES
%% ==========================================================

subgraph L0[Layer 0 — Data Sources]
    MKT[Market Data\n• Binance Klines\n• Orderbook L1/L2\n• Trades]
    ALT[Alternative Data\n• News\n• Twitter/X\n• Reddit]
    OPT[Derivatives Data\n• Option Chains\n• IV Surface\n• OU Model]
end

%% ==========================================================
%% LAYER 1 — DATA INGESTION
%% ==========================================================

subgraph L1[Layer 1 — Data Ingestion]
    HDH[HistoricalDataHandler\n(clean, check, rescale)]
    RTDH[RealTimeDataHandler\n(stream bars, update windows)]
    SLOAD[SentimentLoader\n(fetch news/tweets, cache)]
end

MKT --> HDH
MKT --> RTDH
ALT --> SLOAD
OPT --> HDH


%% ==========================================================
%% LAYER 2 — FEATURE LAYER
%% ==========================================================

subgraph L2[Layer 2 — Feature Layer]
    FE[FeatureExtractor\n• TA indicators\n• Microstructure\n• Vol indicators\n• IV factors]
    SENTPIPE[SentimentPipeline\n• text cleaning\n• FinBERT/VADER fusion\n• score/vol/velocity]
    MERGE[Merge Features\n(TA + microstructure + vol + sentiment)]
end

HDH --> FE
RTDH --> FE
SLOAD --> SENTPIPE
FE --> MERGE
SENTPIPE --> MERGE


%% ==========================================================
%% LAYER 3 — MODELING LAYER
%% ==========================================================

subgraph L3[Layer 3 — Modeling Layer (ModelProto)]
    MODEL[Model Library\n• Statistical\n• ML models\n• Regime classifier\n• Physics/OU models]
end

MERGE --> MODEL


%% ==========================================================
%% LAYER 4 — DECISION LAYER
%% ==========================================================

subgraph L4[Layer 4 — Decision Layer (DecisionProto)]
    DECIDE[Decision Engine\n• Signal + sentiment regime fusion\n• Threshold gating]
end

MODEL --> DECIDE
SENTPIPE --> DECIDE


%% ==========================================================
%% LAYER 5 — RISK LAYER
%% ==========================================================

subgraph L5[Layer 5 — Risk Layer (RiskProto)]
    RISK[Risk Engine\n• SL/TP\n• ATR volatility\n• Sentiment-scaled size\n• Portfolio exposure]
end

DECIDE --> RISK
SENTPIPE --> RISK


%% ==========================================================
%% LAYER 6 — EXECUTION LAYER
%% ==========================================================

subgraph L6[Layer 6 — Execution Layer]
    POLICY[ExecutionPolicy\n• Immediate\n• TWAP\n• Maker-first]
    ROUTER[Router\n• L1/L2 aware\n• timeout rules]
    SLIP[SlippageModel\n• Linear impact\n• Depth model]
    MATCH[MatchingEngine\nBacktest = Live]
end

RISK --> POLICY
POLICY --> ROUTER
ROUTER --> SLIP
SLIP --> MATCH


%% ==========================================================
%% LAYER 7 — PORTFOLIO UPDATE
%% ==========================================================

subgraph L7[Portfolio & Accounting]
    PORT[Portfolio Manager\n• positions\n• P&L\n• leverage\n• exposures]
end

MATCH --> PORT


%% ==========================================================
%% LAYER 8 — REPORTING
%% ==========================================================

subgraph L8[Reporting Engine]
    REPORT[Reporting\n• Backtest metrics\n• IS/Slippage\n• Factor exposure\n• Sentiment regime attribution]
end

PORT --> REPORT
DECIDE --> REPORT
RISK --> REPORT
SENTPIPE --> REPORT
