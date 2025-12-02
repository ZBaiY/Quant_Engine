# Quant_Engine
This is my personal Quant Engine


```mermaid
flowchart TD

%% ==========================================================
%% LAYER 0 — DATA SOURCES
%% ==========================================================

subgraph L0[Layer 0 — Data Sources]
    MKT[Market Data<br>Binance Klines<br>Orderbook L1 L2<br>Trades]
    ALT[Alternative Data<br>News<br>Twitter X<br>Reddit]
    OPT[Derivatives Data<br>Option Chains<br>IV Surface<br>OU Model]
end


%% ==========================================================
%% LAYER 1 — DATA INGESTION
%% ==========================================================

subgraph L1[Layer 1 — Data Ingestion]
    HDH[HistoricalDataHandler<br>clean check rescale]
    RTDH[RealTimeDataHandler<br>stream bars<br>update windows]
    SLOAD[SentimentLoader<br>fetch news tweets<br>cache dedupe]
end

MKT --> HDH
MKT --> RTDH
ALT --> SLOAD
OPT --> HDH


%% ==========================================================
%% LAYER 2 — FEATURE LAYER
%% ==========================================================

subgraph L2[Layer 2 — Feature Layer]
    FE[FeatureExtractor<br>TA indicators<br>Microstructure<br>Vol indicators<br>IV factors]
    SENTPIPE[SentimentPipeline<br>text cleaning<br>FinBERT VADER fusion<br>sentiment score vol velocity]
    MERGE[Merge Features<br>TA + microstructure + vol + sentiment]
end

HDH --> FE
RTDH --> FE
SLOAD --> SENTPIPE
FE --> MERGE
SENTPIPE --> MERGE


%% ==========================================================
%% LAYER 3 — MODELING LAYER
%% ==========================================================

subgraph L3[Layer 3 — Modeling Layer ModelProto]
    MODEL[Model Library<br>Statistical<br>ML models<br>Regime classifier<br>Physics OU models]
end

MERGE --> MODEL


%% ==========================================================
%% LAYER 4 — DECISION LAYER
%% ==========================================================

subgraph L4[Layer 4 — Decision Layer DecisionProto]
    DECIDE[Decision Engine<br>Signal + sentiment regime fusion<br>Threshold gating]
end

MODEL --> DECIDE
SENTPIPE --> DECIDE


%% ==========================================================
%% LAYER 5 — RISK LAYER
%% ==========================================================

subgraph L5[Layer 5 — Risk Layer RiskProto]
    RISK[Risk Engine<br>SL TP<br>ATR volatility<br>Sentiment scaled size<br>Portfolio exposure]
end

DECIDE --> RISK
SENTPIPE --> RISK


%% ==========================================================
%% LAYER 6 — EXECUTION LAYER
%% ==========================================================

subgraph L6[Layer 6 — Execution Layer]
    POLICY[ExecutionPolicy<br>Immediate<br>TWAP<br>Maker first]
    ROUTER[Router<br>L1 L2 aware<br>timeout rules]
    SLIP[SlippageModel<br>Linear impact<br>Depth model]
    MATCH[MatchingEngine<br>Backtest = Live]
end

RISK --> POLICY
POLICY --> ROUTER
ROUTER --> SLIP
SLIP --> MATCH


%% ==========================================================
%% LAYER 7 — PORTFOLIO UPDATE
%% ==========================================================

subgraph L7[Portfolio and Accounting]
    PORT[Portfolio Manager<br>positions<br>PnL<br>leverage<br>exposures]
end

MATCH --> PORT


%% ==========================================================
%% LAYER 8 — REPORTING
%% ==========================================================

subgraph L8[Reporting Engine]
    REPORT[Reporting<br>Backtest metrics<br>IS Slippage<br>Factor exposure<br>Sentiment regime attribution]
end

PORT --> REPORT
DECIDE --> REPORT
RISK --> REPORT
SENTPIPE --> REPORT
