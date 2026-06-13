# LedgerLens API

Hybrid on-chain fraud detection for the Stellar DEX — detecting wash
trading and artificial volume using **Benford's Law** and an **ensemble
machine-learning layer**, exposed via a FastAPI service and a Soroban
smart contract.

This repository implements the architecture described in
[`LedgerLens_README.md`](LedgerLens_README.md): a data ingestion layer for
the Stellar Horizon API, a detection engine combining Benford's Law digit
analysis with on-chain feature extraction, a heuristic risk-scoring layer
(the Phase 1 baseline for the planned RF/XGBoost/LightGBM ensemble), and a
public REST API for querying LedgerLens Risk Scores.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the API (serves the demo dataset)
uvicorn api.main:app --reload

# Run the test suite
pytest
```

The API will be available at `http://127.0.0.1:8000`, with interactive
docs at `http://127.0.0.1:8000/docs`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/score/{wallet}/{pair}` | LedgerLens Risk Score (0-100) for a wallet on an asset pair, e.g. `/score/GABC.../XLM/USDC:GISSUER...` |
| `GET` | `/alerts/recent` | Wallet/asset-pair combinations currently flagged as high-risk, with reasons |
| `GET` | `/assets/risk-ranking` | Asset pairs ranked by aggregate wallet risk score |

Asset pairs are identified as `BASE/COUNTER`, where each asset is either
`XLM` (native) or `CODE:ISSUER`.

---

## Batch Scoring Pipeline

`run_pipeline.py` runs the detection pipeline offline against live Horizon
data for a given asset pair, printing risk scores for every active wallet
as JSON:

```bash
python3 run_pipeline.py XLM "USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
```

---

## Repository Structure

```
.
├── README.md
├── LedgerLens_README.md          ← Full project methodology and roadmap
├── requirements.txt
├── pytest.ini
├── run_pipeline.py                ← Batch detection pipeline entry point
│
├── ingestion/
│   ├── data_models.py             ← Pydantic schemas for trades, assets, accounts
│   ├── historical_loader.py       ← Bulk historical trade ingestion (Horizon REST)
│   └── horizon_streamer.py         ← Real-time trade streaming (Horizon SSE)
│
├── detection/
│   ├── benford_engine.py          ← Benford's Law chi-square, z-score, MAD
│   ├── feature_engineering.py     ← Trade-pattern and volume/timing features
│   └── model_inference.py         ← LedgerLens Risk Score (0-100) computation
│
├── api/
│   ├── main.py                    ← FastAPI app
│   ├── schemas.py                 ← API response models
│   ├── storage.py                 ← Demo data store and score aggregation
│   └── routes/
│       ├── scores.py              ← GET /score/{wallet}/{pair}
│       ├── alerts.py              ← GET /alerts/recent
│       └── assets.py              ← GET /assets/risk-ranking
│
├── contracts/
│   ├── ledgerlens-score/          ← Soroban smart contract (Rust)
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   └── deploy.sh                  ← Testnet deployment script
│
└── tests/
    ├── test_benford.py
    ├── test_features.py
    ├── test_model_inference.py
    ├── test_ingestion.py
    └── test_api.py
```

---

## How Scoring Works

For a given wallet and asset pair:

1. **Benford analysis** — the leading-digit distribution of the wallet's
   trade amounts is compared against Benford's Law via chi-square, per-digit
   z-scores, and Mean Absolute Deviation (`detection/benford_engine.py`).
2. **Feature extraction** — trade-pattern features (counterparty
   concentration, round-trip frequency, self-matching rate) and
   volume/timing features (intra-minute clustering, off-hours activity) are
   computed from the wallet's trade history (`detection/feature_engineering.py`).
3. **Risk scoring** — the Benford and feature signals are combined into a
   0-100 LedgerLens Risk Score, with `benford_flag` and `ml_flag` booleans
   and a confidence value (`detection/model_inference.py`).

The current scorer is a weighted heuristic — the Phase 1 baseline described
in the [roadmap](LedgerLens_README.md#9-roadmap). It is designed as a
drop-in replacement target for the trained RF/XGBoost/LightGBM ensemble
planned for Phase 2.

---

## On-Chain Registry

`contracts/ledgerlens-score` is a Soroban contract that stores the latest
risk score per `(wallet, asset_pair)`. The authorised LedgerLens service
account writes scores via `submit_score`; any other contract can read them
via `get_score`, enabling composable on-chain risk gating for AMMs, lending
protocols, and DEX aggregators.

---

## Full Project Methodology

See [`LedgerLens_README.md`](LedgerLens_README.md) for the complete problem
statement, Benford's Law background, ML feature catalogue, and roadmap.
