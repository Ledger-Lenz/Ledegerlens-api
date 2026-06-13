# LedgerLens 🔍

### Hybrid On-Chain Fraud Detection for the Stellar DEX
**Detecting Wash Trading and Artificial Volume Using Benford's Law + Ensemble Machine Learning on Soroban**

---

> *"On a transparent ledger, every transaction is visible. LedgerLens makes them legible."*

[![Built on Stellar](https://img.shields.io/badge/Built%20on-Stellar-blue?logo=stellar)](https://stellar.org)
[![Soroban Smart Contracts](https://img.shields.io/badge/Smart%20Contracts-Soroban-purple)](https://soroban.stellar.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Drip Wave Application](https://img.shields.io/badge/Drip%20Wave-Application-orange)](https://dripwave.stellar.org)

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [What LedgerLens Does](#2-what-ledgerlens-does)
3. [Why Stellar and Soroban](#3-why-stellar-and-soroban)
4. [How It Works — Technical Architecture](#4-how-it-works--technical-architecture)
5. [Benford's Law on the Blockchain](#5-benfords-law-on-the-blockchain)
6. [Machine Learning Layer](#6-machine-learning-layer)
7. [Soroban Smart Contract Layer](#7-soroban-smart-contract-layer)
8. [Repository Structure](#8-repository-structure)
9. [Roadmap](#9-roadmap)
10. [Why This Matters for the Stellar Ecosystem](#10-why-this-matters-for-the-stellar-ecosystem)
11. [About and Collaboration](#11-about-and-collaboration)
12. [References](#12-references)

---

## 1. The Problem

Wash trading — the practice of simultaneously buying and selling the same asset to artificially inflate trading volume — is one of the most pervasive and damaging forms of market manipulation in decentralised finance. While blockchain's transparency means every transaction is recorded, the sheer volume of on-chain activity makes manual detection impossible. Bad actors exploit this gap.

On decentralised exchanges (DEXs), wash trading causes real harm:

- **Traders are misled** into believing an asset has genuine liquidity and market interest when it does not
- **Token issuers manipulate rankings** on DEX aggregators and data platforms by inflating 24-hour volume figures
- **Liquidity providers lose funds** by entering pools that appear active but are dominated by self-dealing activity
- **Ecosystem credibility suffers** — inflated volume metrics on the Stellar DEX undermine confidence from institutional participants, exchanges, and new users

Existing detection approaches are either manual (slow and unscalable) or rely on simple heuristics (easily gamed). No production-grade, open-source wash trading detection system exists for the Stellar DEX. **LedgerLens is built to fill that gap.**

---

## 2. What LedgerLens Does

LedgerLens is a hybrid fraud detection system that combines two powerful analytical techniques — **Benford's Law digit analysis** and **ensemble machine learning** — to detect wash trading patterns on the Stellar Decentralised Exchange (SDEX) in near real-time.

At a high level, it does three things:

### 🔍 Detects
Identifies wallet pairs, trading clusters, and asset pools exhibiting statistically anomalous transaction patterns consistent with wash trading — including circular trade routing, self-matching order behaviour, and artificial volume concentration.

### 📊 Scores
Assigns each wallet and each trading pair a **LedgerLens Risk Score (0–100)** based on the combined output of its Benford anomaly metrics and machine learning classifiers. Scores update continuously as new ledger data is processed.

### 📡 Reports
Exposes risk scores and flagged activity through a public API and a lightweight dashboard, making the intelligence accessible to DEX users, protocol teams, wallet providers, and compliance integrators — without requiring any technical expertise to consume.

---

## 3. Why Stellar and Soroban

Stellar is uniquely positioned for this kind of application for three reasons:

**Speed and cost** — Stellar's 3–5 second settlement finality and sub-cent transaction fees mean that wash trading can be executed at enormous scale very cheaply. This creates a genuine detection problem that does not exist at the same severity on slower, more expensive chains.

**The SDEX is native and transparent** — Unlike EVM chains where DEX activity is scattered across dozens of protocols with varying data standards, the Stellar DEX is a first-class, protocol-level construct. Every trade produces a structured, standardised ledger entry accessible via Horizon API. This consistency makes the transaction data ideal for machine learning feature extraction.

**Soroban enables on-chain logic** — With Soroban smart contracts, LedgerLens can register risk scores and anomaly flags directly on-chain, making them composable. Other protocols — AMMs, lending platforms, DEX aggregators — can query LedgerLens scores natively within their own contract logic to gate suspicious activity without relying on an external oracle.

---

## 4. How It Works — Technical Architecture

LedgerLens operates as a three-layer system:

```
┌─────────────────────────────────────────────────────────────┐
│                     LAYER 1: DATA INGESTION                 │
│                                                             │
│  Stellar Horizon API → Trade history, order book events,   │
│  account activity, asset metadata, payment paths           │
│  Streamed continuously via SSE or polled per ledger close   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: DETECTION ENGINE                  │
│                                                             │
│  ┌─────────────────────┐   ┌──────────────────────────┐    │
│  │  Benford's Law       │   │  Ensemble ML Models       │   │
│  │  Anomaly Engine      │   │  (RF, XGBoost, LightGBM) │   │
│  │                      │   │                           │   │
│  │  • Chi-square stat   │   │  • 30+ on-chain features  │   │
│  │  • Z-score per digit │   │  • Trained on labelled    │   │
│  │  • MAD score         │   │    wash trade patterns    │   │
│  │  • Per asset, per    │   │  • SHAP interpretability  │   │
│  │    wallet, per pair  │   │  • Continuous retraining  │   │
│  └──────────┬──────────┘   └──────────────┬────────────┘   │
│             │                             │                  │
│             └──────────┬──────────────────┘                 │
│                        ▼                                     │
│              LedgerLens Risk Score (0–100)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               LAYER 3: SOROBAN CONTRACT + API               │
│                                                             │
│  • Risk scores registered on-chain via Soroban contract     │
│  • Public REST API for external integrations                │
│  • Lightweight web dashboard for ecosystem visibility       │
│  • Webhook alerts for protocol teams                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Benford's Law on the Blockchain

Benford's Law states that in many naturally occurring numerical datasets, the leading digit is not uniformly distributed — the digit 1 appears approximately 30.1% of the time, declining to 4.6% for the digit 9. This distribution holds for organic financial activity because genuine trading decisions produce a wide, unbiased spread of transaction sizes.

**Wash trading violates this.** Because wash traders typically use programmatic bots that operate with fixed lot sizes, round-number amounts, or algorithmically generated values designed to mimic volume without attracting attention, the digit distribution of their transaction amounts deviates systematically from the Benford expectation.

LedgerLens applies three Benford metrics to transaction amount data for each wallet and each trading pair over rolling time windows:

| Metric | What it measures |
|---|---|
| **Chi-square statistic** | Whether the overall digit distribution deviates significantly from Benford's expected distribution |
| **Z-score (per digit)** | Whether any individual digit (1–9) appears with significantly higher or lower frequency than expected |
| **Mean Absolute Deviation (MAD)** | A composite measure of distributional divergence; values above 0.015 indicate non-conformity |

These metrics alone are not definitive — legitimate high-frequency market makers can also produce non-Benford distributions. This is precisely why LedgerLens combines Benford signals with the machine learning layer rather than using either in isolation.

---

## 6. Machine Learning Layer

The ML layer trains ensemble classifiers on a labelled dataset of confirmed wash trading patterns derived from known SDEX manipulation events, cross-referenced with Stellar expert community reports and comparable on-chain forensics datasets from other chains.

### Features (30+, grouped by category)

**Benford Features (15)**
- Chi-square, Z-score, and MAD for transaction amounts across 5 rolling time windows (1h, 4h, 24h, 7d, 30d)

**Trade Pattern Features**
- Counterparty concentration ratio (fraction of volume with a single counterparty)
- Round-trip trade frequency (trades that return assets to the originating wallet within N ledgers)
- Self-matching rate (buy and sell orders from wallets sharing funding sources)
- Order cancellation rate and timing patterns

**Volume and Timing Features**
- Volume-to-unique-counterparty ratio
- Intra-minute trade clustering coefficient
- Off-hours activity ratio (trades at statistically unusual ledger times)
- Volume spike frequency relative to rolling baseline

**Wallet Graph Features**
- Funding source similarity score (wallets funded from the same source)
- Network centrality within trading cluster graphs
- Account age at time of trading activity

### Models

| Model | Role |
|---|---|
| **Random Forest** | Stable baseline; handles missing features gracefully |
| **XGBoost** | Primary classifier; strongest performance on tabular on-chain data |
| **LightGBM** | High-speed inference for real-time scoring |

All models are trained with **SMOTE** to handle class imbalance (genuine wash trades are rare relative to clean activity) and evaluated using **AUC-ROC**, **Precision-Recall AUC**, and **F1-score**. SHAP values provide interpretable explanations of each risk score for end-users and auditors.

---

## 7. Soroban Smart Contract Layer

The Soroban contract serves as the **on-chain truth layer** for LedgerLens risk scores. It provides two core functions:

### `submit_score(wallet: Address, asset_pair: Symbol, score: u32, timestamp: u64)`
Called by the LedgerLens off-chain service to register a computed risk score on-chain. Only the authorised LedgerLens service account can write scores.

### `get_score(wallet: Address, asset_pair: Symbol) → RiskScore`
Read-only function callable by any other Soroban contract. Returns the most recent LedgerLens risk score and timestamp for a given wallet and asset pair.

This composability means that AMMs, lending protocols, and DEX aggregators on Stellar can integrate LedgerLens fraud signals natively — for example, gating liquidity provision from wallets with a risk score above a configurable threshold — without any off-chain dependency.

```rust
// Simplified Soroban interface (Rust pseudocode)
pub struct RiskScore {
    pub score: u32,          // 0–100; higher = more suspicious
    pub benford_flag: bool,  // True if Benford anomaly detected
    pub ml_flag: bool,       // True if ML classifier flagged
    pub timestamp: u64,      // Ledger timestamp of last update
    pub confidence: u32,     // Model confidence 0–100
}
```

---

## 8. Repository Structure

```
ledgerlens/
│
├── README.md                         ← This file
├── requirements.txt                  ← Python dependencies
├── run_pipeline.py                   ← Full detection pipeline entry point
│
├── ingestion/
│   ├── horizon_streamer.py           ← Real-time trade data from Horizon API
│   ├── historical_loader.py          ← Bulk historical trade ingestion
│   └── data_models.py               ← Pydantic schemas for trade records
│
├── detection/
│   ├── benford_engine.py             ← Benford's Law feature computation
│   ├── feature_engineering.py       ← On-chain ML feature extraction
│   ├── model_training.py            ← Train ensemble classifiers
│   ├── model_inference.py           ← Real-time risk scoring
│   └── shap_explainer.py            ← SHAP interpretability layer
│
├── contracts/
│   ├── ledgerlens-score/            ← Soroban smart contract (Rust)
│   │   ├── src/lib.rs               ← Contract logic
│   │   └── Cargo.toml
│   └── deploy.sh                    ← Testnet deployment script
│
├── api/
│   ├── main.py                      ← FastAPI REST API server
│   ├── routes/
│   │   ├── scores.py                ← GET /score/{wallet}/{pair}
│   │   ├── alerts.py                ← GET /alerts/recent
│   │   └── assets.py               ← GET /assets/risk-ranking
│   └── schemas.py
│
├── dashboard/
│   ├── index.html                   ← Lightweight web dashboard
│   ├── app.js                       ← Dashboard frontend logic
│   └── styles.css
│
└── tests/
    ├── test_benford.py
    ├── test_features.py
    └── test_api.py
```

---

## 9. Roadmap

### Phase 1 — Foundation *(Months 1–2)*
- [ ] Stellar Horizon API ingestion pipeline (historical + streaming)
- [ ] Benford's Law engine for on-chain transaction amounts
- [ ] Initial feature engineering from SDEX trade data
- [ ] Baseline ML model training on historical wash trade patterns
- [ ] Internal testing on Stellar Testnet

### Phase 2 — Core Product *(Months 3–4)*
- [ ] Full ensemble model training and evaluation
- [ ] SHAP interpretability integration
- [ ] Soroban smart contract deployment on Testnet
- [ ] Public REST API (v1) with rate limiting
- [ ] Web dashboard (beta)

### Phase 3 — Ecosystem Integration *(Months 5–6)*
- [ ] Mainnet deployment
- [ ] SDK for protocol integrations (Python + JavaScript)
- [ ] Webhook alert system for asset issuers and protocol teams
- [ ] Open dataset release: labelled SDEX wash trade patterns
- [ ] Community feedback and model refinement cycle

### Phase 4 — Scale *(Post-Grant)*
- [ ] Continuous model retraining pipeline
- [ ] Coverage expansion to AMM pools and cross-asset paths
- [ ] Integration partnerships with Stellar DEX aggregators
- [ ] Developer documentation portal

---

## 10. Why This Matters for the Stellar Ecosystem

Stellar's growth as a platform for real-world asset tokenisation, remittances, and DeFi depends on the credibility of its markets. A DEX where volume figures cannot be trusted is a DEX that institutional participants, regulated entities, and serious retail traders will avoid.

LedgerLens addresses this directly:

**For traders** — Know which assets have genuine liquidity before placing orders. The risk score dashboard provides instant, interpretable signals without requiring any on-chain expertise.

**For asset issuers** — Demonstrate that your token's volume is organic. A low LedgerLens risk score is a credibility signal that can be cited in listings, investor materials, and community communications.

**For protocol teams** — Integrate LedgerLens scores into AMM and lending contract logic to automatically protect your users from interacting with wash-traded assets or flagged wallets.

**For the Stellar Foundation and ecosystem** — An open, verifiable, community-maintained fraud detection layer strengthens the case for Stellar as a credible and trustworthy financial infrastructure platform.

LedgerLens is not a surveillance tool. It is an **open-source public good** — the scores, methodology, and training data will be fully transparent and auditable by anyone. In keeping with Stellar's mission of open financial infrastructure, LedgerLens will always be free to query and open to community contribution.

---

## 11. About and Collaboration

LedgerLens is being developed as an open-source contribution to the Stellar ecosystem, submitted as part of the **Drip Wave builder programme**.

**We are actively looking for collaborators** to help build this. If you have experience in any of the following, please reach out:

- Stellar / Soroban smart contract development (Rust)
- Python backend development and ML pipeline engineering
- On-chain data analysis and blockchain forensics
- Frontend development (dashboard)
- DeFi protocol integration

This is an early-stage project with a clear scope, a working methodology, and a real problem to solve. If you want to build something meaningful for the Stellar ecosystem, get in touch.

📧 **Contact:** [your email here]  
🐦 **Twitter/X:** [your handle here]  
💬 **Stellar Discord:** [your handle here]

---

## 12. References

- Benford, F. (1938) 'The law of anomalous numbers', *Proceedings of the American Philosophical Society*, 78(4), pp. 551–572.
- Al Ali, A. et al. (2023) 'A powerful predicting model for financial statement fraud based on optimized XGBoost ensemble learning technique', *Applied Sciences*, 13(4).
- Antonio, G.R. (2023) 'Numbers don't lie: Decoding financial error and fraud through Benford's law', *Journal of Entrepreneurship*.
- Nti, I.K. and Somanathan, A.R. (2024) 'A scalable RF-XGBoost framework for financial fraud mitigation', *IEEE Transactions on Computational Social Systems*, 11(2), pp. 410–422.
- Yadavalli, R. and Polisetti, R. (2025) 'Optimized financial fraud detection using SMOTE-enhanced ensemble learning with CatBoost and LightGBM', *ICVADV 2025*.
- Harea, R. and Mihailă, S. (2025) 'Benford's law: Applicability in accounting and financial anomaly detection', *Challenges of Accounting for Young Researchers*, 3(1).
- Stellar Development Foundation (2024) *Horizon API Documentation*. Available at: https://developers.stellar.org/api/horizon
- Stellar Development Foundation (2024) *Soroban Smart Contract Documentation*. Available at: https://soroban.stellar.org/docs

---

<div align="center">

**LedgerLens** — Making the Stellar ledger legible.

*Built for the Stellar ecosystem. Open source. Community owned.*

</div>
