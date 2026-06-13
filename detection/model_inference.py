"""Real-time risk scoring for wallets and asset pairs.

Combines the Benford's Law anomaly engine with the on-chain feature set to
produce a LedgerLens Risk Score (0-100). The weighted heuristic below is the
Phase 1 baseline scorer described in the roadmap; it is designed to be
swapped for the trained RF/XGBoost/LightGBM ensemble (Phase 2) without
changing the output contract.
"""

from typing import Iterable, Optional

from detection.benford_engine import benford_report
from detection.feature_engineering import extract_wallet_features
from ingestion.data_models import Trade

# Weight applied to each feature's contribution to the heuristic risk score.
# Each feature is already normalised to roughly [0, 1].
FEATURE_WEIGHTS = {
    "counterparty_concentration_ratio": 25.0,
    "round_trip_trade_frequency": 25.0,
    "self_matching_rate": 20.0,
    "intra_minute_clustering_coefficient": 10.0,
    "off_hours_activity_ratio": 5.0,
}

# Contribution of the Benford MAD score to the overall risk score.
# MAD scores above ~0.03 are strongly non-conforming; this maps that
# range onto roughly 0-15 points.
BENFORD_WEIGHT = 500.0
BENFORD_MAX_CONTRIBUTION = 15.0

# Risk score above this threshold sets `ml_flag` (heuristic stand-in for an
# ML classifier decision boundary).
ML_FLAG_THRESHOLD = 50


def score_wallet(
    trades: Iterable[Trade],
    wallet: str,
    funder_by_account: Optional[dict[str, str]] = None,
) -> dict:
    """Compute a LedgerLens Risk Score for `wallet` from its trade history."""
    trades = list(trades)
    wallet_trades = [t for t in trades if wallet in (t.base_account, t.counter_account)]

    amounts = [t.base_amount for t in wallet_trades]
    benford = benford_report(amounts)
    features = extract_wallet_features(wallet_trades, wallet, funder_by_account)

    feature_score = sum(
        FEATURE_WEIGHTS[name] * features[name] for name in FEATURE_WEIGHTS
    )
    benford_contribution = min(
        benford["mad"] * BENFORD_WEIGHT, BENFORD_MAX_CONTRIBUTION
    )

    raw_score = feature_score + benford_contribution
    score = round(min(max(raw_score, 0.0), 100.0))

    confidence = round(min(100.0, 40 + len(wallet_trades)), 1) if wallet_trades else 0.0

    return {
        "wallet": wallet,
        "score": score,
        "benford_flag": benford["non_conforming"],
        "ml_flag": score >= ML_FLAG_THRESHOLD,
        "confidence": confidence,
        "components": {
            "benford": benford,
            "features": features,
        },
    }
