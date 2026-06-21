"""Real-time risk scoring for wallets and asset pairs.

Combines the Benford's Law anomaly engine with the on-chain feature set to
produce a LedgerLens Risk Score (0-100). Implements the Phase 2 trained ML
ensemble with SHAP interpretability, seamlessly falling back to the Phase 1
weighted heuristic if the model artifact is missing or fails to load.
"""

import os
import logging
from typing import Iterable, Optional
import numpy as np

try:
    import joblib
    import shap
    _ML_DEPS_AVAILABLE = True
except ImportError:
    _ML_DEPS_AVAILABLE = False

from detection.benford_engine import benford_report
from detection.feature_engineering import extract_wallet_features
from ingestion.data_models import Trade

logger = logging.getLogger(__name__)

# Heuristic configuration (Fallback)
FEATURE_WEIGHTS = {
    "counterparty_concentration_ratio": 25.0,
    "round_trip_trade_frequency": 25.0,
    "self_matching_rate": 20.0,
    "intra_minute_clustering_coefficient": 10.0,
    "off_hours_activity_ratio": 5.0,
}
BENFORD_WEIGHT = 500.0
BENFORD_MAX_CONTRIBUTION = 15.0
ML_FLAG_THRESHOLD = 50

# ML configuration
# Probability threshold for the ML classifier. 0.5 is standard,
# but can be tuned based on precision/recall goals.
ML_CLASSIFIER_THRESHOLD = 0.5
MODEL_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble.joblib")

_MODEL_CACHE = None
_EXPLAINER_CACHE = None

def _load_model_artifact():
    """Attempt to load the trained ensemble and SHAP explainer."""
    global _MODEL_CACHE, _EXPLAINER_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _EXPLAINER_CACHE
        
    if not _ML_DEPS_AVAILABLE or not os.path.exists(MODEL_ARTIFACT_PATH):
        return None, None
        
    try:
        _MODEL_CACHE = joblib.load(MODEL_ARTIFACT_PATH)
        # Using TreeExplainer for tree-based ensembles (XGBoost/RF)
        _EXPLAINER_CACHE = shap.TreeExplainer(_MODEL_CACHE)
        return _MODEL_CACHE, _EXPLAINER_CACHE
    except Exception as e:
        logger.warning(f"Failed to load ML model artifact, triggering fallback: {e}")
        return None, None


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

    model, explainer = _load_model_artifact()

    if model is not None and explainer is not None:
        # --- ML Inference Path ---
        # Construct feature vector exactly matching training order
        feature_names = list(FEATURE_WEIGHTS.keys()) + ["benford_mad"]
        feature_vector = [features[name] for name in FEATURE_WEIGHTS] + [benford["mad"]]
        X = np.array([feature_vector])
        
        # Predict probability
        prob = model.predict_proba(X)[0][1]
        score = int(round(prob * 100))
        ml_flag = prob >= ML_CLASSIFIER_THRESHOLD
        
        # Compute SHAP values for explainability
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
            
        shap_attributions = {name: float(val) for name, val in zip(feature_names, shap_vals)}
        
        components = {
            "benford": benford,
            "features": features,
            "shap": shap_attributions,
        }
    else:
        # --- Heuristic Fallback Path ---
        feature_score = sum(
            FEATURE_WEIGHTS[name] * features[name] for name in FEATURE_WEIGHTS
        )
        benford_contribution = min(
            benford["mad"] * BENFORD_WEIGHT, BENFORD_MAX_CONTRIBUTION
        )
        raw_score = feature_score + benford_contribution
        score = int(round(min(max(raw_score, 0.0), 100.0)))
        ml_flag = score >= ML_FLAG_THRESHOLD
        
        components = {
            "benford": benford,
            "features": features,
        }

    confidence = round(min(100.0, 40 + len(wallet_trades)), 1) if wallet_trades else 0.0

    return {
        "wallet": wallet,
        "score": score,
        "benford_flag": benford["non_conforming"],
        "ml_flag": ml_flag,
        "confidence": confidence,
        "components": components,
    }
