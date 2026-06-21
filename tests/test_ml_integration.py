import pytest
from unittest.mock import patch
from detection.model_inference import score_wallet, MODEL_ARTIFACT_PATH
from ingestion.data_models import Trade
from datetime import datetime, timezone

def test_heuristic_fallback_when_model_missing():
    """Verify that score_wallet falls back to the heuristic if no model is found."""
    mock_trades = [
        Trade(
            id="1", 
            paging_token="1", 
            ledger_close_time=datetime.now(timezone.utc),
            offer_id="1", 
            base_offer_id="1", 
            base_account="WALLET_A", 
            base_amount="100.0", 
            base_asset_type="native", 
            counter_offer_id="2", 
            counter_account="WALLET_B", 
            counter_amount="100.0", 
            counter_asset_type="native", 
            base_is_seller=True, 
            price={"n": 1, "d": 1}
        )
    ]
    
    with patch("detection.model_inference._ML_DEPS_AVAILABLE", False):
        res = score_wallet(mock_trades, "WALLET_A")
        assert "score" in res
        assert "components" in res
        assert "shap" not in res["components"]
        assert 0 <= res["score"] <= 100

def test_ml_performance_regression_gate():
    """
    CI performance-regression gate.
    Ensures that if the model artifact exists, its predictions are sane
    and it correctly outputs SHAP values.
    """
    # If the model artifact does not exist, we skip rather than fail, 
    # to allow CI to run before training is complete.
    import os
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        pytest.skip("No trained model artifact found. Skipping ML regression gate.")
        
    mock_trades = [
        Trade(
            id="1", 
            paging_token="1", 
            ledger_close_time=datetime.now(timezone.utc),
            offer_id="1", 
            base_offer_id="1", 
            base_account="WALLET_A", 
            base_amount="100.0", 
            base_asset_type="native", 
            counter_offer_id="2", 
            counter_account="WALLET_B", 
            counter_amount="100.0", 
            counter_asset_type="native", 
            base_is_seller=True, 
            price={"n": 1, "d": 1}
        )
    ]
    
    res = score_wallet(mock_trades, "WALLET_A")
    assert "shap" in res["components"], "SHAP attributions missing from ML model output"
    assert 0 <= res["score"] <= 100, "ML Score out of bounds"
