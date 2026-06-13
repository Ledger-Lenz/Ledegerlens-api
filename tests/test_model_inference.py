from datetime import datetime, timedelta

from detection.model_inference import score_wallet
from ingestion.data_models import Asset, Trade

XLM = Asset(code="XLM")
USDC = Asset(code="USDC", issuer="GISSUER")


def make_trade(trade_id, base_account, counter_account, when, amount=500.0, base_is_seller=False):
    return Trade(
        id=trade_id,
        ledger_close_time=when,
        base_account=base_account,
        counter_account=counter_account,
        base_asset=XLM,
        counter_asset=USDC,
        base_amount=amount,
        counter_amount=amount * 0.1,
        price=0.1,
        base_is_seller=base_is_seller,
    )


def test_score_wallet_returns_expected_shape():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [make_trade("1", "WALLET_A", "WALLET_B", t0)]
    result = score_wallet(trades, "WALLET_A")
    assert result["wallet"] == "WALLET_A"
    assert 0 <= result["score"] <= 100
    assert isinstance(result["benford_flag"], bool)
    assert isinstance(result["ml_flag"], bool)
    assert "benford" in result["components"]
    assert "features" in result["components"]


def test_wash_trading_pattern_scores_higher_than_clean():
    t0 = datetime(2025, 1, 1, 12, 0, 0)

    # Wash pattern: same counterparty, fixed amounts, rapid round-trips.
    wash_trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0, amount=500, base_is_seller=False),
        make_trade(
            "2", "WALLET_A", "WALLET_B", t0 + timedelta(seconds=10), amount=500, base_is_seller=True
        ),
    ]

    # Clean pattern: varied counterparties and amounts, no quick reversal.
    clean_trades = [
        make_trade("1", "WALLET_C", "WALLET_X", t0, amount=187, base_is_seller=False),
        make_trade(
            "2", "WALLET_C", "WALLET_Y", t0 + timedelta(hours=5), amount=342, base_is_seller=False
        ),
    ]

    wash_score = score_wallet(wash_trades, "WALLET_A")["score"]
    clean_score = score_wallet(clean_trades, "WALLET_C")["score"]

    assert wash_score > clean_score


def test_empty_trade_history():
    result = score_wallet([], "WALLET_A")
    assert result["score"] == 0
    assert result["confidence"] == 0.0
