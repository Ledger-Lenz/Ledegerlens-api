from datetime import datetime, timedelta

from detection.feature_engineering import (
    counterparty_concentration_ratio,
    extract_wallet_features,
    intra_minute_clustering_coefficient,
    off_hours_activity_ratio,
    round_trip_trade_frequency,
    self_matching_rate,
    volume_spike_frequency,
    volume_to_unique_counterparty_ratio,
)
from ingestion.data_models import Asset, Trade

XLM = Asset(code="XLM")
USDC = Asset(code="USDC", issuer="GISSUER")


def make_trade(
    trade_id, base_account, counter_account, when, amount=100.0, base_is_seller=False
):
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


def test_counterparty_concentration_ratio_all_same_counterparty():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0),
        make_trade("2", "WALLET_A", "WALLET_B", t0 + timedelta(seconds=10)),
    ]
    assert counterparty_concentration_ratio(trades, "WALLET_A") == 1.0


def test_counterparty_concentration_ratio_split_evenly():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0, amount=100),
        make_trade("2", "WALLET_A", "WALLET_C", t0, amount=100),
    ]
    assert counterparty_concentration_ratio(trades, "WALLET_A") == 0.5


def test_round_trip_trade_frequency_detects_reversal():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0, base_is_seller=False),
        make_trade("2", "WALLET_A", "WALLET_B", t0 + timedelta(seconds=30), base_is_seller=True),
    ]
    # Trade 1 finds its reversal in trade 2; trade 2 has no later match.
    assert round_trip_trade_frequency(trades, "WALLET_A") == 0.5


def test_round_trip_trade_frequency_no_reversal_outside_window():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0, base_is_seller=False),
        make_trade(
            "2", "WALLET_A", "WALLET_B", t0 + timedelta(seconds=600), base_is_seller=True
        ),
    ]
    assert round_trip_trade_frequency(trades, "WALLET_A", max_ledger_gap_seconds=300) == 0.0


def test_self_matching_rate():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [make_trade("1", "WALLET_A", "WALLET_B", t0)]
    funder_by_account = {"WALLET_A": "FUNDER_X", "WALLET_B": "FUNDER_X"}
    assert self_matching_rate(trades, funder_by_account) == 1.0


def test_volume_to_unique_counterparty_ratio():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0, amount=100),
        make_trade("2", "WALLET_A", "WALLET_C", t0, amount=100),
    ]
    assert volume_to_unique_counterparty_ratio(trades, "WALLET_A") == 100.0


def test_intra_minute_clustering_coefficient():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", t0),
        make_trade("2", "WALLET_A", "WALLET_B", t0 + timedelta(seconds=30)),
        make_trade("3", "WALLET_A", "WALLET_B", t0 + timedelta(hours=2)),
    ]
    assert intra_minute_clustering_coefficient(trades) == 0.5


def test_off_hours_activity_ratio():
    trades = [
        make_trade("1", "WALLET_A", "WALLET_B", datetime(2025, 1, 1, 2, 0, 0)),
        make_trade("2", "WALLET_A", "WALLET_B", datetime(2025, 1, 1, 14, 0, 0)),
    ]
    assert off_hours_activity_ratio(trades) == 0.5


def test_volume_spike_frequency():
    volumes = [100, 100, 100, 1000]
    assert volume_spike_frequency(volumes, spike_multiplier=2.0) == 0.25


def test_extract_wallet_features_keys():
    t0 = datetime(2025, 1, 1, 12, 0, 0)
    trades = [make_trade("1", "WALLET_A", "WALLET_B", t0)]
    features = extract_wallet_features(trades, "WALLET_A")
    expected_keys = {
        "counterparty_concentration_ratio",
        "round_trip_trade_frequency",
        "self_matching_rate",
        "volume_to_unique_counterparty_ratio",
        "intra_minute_clustering_coefficient",
        "off_hours_activity_ratio",
    }
    assert set(features.keys()) == expected_keys
