"""On-chain feature extraction for the LedgerLens ML layer.

Computes trade-pattern, volume/timing, and wallet-graph features for a
wallet's trade history, as described in the LedgerLens methodology.
"""

from collections import Counter
from typing import Iterable, Optional

from ingestion.data_models import Account, Trade

# Trades within this many seconds of each other count as "intra-minute".
INTRA_MINUTE_WINDOW_SECONDS = 60

# Hours of day (UTC) considered "off-hours" for activity-ratio purposes.
OFF_HOURS = set(range(0, 6))


def _counterparty(trade: Trade, wallet: str) -> str:
    """The account on the other side of `trade` from `wallet`."""
    return trade.counter_account if trade.base_account == wallet else trade.base_account


def counterparty_concentration_ratio(trades: Iterable[Trade], wallet: str) -> float:
    """Fraction of a wallet's volume traded against its single largest counterparty."""
    trades = [t for t in trades if wallet in (t.base_account, t.counter_account)]
    if not trades:
        return 0.0
    volume_by_counterparty: Counter[str] = Counter()
    total_volume = 0.0
    for t in trades:
        volume_by_counterparty[_counterparty(t, wallet)] += t.base_amount
        total_volume += t.base_amount
    if total_volume == 0:
        return 0.0
    return max(volume_by_counterparty.values()) / total_volume


def round_trip_trade_frequency(
    trades: Iterable[Trade], wallet: str, max_ledger_gap_seconds: int = 300
) -> float:
    """Fraction of a wallet's trades that are reversed with the same
    counterparty within `max_ledger_gap_seconds`."""
    trades = sorted(
        (t for t in trades if wallet in (t.base_account, t.counter_account)),
        key=lambda t: t.ledger_close_time,
    )
    if not trades:
        return 0.0
    round_trips = 0
    for i, t in enumerate(trades):
        counterparty = _counterparty(t, wallet)
        for other in trades[i + 1 :]:
            gap = (other.ledger_close_time - t.ledger_close_time).total_seconds()
            if gap > max_ledger_gap_seconds:
                break
            if (
                _counterparty(other, wallet) == counterparty
                and other.base_is_seller != t.base_is_seller
            ):
                round_trips += 1
                break
    return round_trips / len(trades)


def self_matching_rate(trades: Iterable[Trade], funder_by_account: dict[str, str]) -> float:
    """Fraction of trades where both sides share the same funding source,
    a signal of wallets controlled by a single actor."""
    trades = list(trades)
    if not trades:
        return 0.0
    self_matches = 0
    for t in trades:
        base_funder = funder_by_account.get(t.base_account)
        counter_funder = funder_by_account.get(t.counter_account)
        if base_funder is not None and base_funder == counter_funder:
            self_matches += 1
    return self_matches / len(trades)


def volume_to_unique_counterparty_ratio(trades: Iterable[Trade], wallet: str) -> float:
    """Total volume traded divided by the number of unique counterparties."""
    trades = [t for t in trades if wallet in (t.base_account, t.counter_account)]
    if not trades:
        return 0.0
    total_volume = sum(t.base_amount for t in trades)
    unique_counterparties = {_counterparty(t, wallet) for t in trades}
    if not unique_counterparties:
        return 0.0
    return total_volume / len(unique_counterparties)


def intra_minute_clustering_coefficient(trades: Iterable[Trade]) -> float:
    """Fraction of trades that occur within `INTRA_MINUTE_WINDOW_SECONDS`
    of another trade in the same set."""
    trades = sorted(trades, key=lambda t: t.ledger_close_time)
    if len(trades) < 2:
        return 0.0
    clustered = 0
    for i in range(1, len(trades)):
        gap = (
            trades[i].ledger_close_time - trades[i - 1].ledger_close_time
        ).total_seconds()
        if gap <= INTRA_MINUTE_WINDOW_SECONDS:
            clustered += 1
    return clustered / (len(trades) - 1)


def off_hours_activity_ratio(trades: Iterable[Trade]) -> float:
    """Fraction of trades executed during off-hours (00:00-06:00 UTC)."""
    trades = list(trades)
    if not trades:
        return 0.0
    off_hours_trades = sum(1 for t in trades if t.ledger_close_time.hour in OFF_HOURS)
    return off_hours_trades / len(trades)


def volume_spike_frequency(
    volumes_by_window: Iterable[float], spike_multiplier: float = 3.0
) -> float:
    """Fraction of rolling-window volumes that spike above
    `spike_multiplier` times the mean of all windows."""
    volumes = list(volumes_by_window)
    if not volumes:
        return 0.0
    mean_volume = sum(volumes) / len(volumes)
    if mean_volume == 0:
        return 0.0
    spikes = sum(1 for v in volumes if v > spike_multiplier * mean_volume)
    return spikes / len(volumes)


def funding_source_similarity_score(accounts: Iterable[Account]) -> float:
    """Fraction of accounts in a cluster that share the most common funder."""
    accounts = [a for a in accounts if a.funder is not None]
    if not accounts:
        return 0.0
    funder_counts = Counter(a.funder for a in accounts)
    return funder_counts.most_common(1)[0][1] / len(accounts)


def account_age_days(account: Account, as_of) -> Optional[float]:
    """Age of `account` in days as of `as_of`, or None if creation time unknown."""
    if account.created_at is None:
        return None
    return (as_of - account.created_at).total_seconds() / 86400


def extract_wallet_features(
    trades: Iterable[Trade],
    wallet: str,
    funder_by_account: Optional[dict[str, str]] = None,
) -> dict:
    """Compute the trade-pattern and volume/timing feature set for `wallet`."""
    trades = list(trades)
    wallet_trades = [t for t in trades if wallet in (t.base_account, t.counter_account)]
    return {
        "counterparty_concentration_ratio": counterparty_concentration_ratio(
            wallet_trades, wallet
        ),
        "round_trip_trade_frequency": round_trip_trade_frequency(wallet_trades, wallet),
        "self_matching_rate": self_matching_rate(
            wallet_trades, funder_by_account or {}
        ),
        "volume_to_unique_counterparty_ratio": volume_to_unique_counterparty_ratio(
            wallet_trades, wallet
        ),
        "intra_minute_clustering_coefficient": intra_minute_clustering_coefficient(
            wallet_trades
        ),
        "off_hours_activity_ratio": off_hours_activity_ratio(wallet_trades),
    }
