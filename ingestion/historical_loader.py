"""Bulk historical trade ingestion from the Stellar Horizon API."""

from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.data_models import Asset, Trade

DEFAULT_HORIZON_URL = "https://horizon.stellar.org"


def _asset_from_horizon(record: dict, prefix: str) -> Asset:
    """Build an Asset from Horizon's `{prefix}_asset_type`/`_code`/`_issuer` fields."""
    if record.get(f"{prefix}_asset_type") == "native":
        return Asset(code="XLM", issuer=None)
    return Asset(
        code=record[f"{prefix}_asset_code"],
        issuer=record[f"{prefix}_asset_issuer"],
    )


def parse_trade_record(record: dict) -> Trade:
    """Convert a raw Horizon `/trades` record into a `Trade`."""
    return Trade(
        id=record["id"],
        ledger_close_time=datetime.fromisoformat(
            record["ledger_close_time"].replace("Z", "+00:00")
        ),
        base_account=record["base_account"],
        counter_account=record["counter_account"],
        base_asset=_asset_from_horizon(record, "base"),
        counter_asset=_asset_from_horizon(record, "counter"),
        base_amount=float(record["base_amount"]),
        counter_amount=float(record["counter_amount"]),
        price=float(record["price"]["n"]) / float(record["price"]["d"]),
        base_is_seller=bool(record.get("base_is_seller", False)),
    )


def fetch_trades(
    base_asset: Asset,
    counter_asset: Asset,
    limit: int = 200,
    cursor: Optional[str] = None,
    horizon_url: str = DEFAULT_HORIZON_URL,
    client: Optional[httpx.Client] = None,
) -> list[Trade]:
    """Fetch a page of historical trades for an asset pair from Horizon.

    `client` may be supplied for testing; otherwise a short-lived
    `httpx.Client` is created for the request.
    """
    params: dict = {
        "base_asset_type": "native" if base_asset.issuer is None else "credit_alphanum12",
        "counter_asset_type": "native" if counter_asset.issuer is None else "credit_alphanum12",
        "limit": limit,
        "order": "asc",
    }
    if base_asset.issuer is not None:
        params["base_asset_code"] = base_asset.code
        params["base_asset_issuer"] = base_asset.issuer
    if counter_asset.issuer is not None:
        params["counter_asset_code"] = counter_asset.code
        params["counter_asset_issuer"] = counter_asset.issuer
    if cursor is not None:
        params["cursor"] = cursor

    owns_client = client is None
    client = client or httpx.Client(base_url=horizon_url, timeout=30.0)
    try:
        response = client.get("/trades", params=params)
        response.raise_for_status()
        records = response.json()["_embedded"]["records"]
    finally:
        if owns_client:
            client.close()

    return [parse_trade_record(r) for r in records]


def load_all_trades(
    base_asset: Asset,
    counter_asset: Asset,
    since: Optional[datetime] = None,
    page_size: int = 200,
    horizon_url: str = DEFAULT_HORIZON_URL,
    client: Optional[httpx.Client] = None,
) -> list[Trade]:
    """Page through Horizon `/trades` until exhausted or `since` is reached."""
    all_trades: list[Trade] = []
    cursor: Optional[str] = None
    since = since or datetime.min.replace(tzinfo=timezone.utc)

    while True:
        page = fetch_trades(
            base_asset,
            counter_asset,
            limit=page_size,
            cursor=cursor,
            horizon_url=horizon_url,
            client=client,
        )
        if not page:
            break

        for trade in page:
            if trade.ledger_close_time < since:
                return all_trades
            all_trades.append(trade)

        if len(page) < page_size:
            break
        cursor = page[-1].id

    return all_trades
