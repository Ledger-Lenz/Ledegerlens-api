"""Real-time trade streaming from the Stellar Horizon API via Server-Sent Events."""

import json
from typing import AsyncIterator, Optional

import httpx

from ingestion.data_models import Asset, Trade
from ingestion.historical_loader import DEFAULT_HORIZON_URL, parse_trade_record


async def stream_trades(
    base_asset: Asset,
    counter_asset: Asset,
    cursor: str = "now",
    horizon_url: str = DEFAULT_HORIZON_URL,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[Trade]:
    """Yield `Trade` records as they occur, via Horizon's SSE trade stream.

    `cursor="now"` (the default) starts the stream from the next ledger
    close. `client` may be supplied for testing with a mock transport.
    """
    params: dict = {
        "base_asset_type": "native" if base_asset.issuer is None else "credit_alphanum12",
        "counter_asset_type": "native" if counter_asset.issuer is None else "credit_alphanum12",
        "cursor": cursor,
    }
    if base_asset.issuer is not None:
        params["base_asset_code"] = base_asset.code
        params["base_asset_issuer"] = base_asset.issuer
    if counter_asset.issuer is not None:
        params["counter_asset_code"] = counter_asset.code
        params["counter_asset_issuer"] = counter_asset.issuer

    headers = {"Accept": "text/event-stream"}

    owns_client = client is None
    client = client or httpx.AsyncClient(base_url=horizon_url, timeout=None)
    try:
        async with client.stream("GET", "/trades", params=params, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload in ("", '"hello"'):
                    continue
                record = json.loads(payload)
                yield parse_trade_record(record)
    finally:
        if owns_client:
            await client.aclose()
