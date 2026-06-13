import json

import httpx
import pytest

from ingestion.data_models import Asset
from ingestion.historical_loader import fetch_trades, load_all_trades, parse_trade_record
from ingestion.horizon_streamer import stream_trades

XLM = Asset(code="XLM")
USDC = Asset(code="USDC", issuer="GISSUER")


def _record(record_id, when, base_amount="100.0", counter_amount="10.0"):
    return {
        "id": record_id,
        "ledger_close_time": when,
        "base_account": "GBASEACCOUNT",
        "counter_account": "GCOUNTERACCOUNT",
        "base_asset_type": "native",
        "counter_asset_type": "credit_alphanum12",
        "counter_asset_code": "USDC",
        "counter_asset_issuer": "GISSUER",
        "base_amount": base_amount,
        "counter_amount": counter_amount,
        "price": {"n": 1, "d": 10},
        "base_is_seller": False,
    }


def test_parse_trade_record():
    record = _record("123-0", "2026-06-01T00:00:00Z")
    trade = parse_trade_record(record)
    assert trade.id == "123-0"
    assert trade.base_asset.code == "XLM"
    assert trade.counter_asset.code == "USDC"
    assert trade.counter_asset.issuer == "GISSUER"
    assert trade.price == pytest.approx(0.1)


def test_fetch_trades_single_page():
    records = [_record(f"{i}-0", "2026-06-01T00:00:00Z") for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"_embedded": {"records": records}})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://horizon.test")

    trades = fetch_trades(XLM, USDC, client=client)
    assert len(trades) == 3
    assert all(t.base_asset.code == "XLM" for t in trades)


def test_load_all_trades_stops_on_short_page():
    records = [_record(f"{i}-0", "2026-06-01T00:00:00Z") for i in range(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"_embedded": {"records": records}})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://horizon.test")

    trades = load_all_trades(XLM, USDC, page_size=200, client=client)
    assert len(trades) == 2


@pytest.mark.asyncio
async def test_stream_trades_yields_parsed_records():
    record = _record("123-0", "2026-06-01T00:00:00Z")
    sse_body = f'data: "hello"\r\n\r\ndata: {json.dumps(record)}\r\n\r\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://horizon.test")

    trades = []
    async for trade in stream_trades(XLM, USDC, client=client):
        trades.append(trade)

    assert len(trades) == 1
    assert trades[0].id == "123-0"
