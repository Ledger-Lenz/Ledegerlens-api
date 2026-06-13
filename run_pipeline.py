#!/usr/bin/env python3
"""LedgerLens detection pipeline entry point.

Loads historical trades for an asset pair from Horizon, scores every wallet
active on that pair, and prints the resulting LedgerLens Risk Scores as
JSON. This is the offline counterpart to the live API in `api/main.py`,
useful for batch scoring runs and Soroban `submit_score` updates.
"""

import argparse
import json
import sys

from detection.model_inference import score_wallet
from ingestion.data_models import Asset
from ingestion.historical_loader import load_all_trades


def parse_asset(value: str) -> Asset:
    """Parse an asset given as `CODE` (native) or `CODE:ISSUER`."""
    if ":" in value:
        code, issuer = value.split(":", 1)
        return Asset(code=code, issuer=issuer)
    return Asset(code=value, issuer=None)


def run(base_asset: Asset, counter_asset: Asset, horizon_url: str) -> list[dict]:
    trades = load_all_trades(base_asset, counter_asset, horizon_url=horizon_url)

    wallets = set()
    for trade in trades:
        wallets.add(trade.base_account)
        wallets.add(trade.counter_account)

    pair_id = f"{base_asset.identifier}/{counter_asset.identifier}"
    results = []
    for wallet in sorted(wallets):
        scored = score_wallet(trades, wallet)
        results.append(
            {
                "wallet": scored["wallet"],
                "asset_pair": pair_id,
                "score": scored["score"],
                "benford_flag": scored["benford_flag"],
                "ml_flag": scored["ml_flag"],
                "confidence": scored["confidence"],
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LedgerLens detection pipeline.")
    parser.add_argument("base_asset", help="Base asset, e.g. 'XLM' or 'CODE:ISSUER'")
    parser.add_argument("counter_asset", help="Counter asset, e.g. 'USDC:ISSUER'")
    parser.add_argument(
        "--horizon-url",
        default="https://horizon.stellar.org",
        help="Horizon API base URL (default: public Stellar Horizon)",
    )
    args = parser.parse_args(argv)

    base_asset = parse_asset(args.base_asset)
    counter_asset = parse_asset(args.counter_asset)

    results = run(base_asset, counter_asset, args.horizon_url)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
