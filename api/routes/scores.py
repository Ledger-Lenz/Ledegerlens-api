"""GET /score/{wallet}/{pair} — LedgerLens Risk Score lookup."""

from fastapi import APIRouter, HTTPException

from api import storage
from api.schemas import RiskScore

router = APIRouter(prefix="/score", tags=["scores"])


@router.get("/{wallet}/{pair:path}", response_model=RiskScore)
def get_score(wallet: str, pair: str) -> RiskScore:
    """Return the current LedgerLens Risk Score for `wallet` on `pair`.

    `pair` is an asset pair identifier in `BASE/COUNTER` form, e.g.
    `XLM/USDC:GISSUER...`.
    """
    import re
    if not re.match(r"^G[A-Z2-7]{55}$", wallet):
        raise HTTPException(
            status_code=400, detail="Invalid Stellar account ID format"
        )

    if pair not in storage.known_pairs():
        raise HTTPException(status_code=404, detail=f"Unknown asset pair: {pair}")
    if wallet not in storage.wallets_for_pair(pair):
        raise HTTPException(
            status_code=404, detail=f"No activity for wallet {wallet} on {pair}"
        )

    result = storage.compute_risk_score(wallet, pair)
    return RiskScore(
        wallet=result["wallet"],
        asset_pair=result["asset_pair"],
        score=result["score"],
        benford_flag=result["benford_flag"],
        ml_flag=result["ml_flag"],
        confidence=result["confidence"],
        timestamp=result["timestamp"],
    )
