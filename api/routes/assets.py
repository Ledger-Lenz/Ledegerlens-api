"""GET /assets/risk-ranking — asset pairs ranked by aggregate risk."""

from fastapi import APIRouter

from api import storage
from api.schemas import AssetRiskRanking

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("/risk-ranking", response_model=list[AssetRiskRanking])
def get_risk_ranking() -> list[AssetRiskRanking]:
    """Return asset pairs ranked by average wallet risk score, highest first."""
    return [AssetRiskRanking(**r) for r in storage.asset_risk_ranking()]
