"""GET /alerts/recent — recently flagged wallet/asset-pair combinations."""

from fastapi import APIRouter, Query

from api import storage
from api.schemas import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/recent", response_model=list[Alert])
def get_recent_alerts(limit: int = Query(default=20, ge=1, le=100)) -> list[Alert]:
    """Return the highest-risk wallet/asset-pair combinations currently flagged."""
    return [Alert(**a) for a in storage.recent_alerts(limit=limit)]
