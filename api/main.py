"""FastAPI entry point for the LedgerLens public REST API."""

from fastapi import FastAPI

from api.routes import alerts, assets, scores

app = FastAPI(
    title="LedgerLens API",
    description=(
        "Hybrid on-chain fraud detection for the Stellar DEX. "
        "Exposes LedgerLens Risk Scores, alerts, and asset risk rankings."
    ),
    version="0.1.0",
)

app.include_router(scores.router)
app.include_router(alerts.router)
app.include_router(assets.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
