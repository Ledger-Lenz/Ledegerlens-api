"""Pydantic schemas for Stellar DEX trade and account records."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Asset(BaseModel):
    """A Stellar asset (native XLM or issued asset)."""

    code: str = Field(..., description="Asset code, e.g. 'XLM' or 'USDC'")
    issuer: Optional[str] = Field(
        None, description="Issuer account ID; None for native XLM"
    )

    @property
    def identifier(self) -> str:
        return self.code if self.issuer is None else f"{self.code}:{self.issuer}"


class AssetPair(BaseModel):
    """A base/counter asset pair traded on the SDEX."""

    base: Asset
    counter: Asset

    @property
    def identifier(self) -> str:
        return f"{self.base.identifier}/{self.counter.identifier}"


class Trade(BaseModel):
    """A single executed trade pulled from the Horizon `/trades` endpoint."""

    id: str
    ledger_close_time: datetime
    base_account: str
    counter_account: str
    base_asset: Asset
    counter_asset: Asset
    base_amount: float = Field(..., gt=0)
    counter_amount: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    base_is_seller: bool = False

    @property
    def pair(self) -> AssetPair:
        return AssetPair(base=self.base_asset, counter=self.counter_asset)


class Account(BaseModel):
    """Minimal account metadata used for wallet graph features."""

    account_id: str
    created_at: Optional[datetime] = None
    funder: Optional[str] = Field(
        None, description="Account ID that funded this account's creation"
    )
    home_domain: Optional[str] = None
