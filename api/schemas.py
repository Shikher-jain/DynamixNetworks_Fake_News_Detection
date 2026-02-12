from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Article text or short claim")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top_tokens: dict[str, float] | None = None
