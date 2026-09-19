"""Pydantic request/response models. Every endpoint is typed, so FastAPI
validates input and publishes an accurate OpenAPI schema at /docs."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Grade = Literal["advisory", "watch", "warning", "emergency"]
Level = Literal["low", "moderate", "high", "critical"]
Role = Literal["admin", "dma", "responder", "public"]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    full_name: str


class LocationOut(BaseModel):
    id: int
    code: str
    name: str
    lat: float
    lon: float
    population: int
    river_name: str | None = None
    coastal: bool
    access_index: float

    model_config = ConfigDict(from_attributes=True)


class RiskRequest(BaseModel):
    """Manual override of any input. Omitted fields fall back to the latest
    observation held for that district."""
    location_code: str
    rain_24h_mm: float | None = Field(default=None, ge=0, le=1200)
    rain_6h_mm: float | None = Field(default=None, ge=0, le=600)
    river_level_m: float | None = Field(default=None, ge=0, le=40)
    soil_moisture_index: float | None = Field(default=None, ge=0, le=1)
    wind_kmh: float | None = Field(default=None, ge=0, le=350)
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    temp_c: float | None = Field(default=None, ge=-10, le=60)
    slope_deg: float | None = Field(default=None, ge=0, le=60)
    elevation_m: float | None = Field(default=None, ge=0, le=3000)
    coastal: bool | None = None
    persist: bool = False


class HazardScores(BaseModel):
    flood: float
    landslide: float
    cyclone: float
    rainfall: float


class RiskResponse(BaseModel):
    location: str
    location_code: str
    hazards: HazardScores
    composite_pct: float
    level: Level
    confidence_pct: float
    model_name: str
    model_version: str
    is_trained_model: bool
    features: dict[str, float]
    contributions: dict[str, list[dict]]
    explanation_source: str
    data_mode: Literal["demo", "live"]
    generated_at: datetime


class AlertCreate(BaseModel):
    location_code: str
    hazard: str
    grade: Grade
    probability_pct: float = Field(ge=0, le=100)
    expected_duration_h: int = Field(default=6, ge=1, le=240)
    recommended_action: str = Field(min_length=10, max_length=2000)
    notify_public: bool = False
    notify_authorities: bool = True


class AlertOut(BaseModel):
    id: int
    location: str
    hazard: str
    grade: Grade
    probability_pct: float
    expected_duration_h: int
    recommended_action: str
    status: str
    issued_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class AllocationRequest(BaseModel):
    location_code: str
    commit: bool = False


class RouteRequest(BaseModel):
    from_node: str
    to_node: str
    vehicle: Literal["Ambulance", "NDRF truck", "Relief convoy", "Rescue boat"] = "Ambulance"
    objective: Literal["fast", "safe", "balanced"] = "balanced"
    include_alternatives: bool = True


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="anon", max_length=64)


class ChatResponse(BaseModel):
    reply: str
    answered_by: Literal["llm", "retrieval"]
    data_mode: Literal["demo", "live"]
    sources: list[str] = []
