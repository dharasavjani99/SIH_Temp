"""SQLAlchemy ORM models — the fifteen tables in the ResQTech schema.

Geometry is stored as plain lat/lon float columns so the schema runs on stock
PostgreSQL. Swap to PostGIS `geography(Point)` columns when spatial queries
(within-radius, intersects) are needed; nothing else in the code changes.
"""
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, JSON,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(190), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="public")  # admin|dma|responder|public
    district_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Location(Base, TimestampMixin):
    """A district / administrative unit plus its static environmental profile."""
    __tablename__ = "locations"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    state: Mapped[str] = mapped_column(String(80), default="Gujarat")
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    population: Mapped[int] = mapped_column(Integer)
    exposure_band_population: Mapped[int] = mapped_column(Integer)
    elevation_m: Mapped[float] = mapped_column(Float)
    slope_deg: Mapped[float] = mapped_column(Float)
    soil_moisture_index: Mapped[float] = mapped_column(Float)
    river_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    river_danger_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    coastal: Mapped[bool] = mapped_column(Boolean, default=False)
    access_index: Mapped[float] = mapped_column(Float, default=0.8)
    hospital_count: Mapped[int] = mapped_column(Integer, default=0)
    school_count: Mapped[int] = mapped_column(Integer, default=0)
    elderly_share: Mapped[float] = mapped_column(Float, default=0.08)
    child_share: Mapped[float] = mapped_column(Float, default=0.21)

    weather = relationship("WeatherData", back_populates="location")
    predictions = relationship("RiskPrediction", back_populates="location")


class WeatherData(Base):
    __tablename__ = "weather_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    rain_6h_mm: Mapped[float] = mapped_column(Float, default=0)
    rain_24h_mm: Mapped[float] = mapped_column(Float, default=0)
    temp_c: Mapped[float] = mapped_column(Float, default=0)
    humidity_pct: Mapped[float] = mapped_column(Float, default=0)
    wind_kmh: Mapped[float] = mapped_column(Float, default=0)
    river_level_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="demo")   # imd|cwc|demo
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)

    location = relationship("Location", back_populates="weather")


class Disaster(Base, TimestampMixin):
    __tablename__ = "disasters"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    hazard: Mapped[str] = mapped_column(String(40))         # flood|landslide|cyclone|rainfall
    severity: Mapped[str] = mapped_column(String(20))       # low|moderate|high|critical
    status: Mapped[str] = mapped_column(String(20), default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer, default=0)
    flood_pct: Mapped[float] = mapped_column(Float)
    landslide_pct: Mapped[float] = mapped_column(Float)
    cyclone_pct: Mapped[float] = mapped_column(Float)
    rainfall_pct: Mapped[float] = mapped_column(Float)
    composite_pct: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(20))
    confidence_pct: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(60))
    model_version: Mapped[str] = mapped_column(String(20))
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    contributions: Mapped[dict] = mapped_column(JSON, default=dict)

    location = relationship("Location", back_populates="predictions")


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    hazard: Mapped[str] = mapped_column(String(40))
    grade: Mapped[str] = mapped_column(String(20))   # advisory|watch|warning|emergency
    probability_pct: Mapped[float] = mapped_column(Float)
    expected_duration_h: Mapped[int] = mapped_column(Integer, default=6)
    recommended_action: Mapped[str] = mapped_column(Text)
    issued_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    channels: Mapped[dict] = mapped_column(JSON, default=dict)  # {"public": true, "authorities": true}
    status: Mapped[str] = mapped_column(String(20), default="draft")


class VulnerablePopulation(Base):
    __tablename__ = "vulnerable_population"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    exposed_total: Mapped[int] = mapped_column(Integer)
    high_priority: Mapped[int] = mapped_column(Integer)
    evacuation_required: Mapped[int] = mapped_column(Integer)
    elderly: Mapped[int] = mapped_column(Integer, default=0)
    children: Mapped[int] = mapped_column(Integer, default=0)
    hospital_inpatients: Mapped[int] = mapped_column(Integer, default=0)
    school_students: Mapped[int] = mapped_column(Integer, default=0)
    vulnerability_index: Mapped[float] = mapped_column(Float)
    priority_score: Mapped[float] = mapped_column(Float)


class Resource(Base, TimestampMixin):
    __tablename__ = "resources"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True)
    resource_type: Mapped[str] = mapped_column(String(60), index=True)
    unit: Mapped[str] = mapped_column(String(20))
    total_qty: Mapped[int] = mapped_column(Integer)
    available_qty: Mapped[int] = mapped_column(Integer)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="ready")
    priority: Mapped[str] = mapped_column(String(20), default="medium")


class ResourceAllocation(Base, TimestampMixin):
    __tablename__ = "resource_allocations"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    resource_type: Mapped[str] = mapped_column(String(60))
    required_qty: Mapped[int] = mapped_column(Integer)
    assigned_qty: Mapped[int] = mapped_column(Integer)
    shortfall_qty: Mapped[int] = mapped_column(Integer, default=0)
    rationale: Mapped[dict] = mapped_column(JSON, default=dict)
    approved_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="proposed")


class Shelter(Base, TimestampMixin):
    __tablename__ = "shelters"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    capacity: Mapped[int] = mapped_column(Integer)
    occupancy: Mapped[int] = mapped_column(Integer, default=0)
    facilities: Mapped[dict] = mapped_column(JSON, default=list)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)

    @property
    def available(self) -> int:
        return max(self.capacity - self.occupancy, 0)


class Hospital(Base, TimestampMixin):
    __tablename__ = "hospitals"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    beds_total: Mapped[int] = mapped_column(Integer)
    beds_free: Mapped[int] = mapped_column(Integer)
    icu_free: Mapped[int] = mapped_column(Integer, default=0)


class RoadNode(Base):
    __tablename__ = "road_nodes"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)


class Road(Base, TimestampMixin):
    """An edge of the routable network."""
    __tablename__ = "roads"
    __table_args__ = (UniqueConstraint("from_node", "to_node", name="uq_road_edge"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    from_node: Mapped[str] = mapped_column(String(12), index=True)
    to_node: Mapped[str] = mapped_column(String(12), index=True)
    condition: Mapped[str] = mapped_column(String(20), default="open")  # open|congested|flooded|landslide
    hazard_score: Mapped[float] = mapped_column(Float, default=0.0)
    length_km: Mapped[float | None] = mapped_column(Float, nullable=True)


class EmergencyTeam(Base, TimestampMixin):
    __tablename__ = "emergency_teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    force: Mapped[str] = mapped_column(String(40), default="NDRF")
    members: Mapped[int] = mapped_column(Integer)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    state: Mapped[str] = mapped_column(String(20), default="standby")
    current_task: Mapped[str | None] = mapped_column(Text, nullable=True)


class ImpactAssessment(Base, TimestampMixin):
    __tablename__ = "impact_assessments"
    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"), nullable=True)
    source_image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scene_footprint_km2: Mapped[float] = mapped_column(Float, default=96.0)
    affected_area_km2: Mapped[float] = mapped_column(Float)
    damaged_structures: Mapped[int] = mapped_column(Integer)
    road_disruption_km: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20))
    severity_score: Mapped[float] = mapped_column(Float)
    confidence_pct: Mapped[float] = mapped_column(Float)
    water_fraction: Mapped[float] = mapped_column(Float, default=0)
    debris_fraction: Mapped[float] = mapped_column(Float, default=0)
    engine: Mapped[str] = mapped_column(String(60), default="classical-cv")
    engine_version: Mapped[str] = mapped_column(String(20), default="0.2")


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(12))   # user|assistant
    content: Mapped[str] = mapped_column(Text)
    answered_by: Mapped[str] = mapped_column(String(30), default="retrieval")  # llm|retrieval
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Report(Base, TimestampMixin):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    generated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    data_mode: Mapped[str] = mapped_column(String(10), default="demo")


class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    service: Mapped[str] = mapped_column(String(160))
    number: Mapped[str] = mapped_column(String(40))
    scope: Mapped[str] = mapped_column(String(40), default="state")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
