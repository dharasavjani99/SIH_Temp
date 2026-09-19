"""Fetches the newest observation per district through the adapter layer and
merges any caller-supplied override into a RawObservation for the engines."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.features import RawObservation
from app.models import Location, WeatherData


def latest_for(db: Session, location: Location) -> WeatherData | None:
    return db.scalars(
        select(WeatherData)
        .where(WeatherData.location_id == location.id)
        .order_by(WeatherData.observed_at.desc())
        .limit(1)
    ).first()


def observation_for(db: Session, location: Location, override: dict | None = None) -> RawObservation:
    w = latest_for(db, location)
    obs = RawObservation(
        rain_24h_mm=w.rain_24h_mm if w else 0.0,
        rain_6h_mm=w.rain_6h_mm if w else 0.0,
        river_level_m=(w.river_level_m if w and w.river_level_m is not None else 0.0),
        river_danger_m=location.river_danger_m or 8.0,
        soil_moisture_index=location.soil_moisture_index,
        elevation_m=location.elevation_m,
        slope_deg=location.slope_deg,
        humidity_pct=w.humidity_pct if w else 70.0,
        wind_kmh=w.wind_kmh if w else 20.0,
        temp_c=w.temp_c if w else 28.0,
        coastal=location.coastal,
    )
    for key, value in (override or {}).items():
        if value is not None and hasattr(obs, key):
            setattr(obs, key, value)
    return obs
