"""Feature engineering for the risk models.

One place defines the feature vector. The training script, the inference
service and the fallback surrogate all import FEATURE_ORDER from here, so a
model can never be served a differently ordered vector than it was fitted on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

FEATURE_ORDER = [
    "rain24", "rain6", "river_ratio", "soil", "low_elev",
    "flat", "humidity", "slope", "wind", "coastal", "temp",
]

FEATURE_LABELS = {
    "rain24": "24-hour rainfall",
    "rain6": "6-hour rainfall burst",
    "river_ratio": "river level against danger mark",
    "soil": "soil saturation",
    "low_elev": "low elevation",
    "flat": "flat terrain / poor drainage",
    "humidity": "humidity",
    "slope": "terrain slope",
    "wind": "sustained wind speed",
    "coastal": "coastal exposure",
    "temp": "temperature",
}

HAZARDS = ("flood", "landslide", "cyclone", "rainfall")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(slots=True)
class RawObservation:
    """Everything the engine needs about one place at one moment."""
    rain_24h_mm: float = 0.0
    rain_6h_mm: float = 0.0
    river_level_m: float = 0.0
    river_danger_m: float = 8.0
    soil_moisture_index: float = 0.5
    elevation_m: float = 50.0
    slope_deg: float = 2.0
    humidity_pct: float = 70.0
    wind_kmh: float = 20.0
    temp_c: float = 28.0
    coastal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def build_features(obs: RawObservation) -> dict[str, float]:
    """Normalise raw observations into the bounded feature space."""
    danger = obs.river_danger_m or 8.0
    return {
        "rain24": _clamp(obs.rain_24h_mm / 300.0, 0, 1.2),
        "rain6": _clamp(obs.rain_6h_mm / 140.0, 0, 1.2),
        "river_ratio": _clamp(obs.river_level_m / danger, 0, 1.3),
        "soil": _clamp(obs.soil_moisture_index, 0, 1),
        "low_elev": _clamp(1 - (obs.elevation_m / 400.0), 0, 1),
        "flat": _clamp(1 - (obs.slope_deg / 20.0), 0, 1),
        "humidity": _clamp((obs.humidity_pct - 50) / 50.0, 0, 1),
        "slope": _clamp(obs.slope_deg / 35.0, 0, 1),
        "wind": _clamp((obs.wind_kmh - 20) / 80.0, 0, 1),
        "coastal": 1.0 if obs.coastal else 0.0,
        "temp": _clamp((obs.temp_c - 20) / 20.0, 0, 1),
    }


def vectorise(features: dict[str, float]) -> list[float]:
    return [float(features[k]) for k in FEATURE_ORDER]
