"""Transparent weighted-logistic surrogate.

This is the fallback the API serves when no trained model artifact is present,
and it is also the model that produced the figures in the frontend prototype.
It is deterministic and fully auditable — every score decomposes into per
feature log-odds contributions — but it is *not* learned from data. Responses
that come from here are labelled model_name="surrogate" so the UI can say so.
"""
from __future__ import annotations

import math

from app.ml.features import FEATURE_LABELS, HAZARDS

WEIGHTS: dict[str, dict[str, float]] = {
    "flood":     {"bias": -5.4, "rain24": 1.9, "rain6": 1.4, "river_ratio": 2.4,
                  "soil": 0.9, "low_elev": 1.0, "flat": 0.5, "humidity": 0.3},
    "landslide": {"bias": -5.2, "rain24": 1.2, "rain6": 1.1, "soil": 1.4,
                  "slope": 2.6, "low_elev": -0.3, "humidity": 0.25},
    "cyclone":   {"bias": -4.4, "wind": 3.4, "coastal": 1.5, "humidity": 0.6, "rain6": 0.5},
    "rainfall":  {"bias": -3.2, "rain24": 2.6, "rain6": 2.4, "humidity": 1.0, "temp": -0.5},
}

VERSION = "surrogate-0.4"


def _logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def predict_hazard(hazard: str, features: dict[str, float]) -> tuple[float, list[dict], float]:
    """Return (probability 0-1, contributions, raw log-odds)."""
    w = WEIGHTS[hazard]
    z = w["bias"]
    contributions = []
    for key, weight in w.items():
        if key == "bias":
            continue
        c = weight * features.get(key, 0.0)
        z += c
        contributions.append({"feature": key, "label": FEATURE_LABELS[key], "log_odds": round(c, 4)})
    contributions.sort(key=lambda d: d["log_odds"], reverse=True)
    return _logistic(z), contributions, z


def predict_all(features: dict[str, float]) -> dict:
    out, logits = {}, {}
    contributions = {}
    for hazard in HAZARDS:
        p, contrib, z = predict_hazard(hazard, features)
        out[hazard] = p
        logits[hazard] = z
        contributions[hazard] = contrib
    return {"probabilities": out, "contributions": contributions, "logits": logits}
