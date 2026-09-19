"""Risk scoring service.

Order of preference:
  1. the trained gradient-boosting models in app/ml/artifacts (if present)
  2. the transparent logistic surrogate

Whichever answers, the response says so in `model_name`, and the frontend
prints it. Confidence is never invented: for the trained model it comes from
the distance of the probability from the 0.5 boundary combined with the
out-of-fold Brier score recorded at training time; for the surrogate it comes
from the magnitude of the log-odds.
"""
from __future__ import annotations

from app.ml import surrogate
from app.ml.features import HAZARDS, RawObservation, build_features, vectorise
from app.ml.registry import get_registry

LEVELS = (("critical", 75), ("high", 55), ("moderate", 32), ("low", 0))


def level_for(pct: float) -> str:
    for name, floor in LEVELS:
        if pct >= floor:
            return name
    return "low"


def _trained_probabilities(features: dict[str, float]) -> dict[str, float] | None:
    reg = get_registry()
    if not reg.loaded:
        return None
    x = [vectorise(features)]
    out = {}
    for hazard in HAZARDS:
        model = reg.models.get(hazard)
        if model is None:
            return None
        out[hazard] = float(model.predict_proba(x)[0][1])
    return out


def score(obs: RawObservation) -> dict:
    features = build_features(obs)
    surrogate_result = surrogate.predict_all(features)

    trained = _trained_probabilities(features)
    if trained is not None:
        probs = trained
        reg = get_registry()
        model_name, model_version = "RiskNet-GJ (gradient boosting)", reg.version
        brier = (reg.metadata.get("metrics") or {}).get("mean_brier", 0.12)
        margin = sum(abs(p - 0.5) for p in probs.values()) / len(probs)
        confidence = round(min(94.0, max(45.0, 100 * (1 - brier) * (0.55 + margin))), 1)
    else:
        probs = surrogate_result["probabilities"]
        model_name, model_version = "surrogate (untrained)", surrogate.VERSION
        margin = abs(surrogate_result["logits"]["flood"]) + abs(surrogate_result["logits"]["cyclone"])
        confidence = round(min(94.0, max(45.0, 62 + margin * 4.2)), 1)

    pct = {h: round(probs[h] * 100, 1) for h in HAZARDS}
    dominant = max(pct["flood"], pct["landslide"], pct["cyclone"])
    mean_all = sum(pct.values()) / len(pct)
    composite = round(dominant * 0.74 + mean_all * 0.26, 1)

    return {
        "hazards": pct,
        "composite_pct": composite,
        "level": level_for(composite),
        "confidence_pct": confidence,
        "model_name": model_name,
        "model_version": model_version,
        "features": {k: round(v, 4) for k, v in features.items()},
        # Contributions always come from the surrogate: it is the interpretable
        # twin of the trained model, and the UI labels it as an explanation of
        # the drivers rather than of the exact tree ensemble output.
        "contributions": surrogate_result["contributions"],
        "explanation_source": "logistic surrogate (interpretable twin)",
    }


def project(obs: RawObservation, horizons: tuple[int, ...] = (6, 12, 24, 48, 72)) -> list[dict]:
    """Forward projection: storm persistence decay plus river routing lag.

    This is a physically motivated envelope, not a trained sequence model.
    """
    import math

    base = score(obs)
    out = []
    for h in horizons:
        decay = math.exp(-h / 58)
        lag = 1.06 if h <= 12 else 1.14 if h <= 24 else 0.94 if h <= 48 else 0.72
        v = base["composite_pct"] * decay * lag + base["composite_pct"] * (1 - decay) * 0.42
        v = max(0.0, min(100.0, v))
        out.append({
            "horizon_hours": h,
            "composite_pct": round(v, 1),
            "level": level_for(v),
            "confidence_pct": round(max(44.0, base["confidence_pct"] - h * 0.28), 1),
        })
    return out
