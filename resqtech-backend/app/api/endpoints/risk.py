from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_roles
from app.core.config import get_settings
from app.database.session import get_db
from app.ml.registry import get_registry
from app.models import Location, RiskPrediction
from app.schemas import RiskRequest, RiskResponse
from app.services import risk_engine
from app.services.observations import observation_for

router = APIRouter(prefix="/risk", tags=["risk"])


def _location(db: Session, code: str) -> Location:
    loc = db.scalar(select(Location).where(Location.code == code))
    if loc is None:
        raise HTTPException(404, f"No district with code '{code}'")
    return loc


@router.get("", summary="Current risk for every district")
def all_risk(db: Session = Depends(get_db), _=Depends(current_user)):
    out = []
    for loc in db.scalars(select(Location)).all():
        r = risk_engine.score(observation_for(db, loc))
        out.append({"code": loc.code, "name": loc.name, "lat": loc.lat, "lon": loc.lon,
                    "hazards": r["hazards"], "composite_pct": r["composite_pct"],
                    "level": r["level"], "confidence_pct": r["confidence_pct"]})
    return {"data_mode": get_settings().data_mode, "model": out[0] if False else
            get_registry().version, "districts": out}


@router.post("/predict", response_model=RiskResponse)
def predict(body: RiskRequest, db: Session = Depends(get_db),
            user=Depends(require_roles("admin", "dma"))):
    loc = _location(db, body.location_code)
    override = body.model_dump(exclude={"location_code", "persist"})
    obs = observation_for(db, loc, override)
    result = risk_engine.score(obs)

    if body.persist:
        db.add(RiskPrediction(
            location_id=loc.id, horizon_hours=0,
            flood_pct=result["hazards"]["flood"], landslide_pct=result["hazards"]["landslide"],
            cyclone_pct=result["hazards"]["cyclone"], rainfall_pct=result["hazards"]["rainfall"],
            composite_pct=result["composite_pct"], level=result["level"],
            confidence_pct=result["confidence_pct"], model_name=result["model_name"],
            model_version=result["model_version"], features=result["features"],
            contributions=result["contributions"]))
        db.commit()

    return RiskResponse(
        location=loc.name, location_code=loc.code,
        hazards=result["hazards"], composite_pct=result["composite_pct"],
        level=result["level"], confidence_pct=result["confidence_pct"],
        model_name=result["model_name"], model_version=result["model_version"],
        is_trained_model=get_registry().loaded,
        features=result["features"], contributions=result["contributions"],
        explanation_source=result["explanation_source"],
        data_mode=get_settings().data_mode,
        generated_at=datetime.now(timezone.utc))


@router.get("/{code}/timeline", summary="Projected risk over 6-72 hours")
def timeline(code: str, db: Session = Depends(get_db), _=Depends(current_user)):
    loc = _location(db, code)
    return {"location": loc.name,
            "horizons": risk_engine.project(observation_for(db, loc)),
            "note": "Physically motivated projection envelope, not a trained sequence model."}


@router.get("/model/info", summary="What model is actually serving predictions")
def model_info():
    reg = get_registry()
    return {
        "trained_model_loaded": reg.loaded,
        "version": reg.version,
        "metrics": reg.metadata.get("metrics"),
        "fallback": "logistic surrogate (app/ml/surrogate.py)",
        "warning": reg.metadata.get("label_provenance"),
    }
