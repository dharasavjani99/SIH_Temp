from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.config import get_settings
from app.database.session import get_db
from app.models import (Alert, EmergencyTeam, Location, Resource, Road, Shelter)
from app.services import risk_engine
from app.services.observations import observation_for
from app.services.vulnerability import assess

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), _=Depends(current_user)):
    districts, exposed_total, high_zones = [], 0, 0
    for loc in db.scalars(select(Location)).all():
        risk = risk_engine.score(observation_for(db, loc))
        vuln = assess(loc, risk)
        exposed_total += vuln["exposed_total"]
        high_zones += 1 if risk["composite_pct"] >= 55 else 0
        districts.append({"code": loc.code, "name": loc.name, "lat": loc.lat, "lon": loc.lon,
                          "composite_pct": risk["composite_pct"], "level": risk["level"],
                          "hazards": risk["hazards"], **vuln})

    districts.sort(key=lambda d: d["priority_score"], reverse=True)

    blocked = db.scalar(select(func.count()).select_from(Road)
                        .where(Road.condition.in_(("flooded", "landslide")))) or 0
    shelter_free = db.scalar(
        select(func.coalesce(func.sum(Shelter.capacity - Shelter.occupancy), 0))) or 0
    deployable = db.scalar(
        select(func.coalesce(func.sum(Resource.available_qty), 0))
        .where(Resource.unit.in_(("units", "teams", "boats", "tankers", "sets")))) or 0
    teams_on_task = db.scalar(select(func.count()).select_from(EmergencyTeam)
                              .where(EmergencyTeam.state == "on_task")) or 0
    active_alerts = db.scalar(select(func.count()).select_from(Alert)
                              .where(Alert.grade.in_(("warning", "emergency")))) or 0

    return {
        "data_mode": get_settings().data_mode,
        "headline": {
            "active_disasters": int(active_alerts),
            "high_risk_zones": high_zones,
            "people_at_risk": exposed_total,
            "resources_available": int(deployable),
            "teams_on_task": int(teams_on_task),
            "roads_impassable": int(blocked),
            "shelter_spaces_free": int(shelter_free),
        },
        "priority_districts": districts[:6],
        "districts": districts,
    }
