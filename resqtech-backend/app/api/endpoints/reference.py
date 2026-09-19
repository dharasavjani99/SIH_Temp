from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.bhuvan import BhuvanAdapter
from app.api.deps import current_user
from app.core.config import get_settings
from app.database.session import get_db
from app.models import Disaster, EmergencyContact, EmergencyTeam, Location
from app.schemas import LocationOut
from app.services import risk_engine
from app.services.observations import observation_for
from app.services.vulnerability import assess

router = APIRouter(tags=["reference"])


@router.get("/locations", response_model=list[LocationOut])
def locations(db: Session = Depends(get_db), _=Depends(current_user)):
    return db.scalars(select(Location).order_by(Location.name)).all()


@router.get("/disasters")
def disasters(db: Session = Depends(get_db), _=Depends(current_user)):
    names = {l.id: l.name for l in db.scalars(select(Location)).all()}
    return [{"id": d.id, "district": names.get(d.location_id), "hazard": d.hazard,
             "severity": d.severity, "status": d.status, "started_at": d.started_at,
             "description": d.description}
            for d in db.scalars(select(Disaster)).all()]


@router.get("/vulnerable")
def vulnerable(db: Session = Depends(get_db), _=Depends(current_user)):
    out = []
    for loc in db.scalars(select(Location)).all():
        risk = risk_engine.score(observation_for(db, loc))
        out.append({"code": loc.code, "district": loc.name,
                    "composite_pct": risk["composite_pct"], "level": risk["level"],
                    "access_index": loc.access_index, **assess(loc, risk)})
    out.sort(key=lambda d: d["priority_score"], reverse=True)
    return out


@router.get("/teams")
def teams(db: Session = Depends(get_db), _=Depends(current_user)):
    return [{"code": t.code, "name": t.name, "force": t.force, "members": t.members,
             "lat": t.lat, "lon": t.lon, "state": t.state, "task": t.current_task}
            for t in db.scalars(select(EmergencyTeam)).all()]


@router.get("/contacts")
def contacts(db: Session = Depends(get_db)):
    """Public on purpose — emergency numbers must work without a login."""
    rows = db.scalars(select(EmergencyContact).order_by(EmergencyContact.sort_order)).all()
    return [{"service": c.service, "number": c.number, "scope": c.scope} for c in rows]


@router.get("/sources")
def sources(_=Depends(current_user)):
    """What each feed is and whether it is live right now."""
    s = get_settings()
    return {
        "data_mode": s.data_mode,
        "sources": [
            {"id": "imd", "name": "IMD Mausam — rainfall and nowcast",
             "homepage": "https://mausam.imd.gov.in", "live": bool(s.imd_api_key) and s.data_mode == "live"},
            {"id": "cwc", "name": "Central Water Commission — river gauges",
             "homepage": "https://cwc.gov.in", "live": bool(s.cwc_api_key) and s.data_mode == "live"},
            {"id": "bhuvan", "name": "ISRO Bhuvan — satellite and flood layers",
             "homepage": "https://bhuvan-app1.nrsc.gov.in/bhuvandisaster",
             "live": bool(s.bhuvan_api_key) and s.data_mode == "live"},
            {"id": "ndma", "name": "NDMA — advisories and SOPs",
             "homepage": "https://ndma.gov.in", "live": False},
        ],
        "map_layers": BhuvanAdapter().layers(),
        "disclaimer": ("Demo mode serves seeded data. Nothing here is an official Government "
                       "of India advisory."),
    }
