from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_roles
from app.database.session import get_db
from app.models import Location, Resource
from app.schemas import AllocationRequest
from app.services import allocation, risk_engine
from app.services.observations import observation_for
from app.services.vulnerability import assess

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("")
def list_resources(db: Session = Depends(get_db), _=Depends(current_user)):
    names = {l.id: l.name for l in db.scalars(select(Location)).all()}
    return [{"code": r.code, "type": r.resource_type, "unit": r.unit,
             "total": r.total_qty, "available": r.available_qty,
             "district": names.get(r.location_id), "status": r.status,
             "priority": r.priority, "lat": r.lat, "lon": r.lon}
            for r in db.scalars(select(Resource)).all()]


@router.post("/allocate")
def allocate(body: AllocationRequest, db: Session = Depends(get_db),
             user=Depends(require_roles("admin", "dma"))):
    loc = db.scalar(select(Location).where(Location.code == body.location_code))
    if loc is None:
        raise HTTPException(404, f"No district with code '{body.location_code}'")

    risk = risk_engine.score(observation_for(db, loc))
    vuln = assess(loc, risk)
    proposal = allocation.propose(db, loc, risk, vuln)

    if body.commit:
        allocation.commit(db, proposal, user.id)
        db.commit()
        proposal["status"] = "committed"
    else:
        proposal["status"] = "proposed"
    return proposal
