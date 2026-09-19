from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.database.session import get_db
from app.models import Hospital, Location, Road, RoadNode, Shelter
from app.schemas import RouteRequest
from app.services import routing

router = APIRouter(tags=["logistics"])


@router.get("/roads")
def roads(db: Session = Depends(get_db), _=Depends(current_user)):
    return {
        "nodes": [{"code": n.code, "name": n.name, "lat": n.lat, "lon": n.lon}
                  for n in db.scalars(select(RoadNode)).all()],
        "edges": [{"name": r.name, "from": r.from_node, "to": r.to_node,
                   "condition": r.condition, "hazard_pct": round(r.hazard_score * 100)}
                  for r in db.scalars(select(Road)).all()],
    }


@router.post("/routes/optimize")
def optimize(body: RouteRequest, db: Session = Depends(get_db), _=Depends(current_user)):
    route = routing.solve(db, body.from_node, body.to_node, body.objective, body.vehicle)
    if route is None:
        raise HTTPException(
            422, "No open path between those points for this vehicle. Try a different "
                 "objective, or route air assets.")
    if body.include_alternatives:
        route["alternatives"] = routing.alternatives(db, body.from_node, body.to_node,
                                                     route, body.vehicle)
    return route


@router.get("/shelters")
def shelters(db: Session = Depends(get_db), _=Depends(current_user)):
    names = {l.id: l.name for l in db.scalars(select(Location)).all()}
    rows = db.scalars(select(Shelter)).all()
    return [{"code": s.code, "name": s.name, "district": names.get(s.location_id),
             "lat": s.lat, "lon": s.lon, "capacity": s.capacity,
             "occupancy": s.occupancy, "available": s.available,
             "occupancy_pct": round(s.occupancy / s.capacity * 100) if s.capacity else 0,
             "facilities": s.facilities, "is_open": s.is_open} for s in rows]


@router.get("/shelters/nearest")
def nearest_shelter(lat: float, lon: float, min_free: int = 1,
                    db: Session = Depends(get_db), _=Depends(current_user)):
    candidates = [s for s in db.scalars(select(Shelter)).all() if s.available >= min_free]
    if not candidates:
        raise HTTPException(404, "Every shelter is at capacity. Open an overflow site.")
    best = min(candidates, key=lambda s: routing.haversine_km(lat, lon, s.lat, s.lon))
    return {"code": best.code, "name": best.name, "available": best.available,
            "capacity": best.capacity, "facilities": best.facilities,
            "distance_km": round(routing.haversine_km(lat, lon, best.lat, best.lon), 1)}


@router.get("/hospitals")
def hospitals(db: Session = Depends(get_db), _=Depends(current_user)):
    names = {l.id: l.name for l in db.scalars(select(Location)).all()}
    return [{"code": h.code, "name": h.name, "district": names.get(h.location_id),
             "lat": h.lat, "lon": h.lon, "beds_total": h.beds_total,
             "beds_free": h.beds_free, "icu_free": h.icu_free}
            for h in db.scalars(select(Hospital)).all()]
