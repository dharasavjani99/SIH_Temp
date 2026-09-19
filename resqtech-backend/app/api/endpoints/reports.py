import csv
import io
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_roles
from app.core.config import get_settings
from app.database.session import get_db
from app.models import Location, Report, Road, Shelter
from app.services import risk_engine
from app.services.observations import observation_for
from app.services.vulnerability import assess

router = APIRouter(prefix="/reports", tags=["reports"])

COLUMNS = ["district", "river", "flood_pct", "landslide_pct", "cyclone_pct", "rainfall_pct",
           "composite_pct", "level", "confidence_pct", "exposed", "high_priority",
           "evacuation_required", "vulnerability_index", "access_index", "priority_score"]


def _rows(db: Session) -> list[dict]:
    out = []
    for loc in db.scalars(select(Location)).all():
        r = risk_engine.score(observation_for(db, loc))
        v = assess(loc, r)
        out.append({
            "district": loc.name, "river": loc.river_name or "",
            "flood_pct": r["hazards"]["flood"], "landslide_pct": r["hazards"]["landslide"],
            "cyclone_pct": r["hazards"]["cyclone"], "rainfall_pct": r["hazards"]["rainfall"],
            "composite_pct": r["composite_pct"], "level": r["level"],
            "confidence_pct": r["confidence_pct"], "exposed": v["exposed_total"],
            "high_priority": v["high_priority"],
            "evacuation_required": v["evacuation_required"],
            "vulnerability_index": v["vulnerability_index"],
            "access_index": loc.access_index, "priority_score": v["priority_score"]})
    out.sort(key=lambda d: d["priority_score"], reverse=True)
    return out


@router.get("")
def situation(db: Session = Depends(get_db), _=Depends(current_user)):
    rows = _rows(db)
    free = sum(s.available for s in db.scalars(select(Shelter)).all())
    degraded = [r.name for r in db.scalars(select(Road)).all() if r.condition != "open"]
    return {
        "generated_at": datetime.now(timezone.utc),
        "data_mode": get_settings().data_mode,
        "headline_district": rows[0]["district"] if rows else None,
        "people_exposed": sum(r["exposed"] for r in rows),
        "evacuation_required": sum(r["evacuation_required"] for r in rows),
        "shelter_spaces_free": free,
        "degraded_links": degraded,
        "districts": rows,
        "prediction_accuracy": None,
        "accuracy_note": ("No validated backtest exists for this model. An accuracy figure "
                          "is deliberately not reported rather than estimated."),
    }


@router.post("/generate")
def generate(db: Session = Depends(get_db), user=Depends(require_roles("admin", "dma"))):
    payload = situation(db, user)
    now = datetime.now(timezone.utc)
    report = Report(title=f"Situation report — {now:%d %b %Y %H:%M} UTC",
                    period_start=now - timedelta(hours=12), period_end=now,
                    generated_by=user.id, payload=payload,
                    data_mode=get_settings().data_mode)
    db.add(report)
    db.commit()
    db.refresh(report)
    return {"id": report.id, "title": report.title, **payload}


@router.get("/export.csv", response_class=Response)
def export_csv(db: Session = Depends(get_db), _=Depends(current_user)):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(_rows(db))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="resqtech-situation-{stamp}.csv"'})
