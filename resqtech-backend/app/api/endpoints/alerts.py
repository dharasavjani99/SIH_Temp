from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_user, require_roles
from app.database.session import get_db
from app.models import Alert, Location
from app.schemas import AlertCreate, AlertOut

router = APIRouter(prefix="/alerts", tags=["alerts"])

GRADE_FLOOR = {"advisory": 0, "watch": 46, "warning": 58, "emergency": 76}


@router.get("", response_model=list[AlertOut])
def list_alerts(db: Session = Depends(get_db), _=Depends(current_user)):
    rows = db.scalars(select(Alert).order_by(Alert.probability_pct.desc())).all()
    names = {l.id: l.name for l in db.scalars(select(Location)).all()}
    return [AlertOut(id=a.id, location=names.get(a.location_id, "—"), hazard=a.hazard,
                     grade=a.grade, probability_pct=a.probability_pct,
                     expected_duration_h=a.expected_duration_h,
                     recommended_action=a.recommended_action, status=a.status,
                     issued_at=a.issued_at) for a in rows]


@router.post("", response_model=AlertOut, status_code=201)
def create_alert(body: AlertCreate, db: Session = Depends(get_db),
                 user=Depends(require_roles("admin", "dma"))):
    loc = db.scalar(select(Location).where(Location.code == body.location_code))
    if loc is None:
        raise HTTPException(404, f"No district with code '{body.location_code}'")
    if body.probability_pct < GRADE_FLOOR[body.grade]:
        raise HTTPException(
            422, f"A {body.grade} needs at least {GRADE_FLOOR[body.grade]}% probability; "
                 f"this alert is at {body.probability_pct}%. Pick a lower grade.")

    alert = Alert(location_id=loc.id, hazard=body.hazard, grade=body.grade,
                  probability_pct=body.probability_pct,
                  expected_duration_h=body.expected_duration_h,
                  recommended_action=body.recommended_action,
                  issued_by=user.id, issued_at=datetime.now(timezone.utc),
                  channels={"public": body.notify_public, "authorities": body.notify_authorities},
                  status="issued")
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return AlertOut(id=alert.id, location=loc.name, hazard=alert.hazard, grade=alert.grade,
                    probability_pct=alert.probability_pct,
                    expected_duration_h=alert.expected_duration_h,
                    recommended_action=alert.recommended_action, status=alert.status,
                    issued_at=alert.issued_at)
