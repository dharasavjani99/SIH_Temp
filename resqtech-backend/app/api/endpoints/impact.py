from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.session import get_db
from app.dl import segmentation
from app.models import ImpactAssessment, Location

router = APIRouter(prefix="/impact", tags=["impact"])

MAX_BYTES = 12 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/webp"}


async def _read(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Upload a PNG, JPEG or WebP image.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Keep the image under 12 MB.")
    if not data:
        raise HTTPException(422, "The file is empty.")
    return data


@router.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  location_code: str | None = Query(default=None),
                  footprint_km2: float = Query(default=96.0, gt=0, le=10000),
                  db: Session = Depends(get_db),
                  user=Depends(require_roles("admin", "dma"))):
    data = await _read(file)
    result = segmentation.analyse(data, footprint_km2)

    loc = (db.scalar(select(Location).where(Location.code == location_code))
           if location_code else None)
    row = ImpactAssessment(
        location_id=loc.id if loc else None, source_image=file.filename,
        scene_footprint_km2=footprint_km2,
        affected_area_km2=result["affected_area_km2"],
        damaged_structures=result["damaged_structures"],
        road_disruption_km=result["road_disruption_km"],
        severity=result["severity"], severity_score=result["severity_score"],
        confidence_pct=result["confidence_pct"],
        water_fraction=result["fractions"]["water"],
        debris_fraction=result["fractions"]["debris"],
        engine=result["engine"], engine_version=result["engine_version"])
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, **result}


@router.post("/overlay", response_class=Response)
async def overlay(file: UploadFile = File(...), _=Depends(require_roles("admin", "dma"))):
    """Returns the segmentation overlay as a PNG for the before/after view."""
    data = await _read(file)
    return Response(content=segmentation.mask_png(data), media_type="image/png")
