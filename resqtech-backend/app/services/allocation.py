"""Resource allocation.

Demand is derived from district relief-manual planning factors applied to the
evacuation caseload, scaled by an urgency multiplier, then capped by what is
actually on the shelf. Shortfalls are reported, never silently absorbed.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Hospital, Location, Resource

PLANNING_FACTORS = {          # units required per 1,000 people evacuating
    "Ambulance": 0.55,
    "Rescue boat": 0.32,
    "NDRF team": 0.18,
    "Water tanker": 1.40,
    "Medical kit": 120.0,
    "Food packet": 380.0,
}


def _pool(db: Session, resource_type: str) -> int:
    return int(db.scalar(
        select(func.coalesce(func.sum(Resource.available_qty), 0))
        .where(Resource.resource_type == resource_type)
    ) or 0)


def propose(db: Session, location: Location, risk: dict, vulnerability: dict) -> dict:
    caseload = vulnerability["evacuation_required"] / 1000
    urgency = 1 + risk["composite_pct"] / 110
    flood_factor = max(0.2, risk["hazards"]["flood"] / 60)

    lines = []
    for rtype, factor in PLANNING_FACTORS.items():
        need = caseload * factor * urgency
        if rtype == "Rescue boat":
            need *= flood_factor
        need = int(-(-need // 1))                 # ceil
        pool = _pool(db, rtype)
        assign = min(need, pool)
        lines.append({
            "resource_type": rtype,
            "required_qty": need,
            "assigned_qty": assign,
            "pool_qty": pool,
            "shortfall_qty": max(need - assign, 0),
        })

    free_beds = int(db.scalar(
        select(func.coalesce(func.sum(Hospital.beds_free), 0))
        .where(Hospital.location_id == location.id)) or 0)

    rationale = [
        f"{vulnerability['exposed_total']:,} people sit inside the modelled exposure band",
        f"{risk['level']} composite risk at {risk['composite_pct']}% gives an urgency multiplier of {urgency:.2f}",
        f"{vulnerability['evacuation_required']:,} residents meet the evacuation threshold, which sets base demand",
        f"accessibility index {location.access_index:.2f} — "
        + ("poor road reach inflates the team requirement" if location.access_index < 0.6
           else "road reach is workable"),
        f"{location.hospital_count} hospitals in district with {free_beds:,} free beds",
    ]

    return {
        "location": location.name,
        "location_id": location.id,
        "urgency_multiplier": round(urgency, 2),
        "lines": lines,
        "shortfalls": [l for l in lines if l["shortfall_qty"] > 0],
        "rationale": rationale,
    }


def commit(db: Session, proposal: dict, user_id: int | None = None) -> list:
    """Decrement stock and write allocation rows. Caller commits the session."""
    from app.models import ResourceAllocation

    written = []
    for line in proposal["lines"]:
        remaining = line["assigned_qty"]
        if remaining <= 0:
            continue
        rows = db.scalars(
            select(Resource)
            .where(Resource.resource_type == line["resource_type"])
            .order_by(Resource.available_qty.desc())
        ).all()
        for r in rows:
            take = min(r.available_qty, remaining)
            if take:
                r.available_qty -= take
                r.status = "deployed"
                remaining -= take
            if remaining <= 0:
                break
        alloc = ResourceAllocation(
            location_id=proposal["location_id"],
            resource_type=line["resource_type"],
            required_qty=line["required_qty"],
            assigned_qty=line["assigned_qty"],
            shortfall_qty=line["shortfall_qty"],
            rationale={"reasons": proposal["rationale"]},
            approved_by=user_id,
            status="committed",
        )
        db.add(alloc)
        written.append(alloc)
    return written
