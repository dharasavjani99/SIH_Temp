#!/usr/bin/env python
"""Load the CSVs in data/ into the database.

Idempotent: re-running updates existing rows instead of duplicating them.
Everything loaded here is demonstration data and is stored with is_live=False.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select                                    # noqa: E402

from app.core.security import hash_password                      # noqa: E402
from app.database.session import SessionLocal, init_db           # noqa: E402
from app.models import (Alert, Disaster, EmergencyContact, EmergencyTeam,      # noqa: E402
                        Hospital, Location, Resource, Road, RoadNode, Shelter,
                        User, WeatherData)
from app.services import risk_engine                             # noqa: E402
from app.services.observations import observation_for            # noqa: E402
from app.services.vulnerability import assess                    # noqa: E402

DATA = ROOT / "data"


def rows(name: str):
    with (DATA / name).open() as fh:
        yield from csv.DictReader(fh)


def upsert(db, model, key_field: str, key_value, **fields):
    obj = db.scalar(select(model).where(getattr(model, key_field) == key_value))
    if obj is None:
        obj = model(**{key_field: key_value}, **fields)
        db.add(obj)
    else:
        for k, v in fields.items():
            setattr(obj, k, v)
    return obj


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        # ---- districts ----
        for r in rows("districts.csv"):
            upsert(db, Location, "code", r["code"],
                   name=r["name"], state=r["state"], lat=float(r["lat"]), lon=float(r["lon"]),
                   population=int(r["population"]),
                   exposure_band_population=int(r["exposure_band_population"]),
                   elevation_m=float(r["elevation_m"]), slope_deg=float(r["slope_deg"]),
                   soil_moisture_index=float(r["soil_moisture_index"]),
                   river_name=r["river_name"] or None,
                   river_danger_m=float(r["river_danger_m"]) if r["river_danger_m"] else None,
                   coastal=r["coastal"] == "1", access_index=float(r["access_index"]),
                   hospital_count=int(r["hospital_count"]), school_count=int(r["school_count"]),
                   elderly_share=float(r["elderly_share"]), child_share=float(r["child_share"]))
        db.commit()
        loc_by_code = {l.code: l for l in db.scalars(select(Location)).all()}

        # ---- observations ----
        now = datetime.now(timezone.utc)
        for r in rows("weather_observations.csv"):
            loc = loc_by_code[r["location_code"]]
            exists = db.scalar(select(WeatherData).where(
                WeatherData.location_id == loc.id, WeatherData.source == r["source"]))
            if exists is None:
                db.add(WeatherData(
                    location_id=loc.id, observed_at=now - timedelta(minutes=12),
                    rain_6h_mm=float(r["rain_6h_mm"]), rain_24h_mm=float(r["rain_24h_mm"]),
                    temp_c=float(r["temp_c"]), humidity_pct=float(r["humidity_pct"]),
                    wind_kmh=float(r["wind_kmh"]), river_level_m=float(r["river_level_m"]),
                    source=r["source"], is_live=False))
        db.commit()

        # ---- facilities, stock, network ----
        for r in rows("shelters.csv"):
            upsert(db, Shelter, "code", r["code"], name=r["name"],
                   location_id=loc_by_code[r["location_code"]].id,
                   lat=float(r["lat"]), lon=float(r["lon"]),
                   capacity=int(r["capacity"]), occupancy=int(r["occupancy"]),
                   facilities=r["facilities"].split("|"))
        for r in rows("hospitals.csv"):
            upsert(db, Hospital, "code", r["code"], name=r["name"],
                   location_id=loc_by_code[r["location_code"]].id,
                   lat=float(r["lat"]), lon=float(r["lon"]),
                   beds_total=int(r["beds_total"]), beds_free=int(r["beds_free"]),
                   icu_free=int(r["icu_free"]))
        for r in rows("resources.csv"):
            upsert(db, Resource, "code", r["code"], resource_type=r["resource_type"],
                   unit=r["unit"], total_qty=int(r["total_qty"]),
                   available_qty=int(r["available_qty"]),
                   location_id=loc_by_code[r["location_code"]].id,
                   lat=float(r["lat"]), lon=float(r["lon"]),
                   status=r["status"], priority=r["priority"])
        for r in rows("road_nodes.csv"):
            upsert(db, RoadNode, "code", r["code"], name=r["name"],
                   lat=float(r["lat"]), lon=float(r["lon"]))
        db.commit()

        for r in rows("road_edges.csv"):
            edge = db.scalar(select(Road).where(Road.from_node == r["from_node"],
                                                Road.to_node == r["to_node"]))
            if edge is None:
                db.add(Road(name=r["name"], from_node=r["from_node"], to_node=r["to_node"],
                            condition=r["condition"], hazard_score=float(r["hazard_score"])))
            else:
                edge.condition, edge.hazard_score = r["condition"], float(r["hazard_score"])

        for r in rows("emergency_teams.csv"):
            upsert(db, EmergencyTeam, "code", r["code"], name=r["name"], force=r["force"],
                   members=int(r["members"]), lat=float(r["lat"]), lon=float(r["lon"]),
                   state=r["state"], current_task=r["current_task"])

        for r in rows("emergency_contacts.csv"):
            existing = db.scalar(select(EmergencyContact)
                                 .where(EmergencyContact.service == r["service"]))
            if existing is None:
                db.add(EmergencyContact(service=r["service"], number=r["number"],
                                        scope=r["scope"], sort_order=int(r["sort_order"])))
        db.commit()

        # ---- demo accounts ----
        for r in rows("demo_users.csv"):
            upsert(db, User, "email", r["email"].lower(), full_name=r["full_name"],
                   role=r["role"], password_hash=hash_password(r["password"]),
                   district_id=loc_by_code[r["district_code"]].id if r["district_code"] else None,
                   is_active=True)
        db.commit()

        # ---- derive alerts and disasters from the engine, not from a fixture ----
        grade_for = lambda p: ("emergency" if p >= 76 else "warning" if p >= 58
                               else "watch" if p >= 46 else "advisory")
        actions = {
            "emergency": "Move residents of low-lying wards to shelters now; pre-position boats and medical teams.",
            "warning": "Prepare evacuation resources and open designated shelters; suspend riverbank activity.",
            "watch": "Place response teams on standby and verify shelter readiness.",
            "advisory": "Monitor gauges and issue a public advisory through district channels.",
        }
        created = 0
        for loc in db.scalars(select(Location)).all():
            risk = risk_engine.score(observation_for(db, loc))
            if risk["composite_pct"] < 38:
                continue
            vuln = assess(loc, risk)
            hazard, prob = max(risk["hazards"].items(), key=lambda kv: kv[1])
            grade = grade_for(risk["composite_pct"])
            if db.scalar(select(Alert).where(Alert.location_id == loc.id,
                                             Alert.hazard == hazard)) is None:
                db.add(Alert(location_id=loc.id, hazard=hazard, grade=grade,
                             probability_pct=prob,
                             expected_duration_h=int(6 + risk["composite_pct"] / 12),
                             recommended_action=actions[grade],
                             channels={"public": grade in ("warning", "emergency"),
                                       "authorities": True},
                             status="issued", issued_at=now))
                created += 1
            if risk["composite_pct"] >= 58 and db.scalar(
                    select(Disaster).where(Disaster.location_id == loc.id,
                                           Disaster.status == "active")) is None:
                db.add(Disaster(location_id=loc.id, hazard=hazard, severity=risk["level"],
                                status="active", started_at=now - timedelta(hours=8),
                                description=(f"{hazard.title()} event, {vuln['exposed_total']:,} "
                                             f"people in the exposure band.")))
        db.commit()

        print(f"seeded: {len(loc_by_code)} districts, "
              f"{db.scalar(select(__import__('sqlalchemy').func.count()).select_from(Shelter))} shelters, "
              f"{created} alerts derived from the risk engine")
    finally:
        db.close()


if __name__ == "__main__":
    main()
