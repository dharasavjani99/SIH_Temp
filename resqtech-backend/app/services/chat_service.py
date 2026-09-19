"""ResQ AI.

The context pack is assembled here, server-side, from the database. The LLM
key never leaves the backend. If no key is configured or the provider call
fails, a deterministic retrieval answer is returned instead and the response
says answered_by="retrieval" so the UI can be honest about it.
"""
from __future__ import annotations

import json
import logging

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (Alert, EmergencyContact, Hospital, Location, Resource,
                        Road, RoadNode, Shelter)
from app.services import risk_engine
from app.services.observations import observation_for
from app.services.vulnerability import assess

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ResQ AI, the assistant inside ResQTech, a disaster
management platform for Gujarat, India. Answer only from the JSON state given
to you. Be concise: two to five short sentences or a tight list, no headings.
Quote specific districts, percentages, counts and shelter names from the data.
If the data does not contain the answer, say so plainly. Never invent figures.
All figures are simulated demonstration data; if asked whether this is real,
say it is a prototype seeded with demo data, not an official government feed.
For safety questions give practical NDMA-aligned guidance and tell the user to
follow local authority instructions and call 1070 or 112 in an emergency."""


def build_context(db: Session) -> dict:
    settings = get_settings()
    districts = []
    for loc in db.scalars(select(Location)).all():
        risk = risk_engine.score(observation_for(db, loc))
        vuln = assess(loc, risk)
        districts.append({
            "district": loc.name, "composite_pct": risk["composite_pct"],
            "level": risk["level"], **risk["hazards"],
            "exposed": vuln["exposed_total"], "evacuate": vuln["evacuation_required"],
            "river": loc.river_name, "access_index": loc.access_index,
        })

    return {
        "data_mode": settings.data_mode,
        "districts": districts,
        "alerts": [{"district": a.location_id, "hazard": a.hazard, "grade": a.grade,
                    "probability_pct": a.probability_pct, "action": a.recommended_action}
                   for a in db.scalars(select(Alert)).all()],
        "shelters": [{"name": s.name, "capacity": s.capacity, "occupied": s.occupancy,
                      "free": s.available, "facilities": s.facilities}
                     for s in db.scalars(select(Shelter)).all()],
        "resources": [{"type": r.resource_type, "available": r.available_qty,
                       "total": r.total_qty, "unit": r.unit, "status": r.status}
                      for r in db.scalars(select(Resource)).all()],
        "hospitals": [{"name": h.name, "free_beds": h.beds_free, "icu_free": h.icu_free}
                      for h in db.scalars(select(Hospital)).all()],
        "roads": [{"road": r.name, "condition": r.condition,
                   "hazard_pct": round(r.hazard_score * 100)}
                  for r in db.scalars(select(Road)).all()],
        "contacts": [{"service": c.service, "number": c.number}
                     for c in db.scalars(select(EmergencyContact)).all()],
    }


def retrieval_answer(db: Session, question: str, ctx: dict) -> str:
    q = question.lower()
    districts = ctx["districts"]

    if any(k in q for k in ("flood risk", "high risk", "at risk", "which area", "what areas")):
        top = sorted(districts, key=lambda d: d["flood"], reverse=True)[:4]
        lines = [f"- {d['district']} — {d['flood']}% flood probability "
                 f"(composite {d['composite_pct']}%, {d['level']})" for d in top]
        return "Highest modelled flood risk right now:\n" + "\n".join(lines)

    if any(k in q for k in ("shelter", "camp", "where to stay", "nearest")):
        top = sorted(ctx["shelters"], key=lambda s: s["free"], reverse=True)[:4]
        return "Shelters with the most space:\n" + "\n".join(
            f"- {s['name']} — {s['free']} free of {s['capacity']} ({', '.join(s['facilities'])})"
            for s in top)

    if any(k in q for k in ("road", "block", "route", "closed", "traffic")):
        bad = [r for r in ctx["roads"] if r["condition"] != "open"]
        return (f"{len(bad)} links are degraded:\n" + "\n".join(
            f"- {r['road']} — {r['condition']}, hazard {r['hazard_pct']}%" for r in bad)
            + "\nThe router penalises these automatically.")

    if any(k in q for k in ("resource", "ambulance", "boat", "stock", "supply", "short")):
        pool: dict[str, int] = {}
        for r in ctx["resources"]:
            pool[r["type"]] = pool.get(r["type"], 0) + r["available"]
        return "Available statewide: " + ", ".join(f"{v:,} {k.lower()}" for k, v in pool.items())

    if any(k in q for k in ("cyclone", "what should i do", "safety", "prepare")):
        return ("During a cyclone: move to a pucca building or the nearest designated cyclone "
                "shelter before winds build, stay away from the coast, trees and hoardings, "
                "switch off the mains if water is entering, and keep a charged phone and "
                "drinking water with you. Follow the district administration's instructions — "
                "call 1070 for the state EOC or 112 in an emergency.")

    if "evacuat" in q:
        top = sorted(districts, key=lambda d: d["evacuate"], reverse=True)[:4]
        return "Evacuation recommendations by volume:\n" + "\n".join(
            f"- {d['district']} — {d['evacuate']:,} residents above the threshold" for d in top)

    if any(k in q for k in ("status", "situation", "summary", "overall")):
        worst = max(districts, key=lambda d: d["composite_pct"])
        exposed = sum(d["exposed"] for d in districts)
        free = sum(s["free"] for s in ctx["shelters"])
        bad = sum(1 for r in ctx["roads"] if r["condition"] != "open")
        return (f"{len(ctx['alerts'])} active alerts. {worst['district']} is the priority district "
                f"at {worst['composite_pct']}% composite risk ({worst['level']}). {exposed:,} people "
                f"are inside the modelled exposure band, {bad} road links are degraded and "
                f"{free:,} shelter spaces remain.")

    if any(k in q for k in ("contact", "helpline", "number", "call")):
        return "\n".join(f"- {c['service']} — {c['number']}" for c in ctx["contacts"])

    return ("I can answer from the live system state: district risk scores, active alerts, "
            "shelter occupancy, resource stock, hospital beds and road conditions. Try "
            '"which districts are at high flood risk" or "where is the nearest shelter with space".')


def ask(db: Session, question: str) -> tuple[str, str]:
    """Returns (reply, answered_by)."""
    ctx = build_context(db)
    settings = get_settings()

    if settings.llm_api_key and settings.llm_provider == "anthropic":
        try:
            payload = {
                "model": settings.llm_model,
                "max_tokens": 600,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content":
                              f"CURRENT RESQTECH STATE (JSON):\n{json.dumps(ctx)}\n\n"
                              f"Question: {question}"}],
            }
            with httpx.Client(timeout=25.0) as client:
                r = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": settings.llm_api_key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json=payload)
                r.raise_for_status()
                blocks = r.json().get("content", [])
                text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if text.strip():
                    return text.strip(), "llm"
        except Exception:
            log.warning("LLM call failed; serving the retrieval fallback", exc_info=True)

    return retrieval_answer(db, question, ctx), "retrieval"
