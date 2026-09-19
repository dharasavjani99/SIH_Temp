"""Emergency route optimisation.

Dijkstra over the road graph. Edge cost blends great-circle distance with a
live condition penalty and a hazard term; the weighting changes with the
objective, so "fastest" and "safest" genuinely return different paths.
"""
from __future__ import annotations

import heapq
import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Road, RoadNode

CONDITION_PENALTY = {"open": 1.0, "congested": 1.7, "flooded": 6.5, "landslide": 9.0}
VEHICLE_SPEED_KMH = {"Ambulance": 52, "NDRF truck": 42, "Relief convoy": 36, "Rescue boat": 18}
# A boat is not stopped by water, so flooding is a mild penalty for it.
VEHICLE_OVERRIDES = {"Rescue boat": {"flooded": 0.55}}


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    d_lat, d_lon = math.radians(b_lat - a_lat), math.radians(b_lon - a_lon)
    la, lb = math.radians(a_lat), math.radians(b_lat)
    h = math.sin(d_lat / 2) ** 2 + math.cos(la) * math.cos(lb) * math.sin(d_lon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _graph(db: Session):
    nodes = {n.code: n for n in db.scalars(select(RoadNode)).all()}
    adj: dict[str, list] = {code: [] for code in nodes}
    for edge in db.scalars(select(Road)).all():
        if edge.from_node in nodes and edge.to_node in nodes:
            adj[edge.from_node].append((edge.to_node, edge))
            adj[edge.to_node].append((edge.from_node, edge))
    return nodes, adj


def solve(db: Session, start: str, end: str, objective: str = "balanced",
          vehicle: str = "Ambulance") -> dict | None:
    nodes, adj = _graph(db)
    if start not in nodes or end not in nodes or start == end:
        return None

    overrides = VEHICLE_OVERRIDES.get(vehicle, {})
    dist = {c: math.inf for c in nodes}
    prev: dict[str, tuple[str, Road]] = {}
    dist[start] = 0.0
    queue = [(0.0, start)]
    seen: set[str] = set()

    while queue:
        d, u = heapq.heappop(queue)
        if u in seen:
            continue
        seen.add(u)
        if u == end:
            break
        for v, edge in adj[u]:
            if v in seen:
                continue
            km = haversine_km(nodes[u].lat, nodes[u].lon, nodes[v].lat, nodes[v].lon)
            penalty = overrides.get(edge.condition, CONDITION_PENALTY[edge.condition])
            if objective == "fast":
                cost = km * (1 + (penalty - 1) * 0.45)
            elif objective == "safe":
                cost = km * penalty * (1 + edge.hazard_score * 2.2)
            else:
                cost = km * penalty * (1 + edge.hazard_score * 0.9)
            if d + cost < dist[v]:
                dist[v] = d + cost
                prev[v] = (u, edge)
                heapq.heappush(queue, (dist[v], v))

    if math.isinf(dist[end]):
        return None

    path, legs, cursor = [end], [], end
    while cursor in prev:
        parent, edge = prev[cursor]
        km = haversine_km(nodes[parent].lat, nodes[parent].lon, nodes[cursor].lat, nodes[cursor].lon)
        legs.append({
            "from": nodes[parent].name, "to": nodes[cursor].name, "road": edge.name,
            "length_km": round(km, 1), "condition": edge.condition,
            "hazard_pct": round(edge.hazard_score * 100),
        })
        cursor = parent
        path.append(cursor)
    path.reverse()
    legs.reverse()

    total_km = round(sum(l["length_km"] for l in legs), 1)
    speed = VEHICLE_SPEED_KMH.get(vehicle, 45)
    delay = sum(l["length_km"] * (CONDITION_PENALTY[l["condition"]] - 1) * 0.12 for l in legs)
    eta = round((total_km / speed) * 60 + delay + 3)
    worst = max((l["hazard_pct"] for l in legs), default=0)

    return {
        "objective": objective, "vehicle": vehicle,
        "path": path, "legs": legs,
        "distance_km": total_km, "eta_minutes": eta, "route_hazard_pct": worst,
        "degraded_links": [l for l in legs if l["condition"] != "open"],
    }


def alternatives(db: Session, start: str, end: str, chosen: dict, vehicle: str) -> list[dict]:
    out, signature = [], [l["road"] for l in chosen["legs"]]
    for objective in ("fast", "safe", "balanced"):
        if objective == chosen["objective"]:
            continue
        alt = solve(db, start, end, objective, vehicle)
        if alt and [l["road"] for l in alt["legs"]] != signature:
            out.append(alt)
    return out
