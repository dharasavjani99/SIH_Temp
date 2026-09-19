"""Routing must actually avoid blocked links, not just report a distance."""
from app.database.session import SessionLocal
from app.services import routing


def test_safe_route_avoids_the_landslide():
    db = SessionLocal()
    try:
        fast = routing.solve(db, "n7", "n10", "fast", "Ambulance")
        safe = routing.solve(db, "n7", "n10", "safe", "Ambulance")
        assert fast and safe
        assert safe["route_hazard_pct"] < fast["route_hazard_pct"]
        assert safe["distance_km"] > fast["distance_km"]   # safety costs distance
    finally:
        db.close()


def test_boat_treats_flooding_differently_from_an_ambulance():
    db = SessionLocal()
    try:
        amb = routing.solve(db, "n7", "n8", "balanced", "Ambulance")
        boat = routing.solve(db, "n7", "n8", "balanced", "Rescue boat")
        assert amb and boat
        # The direct link n7-n8 is flooded: the boat is willing to take it.
        assert [l["road"] for l in boat["legs"]] == ["Tapi south bank"]
        assert [l["road"] for l in amb["legs"]] != ["Tapi south bank"]
    finally:
        db.close()


def test_unknown_node_returns_none():
    db = SessionLocal()
    try:
        assert routing.solve(db, "n1", "nowhere", "fast", "Ambulance") is None
    finally:
        db.close()
