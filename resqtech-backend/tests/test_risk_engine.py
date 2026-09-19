"""The risk engine must respond to its inputs, not return a constant."""
from app.ml.features import RawObservation, build_features
from app.services import risk_engine


def test_more_rain_raises_flood_risk():
    dry = RawObservation(rain_24h_mm=20, rain_6h_mm=5, river_level_m=2.0, river_danger_m=8.5,
                         soil_moisture_index=0.3, elevation_m=53, slope_deg=2)
    wet = RawObservation(rain_24h_mm=260, rain_6h_mm=110, river_level_m=8.1, river_danger_m=8.5,
                         soil_moisture_index=0.8, elevation_m=53, slope_deg=2)
    assert risk_engine.score(wet)["hazards"]["flood"] > risk_engine.score(dry)["hazards"]["flood"] + 20


def test_slope_drives_landslide_not_flood():
    flat = RawObservation(rain_24h_mm=280, rain_6h_mm=120, slope_deg=1, elevation_m=15,
                          soil_moisture_index=0.8)
    hill = RawObservation(rain_24h_mm=280, rain_6h_mm=120, slope_deg=28, elevation_m=900,
                          soil_moisture_index=0.85)
    assert risk_engine.score(hill)["hazards"]["landslide"] > \
           risk_engine.score(flat)["hazards"]["landslide"]


def test_coastal_wind_drives_cyclone():
    inland = RawObservation(wind_kmh=95, coastal=False, humidity_pct=90)
    coast = RawObservation(wind_kmh=95, coastal=True, humidity_pct=90)
    assert risk_engine.score(coast)["hazards"]["cyclone"] > \
           risk_engine.score(inland)["hazards"]["cyclone"]


def test_features_stay_bounded():
    obs = RawObservation(rain_24h_mm=9999, wind_kmh=9999, humidity_pct=100, slope_deg=90)
    for name, value in build_features(obs).items():
        assert 0.0 <= value <= 1.3, f"{name} escaped its range at {value}"


def test_scores_are_deterministic():
    obs = RawObservation(rain_24h_mm=214, rain_6h_mm=96, river_level_m=7.9, river_danger_m=8.5)
    assert risk_engine.score(obs) == risk_engine.score(obs)


def test_projection_is_monotonic_in_confidence():
    obs = RawObservation(rain_24h_mm=214, rain_6h_mm=96)
    conf = [h["confidence_pct"] for h in risk_engine.project(obs)]
    assert conf == sorted(conf, reverse=True)
