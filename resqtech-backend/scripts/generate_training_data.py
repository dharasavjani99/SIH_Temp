#!/usr/bin/env python
"""Build the training table for the risk models.

IMPORTANT, READ BEFORE QUOTING ANY ACCURACY NUMBER
--------------------------------------------------
Labels here are SYNTHETIC. They are drawn from a physically motivated
generator (rainfall accumulation, antecedent soil moisture, river stage against
the danger mark, terrain slope, coastal wind exposure) with noise added, not
from a record of real events. A model trained on this file learns to reproduce
the generator, so its cross-validated score measures internal consistency,
never real-world forecasting skill.

This file exists so the ML pipeline is genuinely end-to-end and runnable today.
Replace it with a real historical table — IMD daily rainfall, CWC gauge series
and EM-DAT / NDMA event records joined on district-date — and retrain. The
schema below is the join target, so nothing downstream changes.

Columns match app.ml.features.FEATURE_ORDER plus one binary label per hazard.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "historical_events.csv"
N_ROWS = 6000
SEED = 20260919

DISTRICT_PROFILES = [
    # name, elevation_m, slope_deg, base_soil, river_danger_m, coastal, monsoon_intensity
    ("Ahmedabad", 53, 2, 0.55, 8.5, 0, 1.00),
    ("Surat", 13, 1, 0.62, 9.4, 1, 1.25),
    ("Vadodara", 39, 2, 0.50, 8.0, 0, 1.05),
    ("Rajkot", 134, 3, 0.36, 7.2, 0, 0.70),
    ("Jamnagar", 20, 2, 0.42, 6.5, 1, 0.80),
    ("Bhuj", 110, 4, 0.22, 5.8, 1, 0.42),
    ("Porbandar", 8, 1, 0.45, 6.2, 1, 0.85),
    ("Valsad", 12, 2, 0.66, 7.0, 1, 1.45),
    ("Dang", 975, 27, 0.70, 6.4, 0, 1.55),
    ("Bhavnagar", 24, 2, 0.47, 7.4, 1, 0.88),
    ("Patan", 70, 1, 0.31, 6.0, 0, 0.55),
    ("Navsari", 11, 1, 0.64, 7.3, 1, 1.35),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def logistic(z):
    return 1 / (1 + math.exp(-z))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    for _ in range(N_ROWS):
        name, elev, slope, base_soil, danger, coastal, intensity = \
            DISTRICT_PROFILES[rng.integers(len(DISTRICT_PROFILES))]

        # Season: 0 = dry, 1 = monsoon. Monsoon days dominate the event record.
        monsoon = rng.random() < 0.62

        # Rainfall: gamma-distributed, heavier in monsoon and in high-intensity
        # districts. 24h totals above ~204 mm are IMD "extremely heavy".
        scale = (22 if monsoon else 4) * intensity
        rain24 = float(rng.gamma(shape=1.7, scale=scale))
        burst = clamp(rng.beta(2.2, 3.0), 0.05, 0.85)
        rain6 = rain24 * burst

        soil = clamp(base_soil + (0.22 if monsoon else -0.12)
                     + rain24 / 900 + rng.normal(0, 0.07), 0.02, 1.0)
        humidity = clamp((78 if monsoon else 52) + rain24 / 12 + rng.normal(0, 7), 25, 100)
        temp = clamp((27.5 if monsoon else 32.5) - rain24 / 55 + rng.normal(0, 2.2), 12, 46)

        # Wind: fat tail for coastal districts to represent cyclone season.
        wind = float(rng.gamma(2.0, 9.0)) + (float(rng.gamma(1.4, 22.0)) if (coastal and rng.random() < 0.12) else 0)
        wind = clamp(wind, 2, 235)

        # River stage responds to basin rainfall with lag and noise.
        river = clamp(danger * (0.28 + rain24 / 420 + soil * 0.28) + rng.normal(0, 0.55), 0.1, danger * 1.45)

        f = {
            "rain24": clamp(rain24 / 300, 0, 1.2),
            "rain6": clamp(rain6 / 140, 0, 1.2),
            "river_ratio": clamp(river / danger, 0, 1.3),
            "soil": soil,
            "low_elev": clamp(1 - elev / 400, 0, 1),
            "flat": clamp(1 - slope / 20, 0, 1),
            "humidity": clamp((humidity - 50) / 50, 0, 1),
            "slope": clamp(slope / 35, 0, 1),
            "wind": clamp((wind - 20) / 80, 0, 1),
            "coastal": float(coastal),
            "temp": clamp((temp - 20) / 20, 0, 1),
        }

        # Latent hazard processes. Coefficients encode domain reasoning, and
        # the noise term is what stops the learner from recovering them exactly.
        z_flood = (-5.9 + 2.1 * f["rain24"] + 1.5 * f["rain6"] + 3.1 * f["river_ratio"]
                   + 1.1 * f["soil"] + 1.2 * f["low_elev"] + 0.6 * f["flat"]
                   + 0.4 * f["humidity"] + rng.normal(0, 0.55))
        z_land = (-5.6 + 1.3 * f["rain24"] + 1.2 * f["rain6"] + 1.7 * f["soil"]
                  + 3.2 * f["slope"] - 0.4 * f["low_elev"] + rng.normal(0, 0.5))
        z_cyc = (-4.8 + 3.9 * f["wind"] + 1.7 * f["coastal"] + 0.7 * f["humidity"]
                 + 0.5 * f["rain6"] + rng.normal(0, 0.45))
        z_rain = (-3.4 + 2.9 * f["rain24"] + 2.6 * f["rain6"] + 1.1 * f["humidity"]
                  - 0.5 * f["temp"] + rng.normal(0, 0.4))

        rows.append({
            "district": name, "monsoon": int(monsoon),
            "rain_24h_mm": round(rain24, 1), "rain_6h_mm": round(rain6, 1),
            "river_level_m": round(river, 2), "river_danger_m": danger,
            "soil_moisture_index": round(soil, 3), "elevation_m": elev,
            "slope_deg": slope, "humidity_pct": round(humidity, 1),
            "wind_kmh": round(wind, 1), "temp_c": round(temp, 1), "coastal": coastal,
            **{k: round(v, 5) for k, v in f.items()},
            "label_flood": int(rng.random() < logistic(z_flood)),
            "label_landslide": int(rng.random() < logistic(z_land)),
            "label_cyclone": int(rng.random() < logistic(z_cyc)),
            "label_rainfall": int(rng.random() < logistic(z_rain)),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows):,} rows to {OUT}")
    for hazard in ("flood", "landslide", "cyclone", "rainfall"):
        rate = sum(r[f"label_{hazard}"] for r in rows) / len(rows)
        print(f"  positive rate {hazard:<10} {rate:.3f}")


if __name__ == "__main__":
    main()
