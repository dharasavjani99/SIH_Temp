# Datasets

Every file here is **demonstration data**, assembled for the SIH 2026 prototype.
It is realistic in structure and plausible in magnitude, but it is not an
official record. Nothing here should be presented as a government feed.

## Files

| File | Rows | What it is |
|---|---|---|
| `districts.csv` | 12 | Gujarat districts with the static environmental profile the risk model needs: centroid, population, exposure-band population, elevation, slope, soil moisture, river and its danger mark, coastal flag, accessibility index, hospital and school counts, elderly and child shares. |
| `weather_observations.csv` | 12 | One current reading per district — 6 h and 24 h rainfall, temperature, humidity, wind, river stage. Loaded with `is_live=false`. |
| `weather_stations.csv` | 5 | AWS points for the map, with 6 h rainfall and trend. |
| `shelters.csv` | 12 | Capacity, live occupancy and facilities. |
| `hospitals.csv` | 7 | Total beds, free beds, free ICU. |
| `resources.csv` | 12 | Stock lines: ambulances, boats, NDRF/SDRF teams, tankers, medical kits, food packets, lighting, excavators, satellite phones. |
| `road_nodes.csv` | 12 | Routable graph vertices. |
| `road_edges.csv` | 17 | Graph edges with live condition (`open`, `congested`, `flooded`, `landslide`) and a hazard score. |
| `emergency_teams.csv` | 5 | NDRF, SDRF, fire and Coast Guard units with current task. |
| `emergency_contacts.csv` | 8 | Helpline numbers. These are the genuine published national/state numbers. |
| `demo_users.csv` | 4 | Demonstration accounts, one per role. **Delete before any real deployment.** |
| `historical_events.csv` | 6,000 | **Generated.** The training table for the risk models. |

## `historical_events.csv` — read this before quoting a metric

Produced by `scripts/generate_training_data.py`, seeded at 20260919 so it is
reproducible. Each row is one district-day: rainfall drawn from a gamma
distribution scaled by a per-district monsoon intensity, soil moisture
responding to rainfall and season, river stage responding to basin rainfall with
noise, a fat-tailed wind component for coastal districts, and four binary labels
sampled from latent logistic processes over those features.

**The labels are synthetic.** A model fitted here learns the generator. Its
cross-validated AUC and Brier score measure internal consistency and say nothing
about real-world forecasting skill. `app/ml/artifacts/metrics.json` carries this
warning in the artifact itself.

### Replacing it with real data

Keep the column names. Build the same table from:

- **Rainfall** — IMD gridded daily rainfall (0.25°) or AWS station series, aggregated to district-day.
- **River stage** — CWC flood-forecasting station levels joined to the district's gauge, with the published danger mark.
- **Terrain** — SRTM/Cartosat DEM for elevation and slope, district zonal means.
- **Soil moisture** — ISRO Bhuvan or SMAP retrievals.
- **Labels** — NDMA situation reports, state relief-commissioner records, or EM-DAT, matched to district-date.

Then rerun `scripts/train_risk_model.py`. Nothing else changes, and the accuracy
number in the report becomes one you can defend.

## Provenance of the demo figures

District population and coastline are from public census and geographic
knowledge and are approximately right. Rainfall, river stages, shelter
occupancy, stock levels and road conditions are **invented** to produce an
interesting operating picture for the demonstration — a monsoon event centred on
south Gujarat with a landslide on the Saputara ghat road. They are not
observations of any real event.
