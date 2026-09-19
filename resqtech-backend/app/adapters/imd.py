"""India Meteorological Department (Mausam) adapter.

IMD does not publish a general-purpose open REST API for nowcast data; access
is arranged per-organisation. Until credentials exist, `fetch` reads the
seeded observations and marks them is_live=False. Do not scrape the website —
the request path below is the shape to fill in once access is granted.
"""
from __future__ import annotations

import httpx

from app.adapters.base import DataAdapter, Observation, utcnow
from app.core.config import get_settings


class IMDAdapter(DataAdapter):
    source_id = "imd"
    homepage = "https://mausam.imd.gov.in"
    base_url = "https://api.mausam.imd.gov.in/v1"   # placeholder, not public

    def __init__(self, demo_rows: dict[str, dict] | None = None):
        self.demo_rows = demo_rows or {}
        self.settings = get_settings()

    @property
    def is_live(self) -> bool:
        return bool(self.settings.imd_api_key) and self.settings.data_mode == "live"

    def fetch(self, location_codes: list[str]) -> list[Observation]:
        if not self.is_live:
            return self._demo(location_codes)
        headers = {"Authorization": f"Bearer {self.settings.imd_api_key}"}
        out = []
        with httpx.Client(timeout=10.0, headers=headers) as client:
            for code in location_codes:
                r = client.get(f"{self.base_url}/nowcast", params={"station": code})
                r.raise_for_status()
                d = r.json()
                out.append(Observation(
                    location_code=code, observed_at=utcnow(),
                    rain_6h_mm=d["rainfall"]["last_6h_mm"],
                    rain_24h_mm=d["rainfall"]["last_24h_mm"],
                    temp_c=d["temperature_c"], humidity_pct=d["humidity_pct"],
                    wind_kmh=d["wind_speed_kmh"], source="imd", is_live=True))
        return out

    def _demo(self, codes):
        rows = []
        for code in codes:
            d = self.demo_rows.get(code)
            if not d:
                continue
            rows.append(Observation(
                location_code=code, observed_at=utcnow(),
                rain_6h_mm=d["rain_6h_mm"], rain_24h_mm=d["rain_24h_mm"],
                temp_c=d["temp_c"], humidity_pct=d["humidity_pct"],
                wind_kmh=d["wind_kmh"], source="imd-demo", is_live=False))
        return rows
