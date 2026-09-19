"""Central Water Commission river-gauge adapter.

CWC publishes flood forecasts and gauge levels through its own portal. The
live branch expects a station-keyed endpoint; the demo branch replays the
seeded levels so the risk engine has the same field either way.
"""
from __future__ import annotations

import httpx

from app.adapters.base import DataAdapter, Observation, utcnow
from app.core.config import get_settings


class CWCAdapter(DataAdapter):
    source_id = "cwc"
    homepage = "https://cwc.gov.in"
    base_url = "https://ffs.india-water.gov.in/api"   # placeholder

    def __init__(self, demo_rows: dict[str, dict] | None = None):
        self.demo_rows = demo_rows or {}
        self.settings = get_settings()

    @property
    def is_live(self) -> bool:
        return bool(self.settings.cwc_api_key) and self.settings.data_mode == "live"

    def fetch(self, location_codes: list[str]) -> list[Observation]:
        if not self.is_live:
            return [Observation(location_code=c, observed_at=utcnow(),
                                river_level_m=self.demo_rows[c]["river_level_m"],
                                source="cwc-demo", is_live=False)
                    for c in location_codes if c in self.demo_rows]
        out = []
        with httpx.Client(timeout=10.0,
                          headers={"Authorization": f"Bearer {self.settings.cwc_api_key}"}) as c:
            for code in location_codes:
                r = c.get(f"{self.base_url}/gauge", params={"station": code})
                r.raise_for_status()
                out.append(Observation(location_code=code, observed_at=utcnow(),
                                       river_level_m=r.json()["level_m"],
                                       source="cwc", is_live=True))
        return out
