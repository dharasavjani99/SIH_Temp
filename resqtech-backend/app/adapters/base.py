"""External data adapters.

Every adapter returns the same shape regardless of whether it fetched a live
feed or read the seeded demo CSV, and every record carries `is_live` so the
API — and therefore the UI badge — can never misreport provenance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class Observation:
    location_code: str
    observed_at: datetime
    rain_6h_mm: float = 0.0
    rain_24h_mm: float = 0.0
    temp_c: float = 0.0
    humidity_pct: float = 0.0
    wind_kmh: float = 0.0
    river_level_m: float | None = None
    source: str = "demo"
    is_live: bool = False


class DataAdapter(ABC):
    source_id: str = "base"
    homepage: str = ""

    @abstractmethod
    def fetch(self, location_codes: list[str]) -> list[Observation]:
        ...

    @property
    def is_live(self) -> bool:
        return False

    def describe(self) -> dict:
        return {"id": self.source_id, "homepage": self.homepage,
                "mode": "live" if self.is_live else "demo"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
