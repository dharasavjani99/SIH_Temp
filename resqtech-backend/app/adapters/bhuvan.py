"""ISRO Bhuvan adapter — satellite tiles and flood-extent layers.

Bhuvan exposes WMS/WMTS layers rather than a JSON observation feed, so this
adapter returns layer descriptors for the map instead of Observations.
"""
from __future__ import annotations

from app.adapters.base import DataAdapter
from app.core.config import get_settings

LAYERS = [
    {"id": "flood_extent", "title": "Flood inundation extent",
     "wms": "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms", "layer": "flood:extent"},
    {"id": "lulc", "title": "Land use / land cover",
     "wms": "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms", "layer": "lulc:GJ"},
]


class BhuvanAdapter(DataAdapter):
    source_id = "bhuvan"
    homepage = "https://bhuvan-app1.nrsc.gov.in/bhuvandisaster"

    def __init__(self):
        self.settings = get_settings()

    @property
    def is_live(self) -> bool:
        return bool(self.settings.bhuvan_api_key) and self.settings.data_mode == "live"

    def fetch(self, location_codes: list[str]):
        return []          # this source contributes map layers, not observations

    def layers(self) -> list[dict]:
        return [{**l, "available": self.is_live} for l in LAYERS]
