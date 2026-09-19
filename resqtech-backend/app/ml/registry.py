"""Loads the trained risk model if one has been built, otherwise reports that
none is available. The service layer decides what to do about it — it never
pretends a surrogate result came from a trained model."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from app.core.config import get_settings

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict = {}
        self.metadata: dict = {}
        self.loaded = False
        self._load()

    def _load(self) -> None:
        path = Path(get_settings().risk_model_path)
        if not path.exists():
            log.warning("No trained risk model at %s — falling back to the surrogate. "
                        "Run scripts/train_risk_model.py to build one.", path)
            return
        try:
            bundle = joblib.load(path)
            self.models = bundle["models"]
            self.metadata = bundle["metadata"]
            self.loaded = True
            log.info("Loaded risk model %s", self.metadata.get("version"))
        except Exception:                                    # pragma: no cover
            log.exception("Risk model at %s could not be loaded", path)

        metrics = path.with_name("metrics.json")
        if metrics.exists():
            self.metadata["metrics"] = json.loads(metrics.read_text())

    @property
    def version(self) -> str:
        return self.metadata.get("version", "none")


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reload_registry() -> ModelRegistry:
    global _registry
    _registry = ModelRegistry()
    return _registry
