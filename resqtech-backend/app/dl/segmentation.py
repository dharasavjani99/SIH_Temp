"""Impact segmentation.

Two implementations behind one contract:

  * ClassicalSegmenter — colour-response water detection plus a Sobel gradient
    pass for debris texture. Runs anywhere, needs no weights, and is what the
    prototype ships with.
  * UNetSegmenter — loads a trained PyTorch checkpoint if SEG_MODEL_PATH
    exists. Same input, same mask codes, same metrics dict out.

Both return: mask (H x W uint8, codes below), fractions, and derived estimates.
The caller records which engine answered; the UI prints it.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings

log = logging.getLogger(__name__)

CLASS_BACKGROUND, CLASS_WATER, CLASS_DEBRIS, CLASS_VEGETATION = 0, 1, 2, 3
CLASS_NAMES = {0: "background", 1: "standing water", 2: "debris/damage", 3: "vegetation"}
TARGET_WIDTH = 360


def _load_rgb(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    h = max(1, round(img.height * (TARGET_WIDTH / img.width)))
    return np.asarray(img.resize((TARGET_WIDTH, h), Image.LANCZOS), dtype=np.float32)


def _sobel(grey: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    pad = np.pad(grey, 1, mode="edge")
    gx = sum(kx[i, j] * pad[i:i + grey.shape[0], j:j + grey.shape[1]]
             for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * pad[i:i + grey.shape[0], j:j + grey.shape[1]]
             for i in range(3) for j in range(3))
    return np.hypot(gx, gy)


class ClassicalSegmenter:
    name = "classical-cv"
    version = "0.2"
    trained = False

    def segment(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        rgb = _load_rgb(data)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        grey = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0

        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        sat = np.where(mx == 0, 0, (mx - mn) / np.maximum(mx, 1e-6))
        water = ((b > r + 12) & (b > g + 4) & (b > 55)) | \
                ((sat < 0.19) & (b > 68) & (b < 175) & (np.abs(r - g) < 26))
        veg = (g > r + 16) & (g > b + 12)

        mask = np.zeros(grey.shape, dtype=np.uint8)
        mask[veg] = CLASS_VEGETATION
        mask[water] = CLASS_WATER

        edges = _sobel(grey)
        debris = ((edges > 0.30) & (mask == CLASS_BACKGROUND)) | \
                 ((edges > 0.60) & (mask == CLASS_VEGETATION))
        mask[debris] = CLASS_DEBRIS
        return mask, edges


class UNetSegmenter:
    """Drop-in trained head. Activated automatically when a checkpoint exists."""
    name = "unet-impactseg"
    trained = True

    def __init__(self, checkpoint: Path):
        import torch                                   # imported lazily
        self.torch = torch
        self.model = torch.jit.load(str(checkpoint), map_location="cpu")
        self.model.eval()
        self.version = getattr(self.model, "version", "1.0")

    def segment(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        rgb = _load_rgb(data) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(x)
        mask = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        grey = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return mask, _sobel(grey)


def get_segmenter():
    path = Path(get_settings().seg_model_path)
    if path.exists():
        try:
            return UNetSegmenter(path)
        except Exception:                                # pragma: no cover
            log.exception("Segmentation checkpoint present but unusable; using classical CV")
    return ClassicalSegmenter()


def analyse(data: bytes, footprint_km2: float = 96.0) -> dict:
    seg = get_segmenter()
    mask, edges = seg.segment(data)
    total = mask.size

    water_frac = float((mask == CLASS_WATER).sum() / total)
    debris_frac = float((mask == CLASS_DEBRIS).sum() / total)
    veg_frac = float((mask == CLASS_VEGETATION).sum() / total)
    edge_density = float(edges.mean())

    affected = round(footprint_km2 * (water_frac + debris_frac * 0.35), 1)
    structures = round(footprint_km2 * (debris_frac * 620 + water_frac * 180))
    road_km = round(footprint_km2 * 0.42 * (water_frac * 1.35 + debris_frac * 0.5), 1)
    severity_score = round(min(100.0, water_frac * 115 + debris_frac * 90 + edge_density * 90), 1)
    confidence = round(min(86.0, max(45.0, 48 + edge_density * 90 + water_frac * 40)), 1)

    severity = ("critical" if severity_score >= 75 else "high" if severity_score >= 55
                else "moderate" if severity_score >= 32 else "low")

    return {
        "engine": seg.name, "engine_version": seg.version, "is_trained_model": seg.trained,
        "scene_footprint_km2": footprint_km2,
        "affected_area_km2": affected,
        "damaged_structures": structures,
        "road_disruption_km": road_km,
        "severity": severity,
        "severity_score": severity_score,
        "confidence_pct": confidence,
        "fractions": {"water": round(water_frac, 4), "debris": round(debris_frac, 4),
                      "vegetation": round(veg_frac, 4)},
        "edge_density": round(edge_density, 4),
        "mask_shape": list(mask.shape),
        "class_names": CLASS_NAMES,
        "caveat": ("Area and structure counts are scaled from the declared scene footprint. "
                   "They are estimates from imagery, not a ground survey."),
    }


def mask_png(data: bytes) -> bytes:
    """Render the segmentation overlay as a PNG for the before/after view."""
    seg = get_segmenter()
    mask, _ = seg.segment(data)
    rgb = _load_rgb(data)
    out = (rgb * 0.55).astype(np.uint8)
    out[mask == CLASS_WATER] = (32, 120, 255)
    out[mask == CLASS_DEBRIS] = (255, 86, 54)
    veg = mask == CLASS_VEGETATION
    out[veg] = (rgb[veg] * np.array([0.7, 0.9, 0.7])).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
