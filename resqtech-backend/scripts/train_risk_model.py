#!/usr/bin/env python
"""Fit one gradient-boosting classifier per hazard and save the bundle.

Outputs:
  app/ml/artifacts/risk_model.joblib   models + metadata, loaded by ModelRegistry
  app/ml/artifacts/metrics.json        honest out-of-fold metrics

The metrics are written verbatim, including the warning that the labels are
synthetic. Do not quote these numbers as forecasting accuracy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.features import FEATURE_ORDER, HAZARDS   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "historical_events.csv"
ARTIFACTS = ROOT / "app" / "ml" / "artifacts"
VERSION = "0.4.0"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found — run scripts/generate_training_data.py first")

    df = pd.read_csv(DATA)
    X = df[FEATURE_ORDER].to_numpy(dtype=float)

    models, metrics = {}, {}
    for hazard in HAZARDS:
        y = df[f"label_{hazard}"].to_numpy(dtype=int)
        base = HistGradientBoostingClassifier(
            max_iter=260, learning_rate=0.06, max_leaf_nodes=24,
            min_samples_leaf=25, l2_regularization=0.6, random_state=7)
        # Probability calibration matters more than raw accuracy here: the UI
        # shows the probability itself, not a yes/no classification.
        clf = CalibratedClassifierCV(base, method="isotonic", cv=4)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics[hazard] = {
            "positive_rate": round(float(y.mean()), 4),
            "roc_auc_oof": round(float(roc_auc_score(y, oof)), 4),
            "brier_oof": round(float(brier_score_loss(y, oof)), 4),
            "n_rows": int(len(y)),
        }
        clf.fit(X, y)
        models[hazard] = clf
        print(f"{hazard:<10} AUC {metrics[hazard]['roc_auc_oof']:.3f}  "
              f"Brier {metrics[hazard]['brier_oof']:.3f}")

    metrics["mean_brier"] = round(float(np.mean([metrics[h]["brier_oof"] for h in HAZARDS])), 4)
    metrics["trained_at"] = datetime.now(timezone.utc).isoformat()
    metrics["version"] = VERSION
    metrics["label_provenance"] = (
        "SYNTHETIC. Labels come from scripts/generate_training_data.py, not from "
        "observed disaster records. These scores measure how well the model "
        "recovers that generator and say nothing about real forecasting skill. "
        "Retrain on IMD/CWC/NDMA historical data before making any accuracy claim."
    )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models,
        "metadata": {"version": VERSION, "features": FEATURE_ORDER,
                     "trained_at": metrics["trained_at"],
                     "label_provenance": metrics["label_provenance"]},
    }, ARTIFACTS / "risk_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved -> {ARTIFACTS / 'risk_model.joblib'}")
    print(f"metrics -> {ARTIFACTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
