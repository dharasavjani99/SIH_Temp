#!/usr/bin/env bash
# Local development, no Docker, no Postgres needed.
set -e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env || true
export DATABASE_URL="sqlite:///./resqtech.db"
python scripts/generate_training_data.py
python scripts/train_risk_model.py
python scripts/seed_db.py
uvicorn app.main:app --reload --port 8000
