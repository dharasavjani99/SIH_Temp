import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'test_resqtech.db'}")

import pytest                                              # noqa: E402
from fastapi.testclient import TestClient                  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    from scripts.seed_db import main as seed
    seed()


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def admin_token(client):
    r = client.post("/api/auth/login",
                    json={"email": "admin@resqtech.gov.in", "password": "demo1234"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture
def auth(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
