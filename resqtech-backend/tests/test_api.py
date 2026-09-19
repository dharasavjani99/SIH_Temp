def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_protected_route_rejects_anonymous(client):
    assert client.get("/api/dashboard").status_code == 401


def test_bad_password_is_rejected(client):
    r = client.post("/api/auth/login",
                    json={"email": "admin@resqtech.gov.in", "password": "wrongpass"})
    assert r.status_code == 401


def test_dashboard_shape(client, auth):
    body = client.get("/api/dashboard", headers=auth).json()
    assert set(body["headline"]) >= {"active_disasters", "people_at_risk", "roads_impassable"}
    assert body["data_mode"] == "demo"
    assert len(body["districts"]) == 12


def test_risk_prediction_reports_its_model(client, auth):
    r = client.post("/api/risk/predict", headers=auth,
                    json={"location_code": "ahd", "rain_24h_mm": 300, "rain_6h_mm": 140})
    assert r.status_code == 200
    body = r.json()
    assert body["hazards"]["flood"] > 50
    assert body["model_name"]
    assert body["contributions"]["flood"]


def test_public_role_cannot_predict(client):
    tok = client.post("/api/auth/login",
                      json={"email": "citizen@resqtech.in",
                            "password": "demo1234"}).json()["access_token"]
    r = client.post("/api/risk/predict", headers={"Authorization": f"Bearer {tok}"},
                    json={"location_code": "ahd"})
    assert r.status_code == 403


def test_alert_grade_is_validated(client, auth):
    r = client.post("/api/alerts", headers=auth, json={
        "location_code": "pat", "hazard": "flood", "grade": "emergency",
        "probability_pct": 12, "recommended_action": "Something reasonable here."})
    assert r.status_code == 422


def test_allocation_explains_itself(client, auth):
    body = client.post("/api/resources/allocate", headers=auth,
                       json={"location_code": "srt", "commit": False}).json()
    assert body["status"] == "proposed"
    assert len(body["rationale"]) >= 4
    assert any(l["resource_type"] == "Ambulance" for l in body["lines"])


def test_route_optimisation(client, auth):
    body = client.post("/api/routes/optimize", headers=auth, json={
        "from_node": "n7", "to_node": "n10", "objective": "safe",
        "vehicle": "Ambulance"}).json()
    assert body["distance_km"] > 0 and body["eta_minutes"] > 0
    assert body["legs"]


def test_chat_falls_back_without_a_key(client, auth):
    body = client.post("/api/chat", headers=auth,
                       json={"message": "which districts are at high flood risk"}).json()
    assert body["answered_by"] == "retrieval"
    assert "flood" in body["reply"].lower()


def test_report_refuses_to_invent_accuracy(client, auth):
    body = client.get("/api/reports", headers=auth).json()
    assert body["prediction_accuracy"] is None
    assert "backtest" in body["accuracy_note"]


def test_contacts_are_public(client):
    r = client.get("/api/contacts")
    assert r.status_code == 200
    assert any(c["number"] == "112" for c in r.json())
