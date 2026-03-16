"""API contract tests for the CPU Power Spike Prediction service."""


class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_cpu_types_loaded(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["cpu_types_loaded"] >= 10
        assert "unknown" in data["available_cpu_types"]
        assert "intel-xeon-e5420" in data["available_cpu_types"]


class TestPredict:
    def test_returns_required_fields(self, client):
        r = client.get("/predict", params={"cpu_pct": 50, "cpu_type": "intel-xeon-e5420"})
        assert r.status_code == 200
        for field in ("cpu_type", "cpu_pct", "power_w", "power_lower_w", "power_upper_w",
                      "is_spike", "spike_threshold_w"):
            assert field in r.json(), f"Missing field: {field}"

    def test_power_positive(self, client):
        r = client.get("/predict", params={"cpu_pct": 0, "cpu_type": "intel-xeon-e5420"})
        assert r.json()["power_w"] > 0

    def test_power_lower_lte_mean_lte_upper(self, client):
        r = client.get("/predict", params={"cpu_pct": 50, "cpu_type": "intel-xeon-e5420"})
        d = r.json()
        assert d["power_lower_w"] <= d["power_w"] <= d["power_upper_w"]

    def test_idle_not_spike(self, client):
        r = client.get("/predict", params={"cpu_pct": 0, "cpu_type": "intel-xeon-e5420"})
        assert r.json()["is_spike"] is False

    def test_full_load_is_spike(self, client):
        r = client.get("/predict", params={"cpu_pct": 100, "cpu_type": "intel-xeon-e5420"})
        assert r.json()["is_spike"] is True

    def test_unknown_cpu_type_falls_back(self, client):
        r = client.get("/predict", params={"cpu_pct": 50, "cpu_type": "totally-unknown-cpu"})
        assert r.status_code == 200
        assert r.json()["cpu_type"] == "unknown"

    def test_default_cpu_type_is_unknown(self, client):
        r = client.get("/predict", params={"cpu_pct": 50})
        assert r.status_code == 200
        assert r.json()["cpu_type"] == "unknown"

    def test_cpu_pct_below_zero_rejected(self, client):
        r = client.get("/predict", params={"cpu_pct": -1})
        assert r.status_code == 422

    def test_cpu_pct_above_100_rejected(self, client):
        r = client.get("/predict", params={"cpu_pct": 101})
        assert r.status_code == 422

    def test_all_cpu_types_return_predictions(self, client):
        cpu_types = [m["cpu_type"] for m in client.get("/models").json()]
        for ct in cpu_types:
            r = client.get("/predict", params={"cpu_pct": 50, "cpu_type": ct})
            assert r.status_code == 200
            assert r.json()["power_w"] > 0

    def test_power_increases_with_load(self, client):
        powers = [
            client.get("/predict", params={"cpu_pct": pct, "cpu_type": "intel-xeon-e5420"}).json()["power_w"]
            for pct in [0, 25, 50, 75, 100]
        ]
        assert powers == sorted(powers), f"Power not monotone: {powers}"

    def test_spike_threshold_consistent(self, client):
        thresholds = {
            client.get("/predict", params={"cpu_pct": pct, "cpu_type": "intel-xeon-e5420"}).json()["spike_threshold_w"]
            for pct in [0, 50, 100]
        }
        assert len(thresholds) == 1


class TestBatchPredict:
    def test_returns_list(self, client):
        r = client.post("/predict/batch", json={"predictions": [
            {"cpu_pct": 30, "cpu_type": "intel-xeon-e5420"},
            {"cpu_pct": 90, "cpu_type": "amd-opteron-2214"},
        ]})
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_batch_spike_flags_correct(self, client):
        r = client.post("/predict/batch", json={"predictions": [
            {"cpu_pct": 0,   "cpu_type": "intel-xeon-e5420"},
            {"cpu_pct": 100, "cpu_type": "intel-xeon-e5420"},
        ]})
        results = r.json()
        assert results[0]["is_spike"] is False
        assert results[1]["is_spike"] is True

    def test_batch_invalid_cpu_pct_rejected(self, client):
        r = client.post("/predict/batch", json={"predictions": [
            {"cpu_pct": 150, "cpu_type": "intel-xeon-e5420"},
        ]})
        assert r.status_code == 422

    def test_batch_empty_list(self, client):
        r = client.post("/predict/batch", json={"predictions": []})
        assert r.status_code == 200
        assert r.json() == []


class TestModels:
    def test_returns_list(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        assert len(r.json()) >= 10

    def test_model_fields(self, client):
        for m in client.get("/models").json():
            for field in ("cpu_type", "idle_w", "full_w", "spike_threshold_w", "mean_std_w"):
                assert field in m, f"Missing field: {field}"

    def test_full_load_gt_idle(self, client):
        for m in client.get("/models").json():
            assert m["full_w"] > m["idle_w"], f"{m['cpu_type']}: full_w <= idle_w"

    def test_spike_threshold_between_idle_and_full(self, client):
        for m in client.get("/models").json():
            assert m["idle_w"] < m["spike_threshold_w"] < m["full_w"], \
                f"{m['cpu_type']}: spike_threshold not between idle and full"
