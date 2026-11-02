import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, sample
from app.model import build_random_checkpoint, load_model
from src.data.darcy import gaussian_random_field, permeability_from_grf, solve_darcy


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


@pytest.fixture
def client(ckpt, monkeypatch):
    # Function-scoped: the lifespan stores the model on the shared app's state.
    monkeypatch.setenv("FNO_CHECKPOINT", str(ckpt))
    with TestClient(app) as c:
        yield c


def test_sample_matches_upstream_solver():
    a, u = sample(3)
    a_ref = permeability_from_grf(gaussian_random_field(64, rng=np.random.default_rng(3)))
    np.testing.assert_array_equal(a, a_ref)
    np.testing.assert_allclose(u, solve_darcy(a_ref), rtol=1e-6)


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "untrained (random weights)" in r.text


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["params"] == 2_368_001


def test_health_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FNO_CHECKPOINT", str(tmp_path / "missing.pt"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json={"a": [[1.0] * 64] * 64}).status_code == 503


def test_sample_endpoint(client):
    body = client.get("/sample", params={"seed": 5}).json()
    a, u = np.array(body["a"]), np.array(body["u"])
    assert body["seed"] == 5 and a.shape == u.shape == (64, 64)
    assert set(np.unique(a)) <= {3.0, 12.0}
    assert client.get("/sample", params={"seed": 5}).json() == body


def test_sample_rejects_negative_seed(client):
    assert client.get("/sample", params={"seed": -1}).status_code == 422


def test_predict(client, ckpt):
    a = client.get("/sample", params={"seed": 7}).json()["a"]
    r = client.post("/predict", json={"a": a})
    assert r.status_code == 200
    u = np.array(r.json()["u"])
    assert u.shape == (64, 64)
    np.testing.assert_allclose(u, load_model(ckpt).predict(a), rtol=1e-6)


@pytest.mark.parametrize("a", [
    [[1.0] * 64] * 63,                      # too few rows
    [[1.0] * 63] * 64,                      # too few cols
    [[1.0] * 64] * 63 + [[1.0] * 65],       # ragged
    [1.0] * 64,                             # 1-D
])
def test_predict_bad_shape(client, a):
    assert client.post("/predict", json={"a": a}).status_code == 422


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_predict_non_finite(client, bad):
    # Standard JSON has no NaN/Inf, but Python's parser accepts these literals.
    rows = ["[" + ",".join(["1.0"] * 64) + "]"] * 64
    rows[10] = "[" + ",".join([bad] + ["1.0"] * 63) + "]"
    r = client.post("/predict", content='{"a": [' + ",".join(rows) + "]}",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_predict_missing_body(client):
    assert client.post("/predict", json={}).status_code == 422
