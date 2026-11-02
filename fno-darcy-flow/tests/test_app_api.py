import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import GRID, TEST_SEED, TEST_SIZE, app, reference, sample
from app.model import build_random_checkpoint, load_model
from src.data.darcy import solve_darcy, split_sample


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


def ones(n=GRID, value=1.0):
    return [[value] * n for _ in range(n)]


def test_sample_is_test_split_with_pdebench_solver():
    a, u = sample(3)
    a_ref = split_sample(3, TEST_SIZE, TEST_SEED, GRID)
    np.testing.assert_array_equal(a, a_ref.astype(np.float32))
    np.testing.assert_allclose(u, solve_darcy(a_ref, h=1 / GRID, face="arithmetic"), rtol=1e-5)


def test_reference_is_pdebench_discretisation():
    a = np.full((GRID, GRID), 0.1)
    np.testing.assert_allclose(reference(a), solve_darcy(a, h=1 / GRID, face="arithmetic"))


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "untrained (random weights)" in r.text


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["trained"] is False
    assert body["params"] == 2_368_001
    assert body["grid"] == GRID == 128


def test_health_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FNO_CHECKPOINT", str(tmp_path / "missing.pt"))
    with TestClient(app) as c:
        body = c.get("/health").json()
        assert body["model_loaded"] is False and body["trained"] is None
        assert c.post("/predict", json={"a": ones()}).status_code == 503
        assert c.post("/solve", json={"a": ones()}).status_code == 200  # solver needs no model


def test_sample_endpoint(client):
    body = client.get("/sample", params={"index": 5}).json()
    a, u = np.array(body["a"]), np.array(body["u"])
    assert body["index"] == 5 and a.shape == u.shape == (GRID, GRID)
    assert set(np.unique(a.astype(np.float32))) <= {np.float32(0.1), np.float32(1.0)}
    assert client.get("/sample", params={"index": 5}).json() == body


@pytest.mark.parametrize("index", [-1, TEST_SIZE])
def test_sample_rejects_out_of_range(client, index):
    assert client.get("/sample", params={"index": index}).status_code == 422


def test_predict(client, ckpt):
    a = client.get("/sample", params={"index": 7}).json()["a"]
    r = client.post("/predict", json={"a": a})
    assert r.status_code == 200
    u = np.array(r.json()["u"])
    assert u.shape == (GRID, GRID)
    assert r.json()["ms"] > 0
    np.testing.assert_allclose(u, load_model(ckpt).predict(a), rtol=1e-6)


def test_solve(client):
    a = client.get("/sample", params={"index": 7}).json()
    r = client.post("/solve", json={"a": a["a"]})
    assert r.status_code == 200 and r.json()["ms"] > 0
    np.testing.assert_allclose(np.array(r.json()["u"]), np.array(a["u"]), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_solve_rejects_non_positive(client, value):
    a = ones()
    a[3][4] = value
    assert client.post("/solve", json={"a": a}).status_code == 422


@pytest.mark.parametrize("a", [
    ones()[:-1],                            # too few rows
    [[1.0] * (GRID - 1)] * GRID,            # too few cols
    ones()[:-1] + [[1.0] * (GRID + 1)],     # ragged
    [1.0] * GRID,                           # 1-D
    ones(64),                               # the old 64x64 grid
])
@pytest.mark.parametrize("path", ["/predict", "/solve"])
def test_bad_shape(client, path, a):
    assert client.post(path, json={"a": a}).status_code == 422


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
@pytest.mark.parametrize("path", ["/predict", "/solve"])
def test_non_finite(client, path, bad):
    # Standard JSON has no NaN/Inf, but Python's parser accepts these literals.
    rows = ["[" + ",".join(["1.0"] * GRID) + "]"] * GRID
    rows[10] = "[" + ",".join([bad] + ["1.0"] * (GRID - 1)) + "]"
    r = client.post(path, content='{"a": [' + ",".join(rows) + "]}",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("path", ["/predict", "/solve"])
def test_missing_body(client, path):
    assert client.post(path, json={}).status_code == 422
