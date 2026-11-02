import numpy as np
import pytest
import torch

from app.model import SMALL_CONFIG, build_random_checkpoint, load_model
from src.data.darcy import gaussian_random_field, permeability_from_grf

SMALL_PARAMS = 2_368_001  # FNO2d(modes=12, width=32, layers=4)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    # Few samples keep the test fast; stats only need to be finite and positive.
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


@pytest.fixture(scope="module")
def a_field():
    return permeability_from_grf(gaussian_random_field(64, rng=np.random.default_rng(1)))


def test_checkpoint_format(ckpt):
    raw = torch.load(ckpt)
    assert set(raw) == {"model", "config", "stats"}
    assert raw["config"] == SMALL_CONFIG == {"modes": 12, "width": 32, "layers": 4}
    assert set(raw["stats"]) == {"a_mean", "a_std", "u_std"}
    assert raw["stats"]["a_std"] > 0 and raw["stats"]["u_std"] > 0


def test_build_is_seeded(ckpt, tmp_path):
    again = build_random_checkpoint(tmp_path / "fno.pt", data_path=ckpt.parent / "darcy.npz")
    w1, w2 = torch.load(ckpt)["model"], torch.load(again)["model"]
    assert all(torch.equal(w1[k], w2[k]) for k in w1)


def test_load_and_param_count(ckpt):
    m = load_model(ckpt)
    assert not m.model.training
    assert m.n_params == SMALL_PARAMS


def test_predict_shape_and_finite(ckpt, a_field):
    u = load_model(ckpt).predict(a_field)
    assert u.shape == (64, 64)
    assert np.isfinite(u).all()


def test_predict_matches_manual_preprocessing(ckpt, a_field):
    m = load_model(ckpt)
    s = m.stats
    a = (torch.tensor(a_field, dtype=torch.float32) - s["a_mean"]) / s["a_std"]
    with torch.no_grad():
        expected = (m.model(a.unsqueeze(0)) * s["u_std"])[0].numpy()
    np.testing.assert_allclose(m.predict(a_field), expected, rtol=1e-6)


def test_predict_deterministic(ckpt, a_field):
    m1, m2 = load_model(ckpt), load_model(ckpt)
    np.testing.assert_array_equal(m1.predict(a_field), m1.predict(a_field))
    np.testing.assert_array_equal(m1.predict(a_field), m2.predict(a_field))
