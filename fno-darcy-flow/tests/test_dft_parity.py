import json

import numpy as np
import pytest
import torch

from app.model import build_random_checkpoint, load_model
from export.fno_dft import (FNODFT, SpectralConvDFT, build_dft, dft_matrices, pick_checkpoint, split_pair,
                            write_golden, write_samples, write_solver_fixtures)
from src.models.fno import SpectralConv2d

N = 128


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def test_dft_matrices_shapes_and_bounds():
    d = dft_matrices(N, 12)
    assert d["cw"].shape == (N, 12) and d["ch"].shape == (24, N)
    assert d["gc"].shape == (N, 24) and d["ar"].shape == (12, N)
    assert torch.all(d["ai"][0] == 0)  # Im of the DC column is dropped, like irfft
    with pytest.raises(ValueError):
        dft_matrices(16, 9)


def test_spectral_conv_dft_matches_fft():
    torch.manual_seed(0)
    conv = SpectralConv2d(4, 5, 6, 6)
    x = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        ref, out = conv(x), SpectralConvDFT(conv, 32)(x)
    assert rel_l2(out.numpy(), ref.numpy()) < 1e-6


def test_dft_model_matches_original_on_20_samples(ckpt):
    model, loaded = build_dft(ckpt)
    assert isinstance(model, FNODFT)
    for i in range(20):
        a, _ = split_pair(i)
        with torch.no_grad():
            u = model(torch.from_numpy(a)[None, None])[0, 0].numpy()
        assert rel_l2(u, loaded.predict(a)) < 1e-5, f"sample {i}"


def test_pick_checkpoint(tmp_path, ckpt):
    assert pick_checkpoint(tmp_path / "missing.pt", ckpt) == ckpt
    assert pick_checkpoint(ckpt, tmp_path / "missing.pt") == ckpt
    with pytest.raises(FileNotFoundError):
        pick_checkpoint(tmp_path / "a.pt", tmp_path / "b.pt")


def test_split_pair_is_test_split_and_exact_solve():
    a, u = split_pair(3, n=32)
    assert a.dtype == u.dtype == np.float32 and a.shape == u.shape == (32, 32)
    assert set(np.unique(a)) <= {np.float32(0.1), np.float32(1.0)}
    assert (u > 0).all()


def test_write_samples_layout(tmp_path):
    meta = write_samples(tmp_path, count=3, n=16)
    raw = np.fromfile(tmp_path / "samples.bin", dtype="<f4").reshape(3, 2, 16, 16)
    a, u = split_pair(2, n=16)
    np.testing.assert_array_equal(raw[2, 0], a)
    np.testing.assert_array_equal(raw[2, 1], u)
    assert json.loads((tmp_path / "samples.json").read_text()) == meta and meta["grid"] == 16


def test_write_golden_is_original_model_output(tmp_path, ckpt):
    loaded = load_model(ckpt)
    g = json.loads(write_golden(tmp_path, loaded, index=1, n=32).read_text())
    a, _ = split_pair(1, n=32)
    np.testing.assert_array_equal(np.float32(g["a"]), a.ravel())
    np.testing.assert_allclose(np.float32(g["u"]), loaded.predict(a).ravel(), rtol=1e-6)


def test_write_solver_fixtures(tmp_path):
    cases = json.loads(write_solver_fixtures(tmp_path / "f.json", sizes=(8, 12)).read_text())
    assert [c["n"] for c in cases] == [8, 12]
    assert all(len(c["a"]) == len(c["u"]) == c["n"] ** 2 for c in cases)
