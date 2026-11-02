import numpy as np
from src.data.darcy import gaussian_random_field, permeability_from_grf, solve_darcy


def test_solver_positive_and_symmetric():
    """Constant permeability with symmetric forcing gives a positive,
    symmetric solution peaked at the domain centre."""
    n = 32
    u = solve_darcy(np.full((n, n), 5.0))
    assert (u > 0).all()
    assert np.allclose(u, u.T, atol=1e-10)
    assert u[n // 2, n // 2] == u.max() or u[n // 2 - 1, n // 2 - 1] == u.max()


def test_solver_scales_inversely_with_permeability():
    n = 32
    u1 = solve_darcy(np.full((n, n), 1.0))
    u2 = solve_darcy(np.full((n, n), 2.0))
    assert np.allclose(u1, 2.0 * u2, rtol=1e-8)


def test_grf_and_threshold():
    grf = gaussian_random_field(64, rng=np.random.default_rng(0))
    a = permeability_from_grf(grf)
    assert set(np.unique(a)) == {3.0, 12.0}


# --- PDEBench comparison ----------------------------------------------------

import h5py
import pytest

from src.data import darcy


def _fake_pdebench(path, n=3, grid=16):
    """Tiny file in PDEBench's Darcy layout: nu (N, n, n), tensor (N, 1, n, n)."""
    rng = np.random.default_rng(1)
    a = np.stack([permeability_from_grf(gaussian_random_field(grid, rng=rng)) for _ in range(n)])
    u = np.stack([solve_darcy(ai) for ai in a])[:, None]
    with h5py.File(path, "w") as f:
        f["nu"] = a.astype(np.float32)
        f["tensor"] = u.astype(np.float32)
    return a, u[:, 0]


def test_load_pdebench_reads_nu_and_tensor(tmp_path):
    a, u = _fake_pdebench(tmp_path / "pb.hdf5")
    a2, u2 = darcy.load_pdebench(tmp_path / "pb.hdf5", n_samples=2)
    assert a2.shape == u2.shape == (2, 16, 16)
    assert np.allclose(a2, a[:2]) and np.allclose(u2, u[:2], rtol=1e-6)


def test_download_pdebench_checks_md5_and_skips_existing(tmp_path):
    src = tmp_path / "src.bin"; src.write_bytes(b"darcy")
    md5 = darcy.md5sum(src)
    out = darcy.download_pdebench(tmp_path / "dl", url=src.as_uri(), md5=md5, name="x.hdf5")
    assert out.read_bytes() == b"darcy"
    src.unlink()  # second call must not re-download
    assert darcy.download_pdebench(tmp_path / "dl", url="file:///missing", md5=md5, name="x.hdf5") == out


def test_download_pdebench_rejects_bad_md5(tmp_path):
    src = tmp_path / "src.bin"; src.write_bytes(b"darcy")
    with pytest.raises(ValueError, match="MD5"):
        darcy.download_pdebench(tmp_path / "dl", url=src.as_uri(), md5="0" * 32, name="x.hdf5")
    assert not any((tmp_path / "dl").iterdir())


def test_block_mean():
    x = np.arange(16.0).reshape(1, 4, 4)
    assert np.allclose(darcy.block_mean(x, 2), [[[2.5, 4.5], [10.5, 12.5]]])


def test_field_stats():
    a = np.array([[[3.0, 12.0], [12.0, 12.0]]]); u = np.array([[[0.0, 1.0], [2.0, 3.0]]])
    s = darcy.field_stats(a, u)
    assert s["a_unique"] == 2 and s["a_high_fraction"] == 0.75
    assert s["u_max"] == 3.0 and s["u_peak_mean"] == 3.0 and s["grid"] == 2


def test_solver_agreement_is_zero_on_own_solutions():
    a = np.full((2, 8, 8), 4.0); u = np.stack([solve_darcy(ai) for ai in a])
    assert np.allclose(darcy.solver_agreement(a, u), 0.0)


def test_compare_pdebench_and_plot(tmp_path):
    _fake_pdebench(tmp_path / "pb.hdf5", n=2, grid=16)
    r = darcy.compare_pdebench(tmp_path / "pb.hdf5", n_samples=2, grid=8, plot=tmp_path / "c.png")
    assert r["solver_rel_l2_max"] < 1e-5  # fake data was made with our solver (float32 storage)
    assert r["pdebench"]["grid"] == r["generated"]["grid"] == 8
    assert r["pdebench_native"]["grid"] == 16
    assert (tmp_path / "c.png").stat().st_size > 0


def test_data_root_from_env(tmp_path):
    import subprocess, sys, os
    env = {**os.environ, "FNO_DATA_DIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", "from src.data.darcy import DATA_ROOT; print(DATA_ROOT)"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(tmp_path)
    assert darcy.DATA_ROOT.is_absolute() and "fno-darcy-flow" not in str(darcy.DATA_ROOT)


def test_solver_pdebench_discretisation():
    """h=1/n, arithmetic faces: constant a reduces to the same Poisson problem; bad face raises."""
    n = 16
    a = np.full((n, n), 2.0)
    u_ar = solve_darcy(a, h=1 / n, face="arithmetic")
    assert np.allclose(u_ar, solve_darcy(a, h=1 / n))  # constant a: averages agree
    assert u_ar.max() > solve_darcy(a).max()            # larger h -> larger u
    with pytest.raises(ValueError):
        solve_darcy(a, face="geometric")
