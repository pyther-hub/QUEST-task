"""Tests for the Kaggle notebooks: the scripts they write via %%writefile are extracted
from the notebook JSON and run on CPU with tiny settings."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from app.model import load_model

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
GEN_NB, TRAIN_NB = NB / "01_generate_dataset.ipynb", NB / "02_train_ddp.ipynb"
TINY_SPLITS = "train:8:0,val:4:1,test:4:2"
GRID = 16


def code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def extract_script(nb_path: Path, name: str, dest: Path) -> Path:
    """Write the body of the notebook's `%%writefile <name>` cell to dest/<name>."""
    header = f"%%writefile {name}\n"
    (src,) = [c for c in code_cells(nb_path) if c.startswith(header)]
    path = dest / name
    path.write_text(src[len(header):])
    return path


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "FNO_REPO": str(ROOT)}
    proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def generate(tmp: Path, out: str, workers: int) -> Path:
    script = extract_script(GEN_NB, "gen_darcy.py", tmp)
    run([sys.executable, str(script), "--splits", TINY_SPLITS, "--grid", str(GRID),
         "--workers", str(workers), "--chunk-size", "3", "--out-dir", out], tmp)
    return tmp / out


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gen")
    return generate(tmp, "data", workers=2)


def test_notebooks_layout():
    assert sorted(p.name for p in NB.glob("*.ipynb")) == [GEN_NB.name, TRAIN_NB.name]
    for nb in (GEN_NB, TRAIN_NB):
        assert "CONFIG" in code_cells(nb)[0]  # important settings come first


def read_h5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {"a": f["nu"][:], "u": f["tensor"][:, 0], "x": f["x-coordinate"][:],
                "y": f["y-coordinate"][:], "beta": f.attrs["beta"], "tensor_shape": f["tensor"].shape}


def test_generated_splits(tiny_data):
    sizes = {"train": 8, "val": 4, "test": 4}
    loaded = {s: read_h5(tiny_data / f"darcy_{s}.hdf5") for s in sizes}
    for s, n in sizes.items():
        a, u = loaded[s]["a"], loaded[s]["u"]
        assert a.shape == u.shape == (n, GRID, GRID)
        assert loaded[s]["tensor_shape"] == (n, 1, GRID, GRID)  # PDEBench layout
        assert loaded[s]["beta"] == 1.0
        np.testing.assert_allclose(loaded[s]["x"], (np.arange(GRID) + 0.5) / GRID)
        assert a.dtype == u.dtype == np.float32
        assert set(np.unique(a)) == {np.float32(0.1), np.float32(1.0)}
        assert np.isfinite(u).all() and (u >= 0).all()
    hashes = [{x.tobytes() for x in loaded[s]["a"]} for s in sizes]
    assert all(len(h) == n for h, n in zip(hashes, sizes.values()))  # no duplicates
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    meta = json.loads((tiny_data / "meta.json").read_text())
    assert meta["splits"]["train"] == {**meta["splits"]["train"], "n": 8, "seed": 0}


def test_generation_independent_of_workers(tiny_data, tmp_path):
    serial = generate(tmp_path, "serial", workers=1)
    for s in ("train", "val", "test"):
        a, b = read_h5(tiny_data / f"darcy_{s}.hdf5"), read_h5(serial / f"darcy_{s}.hdf5")
        np.testing.assert_array_equal(a["a"], b["a"])
        np.testing.assert_array_equal(a["u"], b["u"])


def gen_module(tmp: Path):
    import importlib.util
    path = extract_script(GEN_NB, "gen_darcy.py", tmp)
    spec = importlib.util.spec_from_file_location("gen_darcy_nb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gen_solver_matches_src_with_pdebench_discretisation(tmp_path):
    from src.data.darcy import solve_darcy
    gen = gen_module(tmp_path)
    a = gen.sample_permeability(3, GRID, np.random.default_rng(0))
    for ai in a:
        np.testing.assert_allclose(gen.solve(ai), solve_darcy(ai, h=1 / GRID, face="arithmetic"),
                                   rtol=1e-10)


def test_gen_sampler_matches_src(tmp_path):
    from src.data.darcy import bumps_permeability, split_sample
    gen = gen_module(tmp_path)
    np.testing.assert_array_equal(gen.sample_permeability(7, GRID, np.random.default_rng([3, 1])),
                                  bumps_permeability(7, GRID, np.random.default_rng([3, 1])))
    split = gen.sample_split(450, 2, GRID, 200)  # last chunk is short (50 samples)
    for i in (0, 199, 200, 449):
        np.testing.assert_array_equal(split_sample(i, 450, 2, GRID).astype(np.float32), split[i])


def test_gen_eta_helpers(tmp_path):
    gen = gen_module(tmp_path)
    assert gen.fmt_time(5) == "0m05s" and gen.fmt_time(3725) == "1h02m05s"
    eta = gen.estimate(GRID, n_total=1000, workers=1, per_worker=2)
    assert 0 < eta < 60


def test_gen_prints_progress_with_eta(tmp_path):
    script = extract_script(GEN_NB, "gen_darcy.py", tmp_path)
    proc = run([sys.executable, str(script), "--splits", "train:4:0", "--grid", str(GRID),
                "--workers", "1", "--chunk-size", "2", "--block", "1", "--out-dir", "d"], tmp_path)
    assert "estimated" in proc.stdout
    assert "ETA" in proc.stdout and "4/4 (100.0%)" in proc.stdout


def data_files(data_dir: Path) -> dict:
    return {s: [str(data_dir / f"darcy_{s}.hdf5"), 0, None] for s in ("train", "val", "test")}


def write_config(tmp: Path, data_dir: Path, mode: str, **overrides) -> Path:
    cfg = {"mode": mode, "modes": 4, "width": 8, "layers": 2, "epochs": 2, "batch_size": 2,
           "grad_accum": 1, "lr": 1e-3, "weight_decay": 1e-4, "amp": True, "seed": 0,
           "data_on_gpu": True, "eval_every": 1, "gpu_poll_s": 1.0,
           "precheck_batch_sizes": [1, 2], "precheck_warmup": 1, "precheck_steps": 2,
           "precheck_min_seconds": 0, "data_files": data_files(data_dir), "out_dir": str(tmp / "run"),
           **overrides}
    path = tmp / f"cfg_{mode}.json"
    path.write_text(json.dumps(cfg))
    return path


def check_train_outputs(out: Path, epochs: int):
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [int(r["epoch"]) for r in rows] == list(range(1, epochs + 1))
    for r in rows:
        assert float(r["val_relL2"]) > 0 and float(r["train_relL2_eval"]) > 0
        assert float(r["samples_per_s"]) > 0
    ev = json.loads((out / "eval.json").read_text())
    assert set(ev["splits"]) == {"train", "val", "test"}
    assert ev["splits"]["test"]["n"] == 4
    per = np.load(out / "per_sample_relL2.npz")
    assert per["train"].shape == (8,)
    np.testing.assert_allclose(per["test"].mean(), ev["splits"]["test"]["mean"], rtol=1e-6)
    assert "eval best.pt" in (out / "train.log").read_text()

    ckpt = torch.load(out / "best.pt")
    assert set(ckpt) == {"model", "config", "stats"}
    assert ckpt["config"] == {"modes": 4, "width": 8, "layers": 2}
    m = load_model(out / "best.pt")  # the app's loader
    path, lo, hi = ev["config"]["data_files"]["test"]
    with h5py.File(path, "r") as f:
        a = f["nu"][lo]
    u = m.predict(a)
    assert u.shape == (GRID, GRID) and np.isfinite(u).all()
    assert (out / "last.pt").exists()


def test_train_single_process(tiny_data, tmp_path):
    script = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    run([sys.executable, str(script), "--config",
         str(write_config(tmp_path, tiny_data, "train"))], tmp_path)
    check_train_outputs(tmp_path / "run", epochs=2)


def test_train_ddp_two_processes(tiny_data, tmp_path):
    """DDP path (gloo on CPU) with grad accumulation, i.e. no_sync on non-step micro-batches."""
    script = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    cfg = write_config(tmp_path, tiny_data, "train", grad_accum=2, batch_size=1)
    proc = run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", str(script), "--config", str(cfg)], tmp_path)
    check_train_outputs(tmp_path / "run", epochs=2)
    assert "world=2" in proc.stdout


def test_precheck(tiny_data, tmp_path):
    script = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    proc = run([sys.executable, str(script), "--config",
                str(write_config(tmp_path, tiny_data, "precheck"))], tmp_path)
    out = tmp_path / "run"
    rows = list(csv.DictReader(open(out / "precheck.csv")))
    assert [int(r["batch_size_per_gpu"]) for r in rows] == [1, 2]
    assert all(r["status"] == "ok" and float(r["samples_per_s"]) > 0 for r in rows)
    assert "recommended BATCH_SIZE" in proc.stdout
    assert "gpu monitor=off" in proc.stdout  # no GPU here: monitor is a no-op
    assert not (out / "best.pt").exists() and not (out / "precheck_gpu_log.csv").exists()


def test_train_on_pdebench_style_single_file(tiny_data, tmp_path):
    """DATA_SOURCE="pdebench": one HDF5 file, splits are index ranges."""
    parts = [read_h5(tiny_data / f"darcy_{s}.hdf5") for s in ("train", "val", "test")]
    one = tmp_path / "2D_DarcyFlow_beta1.0_Train.hdf5"
    with h5py.File(one, "w") as f:
        f["nu"] = np.concatenate([p["a"] for p in parts])
        f["tensor"] = np.concatenate([p["u"] for p in parts])[:, None]
    files = {"train": [str(one), 0, 8], "val": [str(one), 8, 12], "test": [str(one), 12, 16]}
    script = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    run([sys.executable, str(script), "--config",
         str(write_config(tmp_path, tiny_data, "train", data_files=files))], tmp_path)
    check_train_outputs(tmp_path / "run", epochs=2)


def test_train_early_stopping_and_gpu_report(tiny_data, tmp_path):
    """lr=0 -> val never improves after epoch 1 -> patience 3 stops at epoch 4 of 20."""
    script = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    run([sys.executable, str(script), "--config",
         str(write_config(tmp_path, tiny_data, "train", epochs=20, lr=0.0, weight_decay=0.0,
                          patience=3, gpu_report_epochs=2))], tmp_path)
    out = tmp_path / "run"
    check_train_outputs(out, epochs=4)
    ev = json.loads((out / "eval.json").read_text())
    assert ev["stopped_early"] and ev["epochs_run"] == 4 and ev["best_epoch"] == 1
    assert [r["epoch"] for r in ev["gpu_first_epochs"]] == [1, 2]
    log = (out / "train.log").read_text()
    assert "early stopping at epoch 4" in log and "no improvement 3/3" in log
    assert "gpu usage epoch  1: n/a" in log and "gpu usage, first 2 epochs: n/a" in log


def test_gpu_summary_flags_low_utilisation(tmp_path, caplog):
    import importlib.util, logging
    path = extract_script(TRAIN_NB, "fno_ddp_train.py", tmp_path)
    spec = importlib.util.spec_from_file_location("fno_ddp_train_nb", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = [{"epoch": e, "util_pct": {0: 95.0, 1: 40.0 + e}, "mem_used_mb": {0: 3000, 1: 2900 + e},
               "torch_peak_mb": 1.0} for e in (1, 2)]
    log = logging.getLogger("t_gpu"); log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="t_gpu"):
        mod.log_gpu_summary(report, log)
    text = caplog.text
    assert "GPU0 mean util 95.0% (min 95%)  max mem 3000 MB" in text
    assert "GPU1 mean util 41.5% (min 41%)  max mem 2902 MB" in text
    assert "GPU1 is under 70% busy" in text and "GPU0 is under" not in text
