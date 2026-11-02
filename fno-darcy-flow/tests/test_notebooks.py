"""Tests for the Kaggle notebooks: the scripts they write via %%writefile are extracted
from the notebook JSON and run on CPU with tiny settings."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from app.model import load_model

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
GEN_NB, TRAIN_NB = NB / "01_generate_dataset.ipynb", NB / "02_train_ddp.ipynb"
TINY_SPLITS = "train:8:0,val:4:1,test:4:2"


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
    run([sys.executable, str(script), "--splits", TINY_SPLITS, "--grid", "64",
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


def test_generated_splits(tiny_data):
    sizes = {"train": 8, "val": 4, "test": 4}
    loaded = {s: np.load(tiny_data / f"darcy_{s}.npz") for s in sizes}
    for s, n in sizes.items():
        a, u = loaded[s]["a"], loaded[s]["u"]
        assert a.shape == u.shape == (n, 64, 64)
        assert a.dtype == u.dtype == np.float32
        assert set(np.unique(a)) <= {3.0, 12.0}
        assert np.isfinite(u).all() and (u >= 0).all()
    hashes = [{x.tobytes() for x in loaded[s]["a"]} for s in sizes]
    assert all(len(h) == n for h, n in zip(hashes, sizes.values()))  # no duplicates
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    meta = json.loads((tiny_data / "meta.json").read_text())
    assert meta["splits"]["train"] == {**meta["splits"]["train"], "n": 8, "seed": 0}


def test_generation_independent_of_workers(tiny_data, tmp_path):
    serial = generate(tmp_path, "serial", workers=1)
    for s in ("train", "val", "test"):
        a, b = np.load(tiny_data / f"darcy_{s}.npz"), np.load(serial / f"darcy_{s}.npz")
        np.testing.assert_array_equal(a["a"], b["a"])
        np.testing.assert_array_equal(a["u"], b["u"])


def write_config(tmp: Path, data_dir: Path, mode: str, **overrides) -> Path:
    cfg = {"mode": mode, "modes": 4, "width": 8, "layers": 2, "epochs": 2, "batch_size": 2,
           "grad_accum": 1, "lr": 1e-3, "weight_decay": 1e-4, "amp": True, "seed": 0,
           "data_on_gpu": True, "eval_every": 1, "gpu_poll_s": 1.0,
           "precheck_batch_sizes": [1, 2], "precheck_warmup": 1, "precheck_steps": 2,
           "precheck_min_seconds": 0, "data_dir": str(data_dir), "out_dir": str(tmp / "run"),
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
    a = np.load(Path(ev["config"]["data_dir"]) / "darcy_test.npz")["a"][0]
    u = m.predict(a)
    assert u.shape == (64, 64) and np.isfinite(u).all()
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
