"""deploy/make_space.sh assembles the Docker build context / Hugging Face Space folder."""

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from app.model import build_random_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "make_space.sh"


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


def make_space(*args, env_dir=None):
    env = {**os.environ, "FNO_DATA_DIR": str(env_dir)} if env_dir else None
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], capture_output=True, text=True, env=env)


def test_assembles_build_context(ckpt, tmp_path):
    out = tmp_path / "space"
    r = make_space(out, ckpt)
    assert r.returncode == 0, r.stderr
    assert (out / "model" / "fno_darcy.pt").read_bytes() == ckpt.read_bytes()
    assert (out / "app" / "static" / "index.html").exists() and (out / "src" / "data" / "darcy.py").exists()
    assert not list(out.rglob("__pycache__"))
    # Every COPY source in the Dockerfile exists in the assembled folder.
    dockerfile = (out / "Dockerfile").read_text()
    sources = [line.split()[-2] for line in dockerfile.splitlines() if line.startswith("COPY")]
    assert sources and all((out / s).exists() for s in sources)
    readme = (out / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme and "ENV" in dockerfile


def test_defaults_come_from_data_dir(ckpt, tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "fno_darcy.pt").write_bytes(ckpt.read_bytes())
    r = make_space(env_dir=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "space" / "model" / "fno_darcy.pt").exists()


def test_refuses_non_empty_dir_and_missing_checkpoint(ckpt, tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    r = make_space(tmp_path, ckpt)
    assert r.returncode != 0 and "not empty" in r.stderr
    assert (tmp_path / "keep.txt").read_text() == "x"
    r = make_space(tmp_path / "new", tmp_path / "missing.pt")
    assert r.returncode != 0 and "no checkpoint" in r.stderr


def test_app_runs_from_assembled_folder(ckpt, tmp_path):
    out = tmp_path / "space"
    assert make_space(out, ckpt).returncode == 0
    code = ("from fastapi.testclient import TestClient\n"
            "from app.main import app\n"
            "with TestClient(app) as c:\n"
            "    h = c.get('/health').json()\n"
            "    assert h['model_loaded'] and h['grid'] == 128, h\n"
            "    a = c.get('/sample', params={'index': 1}).json()['a']\n"
            "    assert c.post('/predict', json={'a': a}).status_code == 200\n"
            "    assert c.post('/solve', json={'a': a}).status_code == 200\n"
            "    print('ok')\n")
    env = {**os.environ, "FNO_CHECKPOINT": str(out / "model" / "fno_darcy.pt"), "PYTHONPATH": ""}
    r = subprocess.run([sys.executable, "-c", code], cwd=out, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_pinned_versions_match_tested_environment():
    pins = dict(re.findall(r"^([\w-]+)==(\S+)$", (ROOT / "deploy" / "requirements.txt").read_text(), re.M))
    assert set(pins) == {"numpy", "scipy", "fastapi", "uvicorn"}
    for name, version in pins.items():
        assert importlib.metadata.version(name) == version, name
    torch_pin = re.search(r"torch==(\S+)", (ROOT / "deploy" / "Dockerfile").read_text()).group(1)
    assert torch.__version__.split("+")[0] == torch_pin
