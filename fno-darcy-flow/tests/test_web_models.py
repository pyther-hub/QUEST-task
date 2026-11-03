import json
import shutil

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from app.model import build_random_checkpoint  # noqa: E402
from export import web_models  # noqa: E402
from export.fno_dft import WEB, build_dft, export_onnx, write_golden  # noqa: E402

N = 32


def run_onnx(path, a, n):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(["u"], {"a": np.float32(a).reshape(1, 1, n, n)})[0].ravel()


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


def test_test_relL2_reads_eval_json(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"splits": {"test": {"mean": 0.0164}}}))
    assert web_models.test_relL2(p) == 0.0164
    assert web_models.test_relL2(tmp_path / "missing.json") is None


@pytest.fixture
def web(tmp_path, ckpt, monkeypatch):
    """A tiny web/ dir as export.fno_dft leaves it, served from a random (untrained) checkpoint."""
    monkeypatch.setattr(web_models, "CKPT_DIR", ckpt.parent)
    model, loaded = build_dft(ckpt, N)
    export_onnx(model, tmp_path / "model.onnx", N)
    write_golden(tmp_path, loaded, n=N)
    (tmp_path / "model.json").write_text(json.dumps({"checkpoint": ckpt.name, "trained": False,
                                                     "params": loaded.n_params, "grid": N}))
    return tmp_path


def test_build_model_entries(web, ckpt):
    ev = web / "eval.json"
    ev.write_text(json.dumps({"splits": {"test": {"mean": 0.02}}}))
    f32, f16 = web_models.build_model("m", "m label", ckpt, ev, web, N)
    assert (f32["file"], f16["file"]) == ("models/m/model.onnx", "models/m/model_fp16w.onnx")
    assert f32["golden"] == f16["golden"] == "models/m/golden.json"
    assert (f32["tol"], f16["tol"]) == (web_models.TOL_FP32, web_models.TOL_FP16W)
    assert f16["file_mb"] < 0.6 * f32["file_mb"] and f32["test_relL2"] == 0.02 and f32["trained"] is False


def test_manifest_untrained_served_model_and_order(web, ckpt):
    other = ckpt.parent / "other.pt"
    shutil.copy(ckpt, other)
    m = web_models.write_manifest(web, specs=[("x", "x", other, web / "none.json"),
                                              ("gone", "gone", web / "missing.pt", web / "none.json")],
                                  v1=("v1", "v1", web / "none.json"), n=N)
    assert json.loads((web / "models.json").read_text()) == m
    ids = [e["id"] for e in m["models"]]
    assert ids == ["served", "x", "x-fp16w"] and m["default"] == "served"  # missing checkpoints are skipped
    assert m["models"][0]["label"] == "untrained (random weights)"
    for e in m["models"]:
        g = json.loads((web / e["golden"]).read_text())
        assert rel_l2(run_onnx(web / e["file"], g["a"], N), np.float32(g["u"])) < e["tol"], e["id"]


@pytest.mark.skipif(not (WEB / "models.json").exists(), reason="web/models.json not exported")
def test_web_manifest_models_pass_their_self_test():
    m = json.loads((WEB / "models.json").read_text())
    assert m["default"] in [e["id"] for e in m["models"]]
    for e in m["models"]:
        g = json.loads((WEB / e["golden"]).read_text())
        assert rel_l2(run_onnx(WEB / e["file"], g["a"], g["grid"]), np.float32(g["u"])) < e["tol"], e["id"]
