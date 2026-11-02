import json
from pathlib import Path

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from app.model import build_random_checkpoint  # noqa: E402
from export.fno_dft import WEB, build_dft, export_onnx, split_pair  # noqa: E402


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    ckpt = build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)
    model, _ = build_dft(ckpt)
    return model, export_onnx(model, d / "model.onnx")


def test_onnx_static_shape_and_opset(exported):
    import onnx
    m = onnx.load(exported[1])
    assert m.opset_import[0].version == 17
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert m.graph.input[0].name == "a" and dims == [1, 1, 128, 128]


def test_onnxruntime_matches_pytorch(exported):
    model, path = exported
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for i in range(20):
        a, _ = split_pair(i)
        x = a[None, None]
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        assert rel_l2(sess.run(["u"], {"a": x})[0], ref) < 1e-5, f"sample {i}"


@pytest.mark.skipif(not (WEB / "model.onnx").exists(), reason="web/model.onnx not exported")
def test_web_golden_matches_web_model():
    g = json.loads((WEB / "golden.json").read_text())
    n = g["grid"]
    sess = ort.InferenceSession(str(WEB / "model.onnx"), providers=["CPUExecutionProvider"])
    u = sess.run(["u"], {"a": np.float32(g["a"]).reshape(1, 1, n, n)})[0].ravel()
    assert rel_l2(u, np.float32(g["u"])) < 1e-5
