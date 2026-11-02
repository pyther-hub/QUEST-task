import json

import h5py
import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from app.model import build_random_checkpoint  # noqa: E402
from export.fno_dft import build_dft, export_onnx, split_pair  # noqa: E402
from export.fp16_onnx import (evaluate, fp16_model, fp16_weights_onnx, load_samples, rel_l2,  # noqa: E402
                              run_variant, summarize)

N = 32


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return build_random_checkpoint(d / "fno.pt", data_path=d / "darcy.npz", n_samples=8)


def test_rel_l2_and_summarize():
    ref = np.ones((2, 3, 3))
    e = rel_l2(ref * np.array([1.0, 1.1])[:, None, None], ref)
    np.testing.assert_allclose(e, [0.0, 0.1])
    assert summarize(e) == pytest.approx({"mean": 0.05, "median": 0.05, "p95": 0.095, "max": 0.1})


def test_run_variant_batch1_and_timing():
    calls = []
    preds, ms = run_variant(lambda x: calls.append(x.shape) or x * 2, np.ones((3, 4, 4)), warmup=1)
    assert preds.shape == (3, 4, 4) and (preds == 2).all() and ms >= 0
    assert calls == [(4, 4)] * 4  # 1 warmup + 3 timed, one sample at a time


def test_load_samples_from_split_and_hdf5(tmp_path):
    a, u = load_samples(None, 2)
    np.testing.assert_array_equal(a[1], split_pair(1)[0])
    path = tmp_path / "t.hdf5"
    with h5py.File(path, "w") as f:
        f["nu"] = np.arange(5 * 4, dtype=np.float32).reshape(5, 2, 2)
        f["tensor"] = -np.arange(5 * 4, dtype=np.float32).reshape(5, 1, 2, 2)
    a, u = load_samples(path, count=2, start=1, stop=4)
    assert a.shape == u.shape == (2, 2, 2) and a[0, 0, 0] == 4 and u[1, 0, 0] == -8


def test_fp16_model_is_half_and_close(ckpt):
    model, _ = build_dft(ckpt, N)
    half = fp16_model(model)
    assert all(b.dtype == torch.float16 for b in half.buffers()) and next(model.parameters()).dtype == torch.float32
    a = torch.from_numpy(split_pair(0, N)[0])[None, None]
    with torch.no_grad():
        ref, out = model(a), half(a.half()).float()
    assert rel_l2(out.numpy(), ref.numpy())[0] < 1e-2


def test_fp16_weights_onnx_halves_size_and_matches(ckpt, tmp_path):
    model, _ = build_dft(ckpt, N)
    f32 = export_onnx(model, tmp_path / "m.onnx", N)
    f16 = fp16_weights_onnx(f32, tmp_path / "m16.onnx")
    assert f16.stat().st_size < 0.6 * f32.stat().st_size
    x = split_pair(0, N)[0][None, None]
    run = lambda p: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(["u"], {"a": x})[0]
    assert rel_l2(run(f16), run(f32))[0] < 1e-2


def test_evaluate_report(ckpt, tmp_path):
    pairs = [split_pair(i, N) for i in range(3)]
    a, u = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    r = evaluate(ckpt, a, u, tmp_path, device="cpu")
    assert set(r["variants"]) == {"pytorch_fp32", "pytorch_fp16", "onnx_fp32", "onnx_fp16w"}
    assert r["variants"]["pytorch_fp32"]["mean_drop_vs_fp32"] == 0
    assert r["variants"]["onnx_fp32"]["max_relL2_vs_fp32_pred"] < 1e-5
    assert json.loads((tmp_path / "export_eval.json").read_text()) == r
    assert (tmp_path / "model.onnx").exists() and (tmp_path / "model_fp16w.onnx").exists()
