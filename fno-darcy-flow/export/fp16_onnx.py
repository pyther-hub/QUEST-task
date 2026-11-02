"""fp16 + ONNX conversion of a trained checkpoint, and how much accuracy / speed each step costs.

    python -m export.fp16_onnx --checkpoint best.pt --out runs/x/export                    # first 200 test-split samples
    python -m export.fp16_onnx --checkpoint best.pt --data darcy_test.hdf5 --out runs/x/export   # PDEBench layout file

Variants, all evaluated against the reference u (relL2 per sample):
    pytorch_fp32   original FNO2d via src.infer.predict (the baseline, what training reports)
    pytorch_fp16   FNODFT.half(): every op in fp16 (FNO2d.half() would need complex-half einsum)
    onnx_fp32      web/model.onnx equivalent on onnxruntime CPU
    onnx_fp16w     same graph with fp16 weights cast back to fp32 at load: half the file, fp32 compute
Writes model.onnx, model_fp16w.onnx and export_eval.json into --out.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from export.fno_dft import FNODFT, build_dft, export_onnx, split_pair

# Initializers smaller than this stay fp32 (scalars / shapes): casting them saves nothing.
MIN_FP16_ELEMENTS = 16


def fp16_weights_onnx(src: Path, dst: Path) -> Path:
    """Store every large float32 initializer as float16, followed by a Cast back to float32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    casts = []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        w = numpy_helper.to_array(init)
        if w.size < MIN_FP16_ELEMENTS:
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(w.astype(np.float16), name + "_fp16"))
        casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT,
                                      name=name + "_cast"))
    # Initializers have no producer, so the casts can lead the topological order.
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    dst = Path(dst)
    onnx.save(m, str(dst))
    onnx.checker.check_model(str(dst))
    return dst


def fp16_model(model: FNODFT) -> FNODFT:
    """A half-precision copy of an FNODFT (weights, DFT matrices and activations in fp16)."""
    return copy.deepcopy(model).half().eval()


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 over the last two axes."""
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    return np.linalg.norm((pred - ref).reshape(len(ref), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)


def summarize(errs: np.ndarray) -> dict:
    return {"mean": float(errs.mean()), "median": float(np.median(errs)),
            "p95": float(np.percentile(errs, 95)), "max": float(errs.max())}


def run_variant(fn, a: np.ndarray, warmup: int = 2) -> tuple[np.ndarray, float]:
    """Apply fn to one (n, n) sample at a time (batch 1, as deployed); returns preds and mean ms/sample."""
    for x in a[:warmup]:
        fn(x)
    t = time.perf_counter()
    preds = np.stack([fn(x) for x in a])
    return preds, (time.perf_counter() - t) * 1e3 / len(a)


def load_samples(data: Path | None, count: int, start: int = 0, stop: int | None = None):
    """(a, u) float32 from a PDEBench-layout HDF5 (nu, tensor[:, 0]) or, without data, the test split."""
    if data is None:
        pairs = [split_pair(i) for i in range(count)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    import h5py
    with h5py.File(data, "r") as f:
        stop = f["nu"].shape[0] if stop is None else stop
        stop = min(stop, start + count)
        return f["nu"][start:stop].astype(np.float32), f["tensor"][start:stop, 0].astype(np.float32)


def evaluate(checkpoint: Path, a: np.ndarray, u: np.ndarray, out: Path, device: str | None = None) -> dict:
    import onnxruntime as ort

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = a.shape[-1]
    model, loaded = build_dft(checkpoint, n)
    onnx32 = export_onnx(model, out / "model.onnx", n)
    onnx16 = fp16_weights_onnx(onnx32, out / "model_fp16w.onnx")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    half = fp16_model(model).to(dev)

    @torch.no_grad()
    def torch_fp16(x):
        return half(torch.from_numpy(x)[None, None].to(dev).half())[0, 0].float().cpu().numpy()

    def ort_fn(path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return lambda x: sess.run(["u"], {"a": x[None, None]})[0][0, 0]

    variants = {
        "pytorch_fp32": (loaded.predict, checkpoint, "cpu"),
        "pytorch_fp16": (torch_fp16, None, dev.type),
        "onnx_fp32": (ort_fn(onnx32), onnx32, "cpu"),
        "onnx_fp16w": (ort_fn(onnx16), onnx16, "cpu"),
    }
    preds, report = {}, {"checkpoint": str(checkpoint), "trained": loaded.trained, "n_samples": len(a),
                         "grid": n, "variants": {}}
    for name, (fn, path, where) in variants.items():
        preds[name], ms = run_variant(fn, a)
        report["variants"][name] = {"relL2_vs_reference": summarize(rel_l2(preds[name], u)),
                                    "ms_per_sample": ms, "device": where,
                                    "file_mb": Path(path).stat().st_size / 1e6 if path else None}
    base = report["variants"]["pytorch_fp32"]["relL2_vs_reference"]["mean"]
    for name, r in report["variants"].items():
        r["mean_drop_vs_fp32"] = r["relL2_vs_reference"]["mean"] - base  # > 0 = less accurate
        r["max_relL2_vs_fp32_pred"] = float(rel_l2(preds[name], preds["pytorch_fp32"]).max())
    (out / "export_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def print_report(r: dict) -> None:
    print(f"{r['n_samples']} samples, {r['grid']}x{r['grid']}, {r['checkpoint']}")
    print(f"{'variant':14s} {'mean relL2':>11s} {'max':>8s} {'drop vs fp32':>13s} {'vs fp32 pred':>13s} "
          f"{'ms/sample':>10s} {'MB':>6s}")
    for name, v in r["variants"].items():
        s = v["relL2_vs_reference"]
        mb = f"{v['file_mb']:.1f}" if v["file_mb"] else "-"
        print(f"{name:14s} {s['mean']:11.5f} {s['max']:8.4f} {v['mean_drop_vs_fp32']:+13.2e} "
              f"{v['max_relL2_vs_fp32_pred']:13.2e} {v['ms_per_sample']:7.1f} {v['device']:3s} {mb:>5s}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None, help="PDEBench-layout HDF5; default: notebook-01 test split")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--count", type=int, default=200, help="max samples to evaluate")
    p.add_argument("--device", default=None, help="device for pytorch_fp16 (default: cuda if available)")
    args = p.parse_args()
    a, u = load_samples(args.data, args.count, args.start, args.stop)
    print_report(evaluate(args.checkpoint, a, u, args.out, args.device))


if __name__ == "__main__":
    main()
