"""Extra models for the browser app's model toggle, plus web/models.json (the toggle's manifest).

    python -m export.web_models     # after python -m export.fno_dft

web/model.onnx + golden.json (from export.fno_dft, the served checkpoint) stay the "v1" entry, so nothing that
reads them changes. Each spec in SPECS whose checkpoint exists adds web/models/<id>/model.onnx (fp32),
model_fp16w.onnx (fp16 weights, fp32 compute) and golden.json (original FNO2d output, src.infer path).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.model import DEFAULT_CHECKPOINT
from export.fno_dft import GRID, WEB, build_dft, export_onnx, write_golden
from export.fp16_onnx import fp16_weights_onnx

CKPT_DIR = DEFAULT_CHECKPOINT.parent
# Self-test tolerance (relL2 vs the fp32 PyTorch golden). fp16 weights measured 3.1e-4 max over 500 test samples.
TOL_FP32, TOL_FP16W = 1e-4, 1e-3

# id, label, checkpoint, eval.json with the training run's test split numbers
SPECS = [
    ("v2", "v2 · 148 epochs", CKPT_DIR / "fno_darcy_v2.pt", CKPT_DIR / "fno_darcy_v2.eval.json"),
]
V1 = ("v1", "v1 · 47 epochs", CKPT_DIR / "fno_darcy_v1.eval.json")


def test_relL2(eval_json: Path) -> float | None:
    """Mean test relL2 from a notebook-02 eval.json, or None when the file is missing."""
    p = Path(eval_json)
    return json.loads(p.read_text())["splits"]["test"]["mean"] if p.exists() else None


def entry(id_: str, label: str, precision: str, file: Path, golden: Path, loaded, web: Path, test: float | None) -> dict:
    return {"id": id_, "label": label, "precision": precision,
            "file": file.relative_to(web).as_posix(), "golden": golden.relative_to(web).as_posix(),
            "tol": TOL_FP16W if precision == "fp16 weights" else TOL_FP32,
            "trained": loaded.trained, "params": loaded.n_params,
            "file_mb": round(file.stat().st_size / 1e6, 2), "test_relL2": test}


def build_model(id_: str, label: str, checkpoint: Path, eval_json: Path, web: Path = WEB, n: int = GRID) -> list[dict]:
    """Export one checkpoint as fp32 + fp16-weight ONNX with its golden; returns its two manifest entries."""
    out = Path(web) / "models" / id_
    model, loaded = build_dft(checkpoint, n)
    f32 = export_onnx(model, out / "model.onnx", n)
    f16 = fp16_weights_onnx(f32, out / "model_fp16w.onnx")
    golden = write_golden(out, loaded, n=n)
    test = test_relL2(eval_json)
    return [entry(id_, label, "fp32", f32, golden, loaded, Path(web), test),
            entry(f"{id_}-fp16w", label, "fp16 weights", f16, golden, loaded, Path(web), test)]


def write_manifest(web: Path = WEB, specs=SPECS, v1=V1, n: int = GRID) -> dict:
    """web/models.json: one entry per selectable model, best (lowest test relL2) first and marked default."""
    web = Path(web)
    base = json.loads((web / "model.json").read_text())
    # web/model.onnx is whatever export.fno_dft served; it is "v1" only while that checkpoint is v1's bytes.
    served, v1_ckpt = CKPT_DIR / base["checkpoint"], Path(v1[2]).with_name("fno_darcy_v1.pt")
    is_v1 = base["trained"] and served.exists() and v1_ckpt.exists() and served.read_bytes() == v1_ckpt.read_bytes()
    label = v1[1] if is_v1 else ("served · " + base["checkpoint"] if base["trained"] else "untrained (random weights)")
    models = [{"id": "served" if not is_v1 else v1[0], "label": label, "precision": "fp32",
               "file": "model.onnx", "golden": "golden.json", "tol": TOL_FP32, "trained": base["trained"],
               "params": base["params"], "file_mb": round((web / "model.onnx").stat().st_size / 1e6, 2),
               "test_relL2": test_relL2(v1[2]) if is_v1 else None}]
    for id_, label, ckpt, ev in specs:
        if Path(ckpt).exists():
            models += build_model(id_, label, ckpt, ev, web, n)
    rank = lambda m: (not m["trained"], np.inf if m["test_relL2"] is None else m["test_relL2"], m["precision"] != "fp32")
    models.sort(key=rank)
    manifest = {"grid": n, "default": models[0]["id"], "models": models}
    (web / "models.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    m = write_manifest()
    for e in m["models"]:
        print(f"{e['id']:10s} {e['precision']:13s} {e['file_mb']:5.2f} MB  test relL2 {e['test_relL2']}  {e['file']}")
