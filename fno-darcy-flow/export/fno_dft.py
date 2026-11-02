"""FNO without FFTs, for the static browser app (onnxruntime-web has no rfft2/irfft2).

    python -m export.fno_dft      # writes web/model.onnx, model.json, golden.json, samples.*, solver fixtures

FNODFT rebuilds src.models.fno.FNO2d with the same weights at a fixed grid n, replacing
rfft2 / irfft2 by precomputed real DFT matrices restricted to the modes SpectralConv2d keeps
(rows [:m] and [-m:], columns [:m] of the rfft half-spectrum). The src.infer preprocessing
is baked in: input is raw a(x) of shape (1, 1, n, n), output is u(x) in physical units.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.model import DEFAULT_CHECKPOINT, GRID, RANDOM_CHECKPOINT, load_model
from src.data.darcy import BUMPS_HI, BUMPS_LO, split_sample, solve_darcy
from src.models.fno import FNO2d, SpectralConv2d

WEB = Path(__file__).resolve().parents[1] / "web"
# Held-out test split of notebooks/01_generate_dataset.ipynb, same as app/main.py.
TEST_SIZE, TEST_SEED = 500, 2
OPSET = 17


def dft_matrices(n: int, m: int) -> dict[str, torch.Tensor]:
    """Real DFT matrices (float64) for an n x n grid keeping m modes per corner.

    Forward:  Y = x @ (cw + i sw)              (n, m)   along the last axis (rfft, k2 < m)
              X = (ch + i sh) @ Y              (2m, n)  along rows k1 in [0..m) and [n-m..n)
    Inverse:  Z = (gc + i gs) @ O / n          (n, 2m)  complex ifft along rows
              u = Re(Z) @ ar - Im(Z) @ ai      (m, n)   irfft along the last axis; ai[0] = sin(0) = 0
                                                       drops Im of the DC column, as C2R irfft does.
    """
    if 2 * m > n or m > n // 2:
        raise ValueError(f"{m} modes per corner do not fit an {n}x{n} grid")
    t = torch.arange(n, dtype=torch.float64)
    k2 = torch.arange(m, dtype=torch.float64)
    k1 = torch.cat((torch.arange(m), torch.arange(n - m, n))).to(torch.float64)
    ang_w = 2 * math.pi * t[:, None] * k2[None, :] / n          # (n, m)
    ang_h = 2 * math.pi * k1[:, None] * t[None, :] / n          # (2m, n)
    weight = torch.full((m, 1), 2.0, dtype=torch.float64)
    weight[0] = 1.0                                             # DC counted once, others twice (Hermitian half)
    return {
        "cw": torch.cos(ang_w), "sw": -torch.sin(ang_w),
        "ch": torch.cos(ang_h), "sh": -torch.sin(ang_h),
        "gc": torch.cos(ang_h).T / n, "gs": torch.sin(ang_h).T / n,
        "ar": weight * torch.cos(ang_w).T / n, "ai": weight * torch.sin(ang_w).T / n,
    }


class SpectralConvDFT(nn.Module):
    """SpectralConv2d at a fixed grid, with real matmuls in place of rfft2 / irfft2."""

    def __init__(self, conv: SpectralConv2d, n: int):
        super().__init__()
        if conv.modes1 != conv.modes2:
            raise ValueError("FNODFT assumes square mode blocks (modes1 == modes2)")
        for k, v in dft_matrices(n, conv.modes1).items():
            self.register_buffer(k, v.float())
        # (i, o, 2m, m, 2): w1 for rows [:m], w2 for rows [-m:], laid out as (2m*m, i, o) for batched matmul.
        w = torch.cat((conv.w1.detach(), conv.w2.detach()), dim=2)
        self.register_buffer("wr", w[..., 0].permute(2, 3, 0, 1).flatten(0, 1).contiguous())
        self.register_buffer("wi", w[..., 1].permute(2, 3, 0, 1).flatten(0, 1).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, _ = x.shape
        yr, yi = x @ self.cw, x @ self.sw                       # (b, c, n, m)
        xr = self.ch @ yr - self.sh @ yi                        # (b, c, 2m, m)
        xi = self.ch @ yi + self.sh @ yr
        k1, k2 = xr.shape[-2:]
        # einsum("bixy,ioxy->boxy") as one batched matmul over the k1*k2 modes.
        xr = xr.flatten(2).permute(2, 0, 1)                     # (k1*k2, b, c)
        xi = xi.flatten(2).permute(2, 0, 1)
        orr = xr @ self.wr - xi @ self.wi                       # (k1*k2, b, o)
        oi = xr @ self.wi + xi @ self.wr
        orr = orr.permute(1, 2, 0).reshape(b, -1, k1, k2)
        oi = oi.permute(1, 2, 0).reshape(b, -1, k1, k2)
        zr = self.gc @ orr - self.gs @ oi                       # (b, o, n, m)
        zi = self.gc @ oi + self.gs @ orr
        return zr @ self.ar - zi @ self.ai                      # (b, o, n, n)


class FNODFT(nn.Module):
    """Raw a (1, 1, n, n) -> u (1, 1, n, n): src.infer.predict with an FFT-free FNO2d."""

    def __init__(self, model: FNO2d, stats: dict, n: int = GRID):
        super().__init__()
        self.lift, self.head = model.lift, model.head
        self.spectral = nn.ModuleList(SpectralConvDFT(b.spectral, n) for b in model.blocks)
        self.pointwise = nn.ModuleList(b.pointwise for b in model.blocks)
        self.act = nn.GELU()
        self.register_buffer("grid", FNO2d._grid(torch.Size((1, n, n)), torch.device("cpu")).clone())
        self.a_mean, self.a_std, self.u_std = (float(stats[k]) for k in ("a_mean", "a_std", "u_std"))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = (a[:, 0] - self.a_mean) / self.a_std                # (b, n, n)
        grid = self.grid.expand(a.shape[0], -1, -1, -1)
        x = self.lift(torch.cat((a.unsqueeze(-1), grid), dim=-1)).permute(0, 3, 1, 2)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = self.act(spectral(x) + pointwise(x))
        u = self.head(x.permute(0, 2, 3, 1)).squeeze(-1)
        return (u * self.u_std).unsqueeze(1)


def pick_checkpoint(trained: Path = DEFAULT_CHECKPOINT, fallback: Path = RANDOM_CHECKPOINT) -> Path:
    """The trained checkpoint if it exists, else the untrained fallback."""
    if Path(trained).exists():
        return Path(trained)
    if Path(fallback).exists():
        return Path(fallback)
    raise FileNotFoundError(f"neither {trained} nor {fallback} exists")


def build_dft(checkpoint: Path, n: int = GRID) -> tuple[FNODFT, "LoadedModel"]:  # noqa: F821
    """Load a checkpoint with app.model.load_model and wrap it as FNODFT (eval mode)."""
    loaded = load_model(checkpoint)
    return FNODFT(loaded.model, loaded.stats, n).eval(), loaded


def split_pair(index: int, n: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """Test-split a and its reference u, float32, identical to app.main.sample / darcy_test.hdf5."""
    a = split_sample(index, TEST_SIZE, TEST_SEED, n).astype(np.float32)
    u = solve_darcy(a.astype(np.float64), h=1.0 / n, face="arithmetic")
    return a, u.astype(np.float32)


def export_onnx(model: FNODFT, path: Path, n: int = GRID) -> Path:
    """Export with a static (1, 1, n, n) input named "a" and output named "u"."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (torch.zeros(1, 1, n, n),), str(path), input_names=["a"], output_names=["u"],
                      opset_version=OPSET, dynamo=True, external_data=False)
    return path


def write_samples(out: Path, count: int = 50, n: int = GRID) -> dict:
    """samples.bin: count x (a, u) float32 little-endian, row-major; samples.json describes the layout."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [split_pair(i, n) for i in range(count)]
    np.stack([np.stack(p) for p in pairs]).astype("<f4").tofile(out / "samples.bin")
    meta = {"count": count, "grid": n, "layout": "count x [a, u] x grid x grid, float32 little-endian",
            "split": "test", "indices": list(range(count)), "values": [BUMPS_LO, BUMPS_HI]}
    (out / "samples.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def write_golden(out: Path, loaded, index: int = 0, n: int = GRID) -> Path:
    """golden.json: one raw input a and the original FNO2d output (src.infer path) for the page self-test."""
    a, _ = split_pair(index, n)
    u = loaded.predict(a)
    path = Path(out) / "golden.json"
    path.write_text(json.dumps({"index": index, "grid": n, "a": a.ravel().tolist(),
                                "u": u.astype(np.float32).ravel().tolist()}) + "\n")
    return path


def write_solver_fixtures(path: Path, sizes: tuple[int, ...] = (16, 24, 32, 32, 128)) -> Path:
    """Python-solved cases (solve_darcy, arithmetic faces, h = 1/n) for web/solver.test.js."""
    rng = np.random.default_rng(0)
    cases = []
    for k, n in enumerate(sizes):
        if n == GRID:
            a = split_sample(k, TEST_SIZE, TEST_SEED, n)
        else:
            a = np.where(rng.uniform(size=(n, n)) > 0.5, BUMPS_HI, BUMPS_LO)
        u = solve_darcy(a, h=1.0 / n, face="arithmetic")
        cases.append({"n": n, "a": a.ravel().tolist(), "u": u.ravel().tolist()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases) + "\n")
    return path


def main(out: Path = WEB) -> None:
    ckpt = pick_checkpoint()
    model, loaded = build_dft(ckpt)
    export_onnx(model, out / "model.onnx")
    meta = {"checkpoint": ckpt.name, "trained": loaded.trained, "params": loaded.n_params, "grid": GRID,
            "opset": OPSET}
    (out / "model.json").write_text(json.dumps(meta, indent=1) + "\n")
    write_golden(out, loaded)
    write_samples(out)
    write_solver_fixtures(out / "fixtures" / "solver_cases.json")
    print(f"wrote {out}: {meta}")


if __name__ == "__main__":
    main()
