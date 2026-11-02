"""Darcy flow dataset generation.

Solves the 2D Darcy flow equation on the unit square with a
finite-difference discretisation:

    -div( a(x) * grad(u(x)) ) = f(x),   x in (0,1)^2
    u(x) = 0 on the boundary

where a(x) is a random permeability field sampled from a Gaussian
random field pushed through a threshold (piecewise-constant media),
and f(x) = 1. The learning task is the operator a(x) -> u(x),
the standard benchmark from Li et al., "Fourier Neural Operator for
Parametric Partial Differential Equations" (ICLR 2021).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Generated data and checkpoints live outside the repo so they are never pushed.
DATA_ROOT = Path(os.environ.get("FNO_DATA_DIR", Path.home() / "Desktop" / "QUEST-data"))


def gaussian_random_field(n: int, alpha: float = 2.0, tau: float = 3.0,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a mean-zero Gaussian random field on an n x n grid.

    Spectral density ~ tau^(alpha-1) * (pi^2 (k1^2 + k2^2) + tau^2)^(-alpha/2),
    i.e. draws from N(0, (-Laplacian + tau^2 I)^(-alpha)).
    """
    rng = rng or np.random.default_rng()
    k = np.arange(n)
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    coef = (np.pi ** 2) * (k1 ** 2 + k2 ** 2) + tau ** 2
    sqrt_eig = (tau ** (alpha - 1.0)) * coef ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0  # remove the mean mode
    xi = rng.standard_normal((n, n))
    # Inverse discrete cosine-like transform via FFT of symmetrised field
    from scipy.fft import idctn
    return idctn(sqrt_eig * xi, norm="ortho")


def permeability_from_grf(grf: np.ndarray, hi: float = 12.0, lo: float = 3.0) -> np.ndarray:
    """Threshold a GRF into a two-phase permeability field (as in the FNO paper)."""
    return np.where(grf >= 0.0, hi, lo)


# PDEBench-setting permeability (same recipe as notebooks/01_generate_dataset.ipynb):
# 5 Gaussian bumps on cell centres, thresholded at the mean of a batch of 200 to {0.1, 1}.
BUMPS_LO, BUMPS_HI, N_BUMPS, BUMPS_BATCH = 0.1, 1.0, 5, 200


def bumps_permeability(n_samples: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n_samples, n, n) two-phase field in {0.1, 1}; axis 1 is x, axis 2 is y."""
    xc = (np.arange(n) + 0.5) / n
    m = rng.uniform(0.0, 1.0, (n_samples, N_BUMPS, 2))
    s = np.maximum(0.5 * rng.uniform(0.0, 1.0, (n_samples, N_BUMPS)), 1e-6)
    field = np.zeros((n_samples, n, n))
    for k in range(N_BUMPS):
        dx2 = (xc[None, :] - m[:, k, 0:1]) ** 2
        dy2 = (xc[None, :] - m[:, k, 1:2]) ** 2
        field += np.exp(-(dx2[:, :, None] + dy2[:, None, :]) / s[:, k, None, None])
    return np.where(field > field.mean(), BUMPS_HI, BUMPS_LO)


def split_sample(index: int, n_split: int, seed: int, n: int, chunk: int = BUMPS_BATCH) -> np.ndarray:
    """Sample `index` of a notebook-01 split of n_split samples (chunk c drawn with default_rng([seed, c]))."""
    if not 0 <= index < n_split:
        raise ValueError(f"index must be in [0, {n_split})")
    c, i = divmod(index, chunk)
    size = min(chunk, n_split - c * chunk)
    return bumps_permeability(size, n, np.random.default_rng([seed, c]))[i]


def solve_darcy(a: np.ndarray, f: float = 1.0, h: float | None = None,
                face: str = "harmonic") -> np.ndarray:
    """Solve -div(a grad u) = f with homogeneous Dirichlet BCs.

    Uses a 5-point finite-volume stencil with harmonic averaging of the
    permeability at cell faces, which is the standard discretisation for
    discontinuous coefficients.

    h defaults to 1/(n+1) (n interior nodes of the unit square). PDEBench's
    discretisation is h=1/n (cell centres, u=0 one cell outside the edge)
    with face="arithmetic".
    """
    n = a.shape[0]
    h = 1.0 / (n + 1) if h is None else h

    if face == "harmonic":
        def harmonic(x, y):
            return 2.0 * x * y / (x + y)
    elif face == "arithmetic":
        def harmonic(x, y):
            return 0.5 * (x + y)
    else:
        raise ValueError(f"unknown face average {face!r}")

    # Face permeabilities (interior cells only)
    a_e = np.zeros((n, n)); a_w = np.zeros((n, n))
    a_n = np.zeros((n, n)); a_s = np.zeros((n, n))
    a_e[:, :-1] = harmonic(a[:, :-1], a[:, 1:]); a_e[:, -1] = a[:, -1]
    a_w[:, 1:] = harmonic(a[:, 1:], a[:, :-1]); a_w[:, 0] = a[:, 0]
    a_n[:-1, :] = harmonic(a[:-1, :], a[1:, :]); a_n[-1, :] = a[-1, :]
    a_s[1:, :] = harmonic(a[1:, :], a[:-1, :]); a_s[0, :] = a[0, :]

    diag = (a_e + a_w + a_n + a_s).ravel()
    idx = np.arange(n * n).reshape(n, n)

    rows, cols, vals = [np.array([], dtype=np.int64)] * 0, [], []
    rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [diag]

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(-v.ravel())

    add(idx[:, :-1], idx[:, 1:], a_e[:, :-1])   # east neighbour
    add(idx[:, 1:], idx[:, :-1], a_w[:, 1:])    # west neighbour
    add(idx[:-1, :], idx[1:, :], a_n[:-1, :])   # north neighbour
    add(idx[1:, :], idx[:-1, :], a_s[1:, :])    # south neighbour

    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    )
    b = np.full(n * n, f * h * h)
    u = spsolve(A, b)
    return u.reshape(n, n)


def generate(n_samples: int, grid: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    a_all = np.empty((n_samples, grid, grid), dtype=np.float32)
    u_all = np.empty((n_samples, grid, grid), dtype=np.float32)
    for i in range(n_samples):
        a = permeability_from_grf(gaussian_random_field(grid, rng=rng))
        a_all[i] = a
        u_all[i] = solve_darcy(a)
        if (i + 1) % 50 == 0:
            print(f"  solved {i + 1}/{n_samples}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, a=a_all, u=u_all)
    print(f"wrote {out} ({n_samples} samples at {grid}x{grid})")


# ---------------------------------------------------------------------------
# PDEBench comparison
#
# PDEBench (Takamoto et al., NeurIPS 2022) ships a Darcy dataset for the same
# PDE with f = beta. beta = 1.0 matches this module's f = 1. URL and MD5 come
# from PDEBench's pdebench/data_download/pdebench_data_urls.csv.
# ---------------------------------------------------------------------------

PDEBENCH_FILE = "2D_DarcyFlow_beta1.0_Train.hdf5"
PDEBENCH_URL = "https://darus.uni-stuttgart.de/api/access/datafile/133219"
PDEBENCH_MD5 = "81694ed31306ff2e5f6b76349b0b4389"
PDEBENCH_BETA = 1.0


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_pdebench(out_dir: Path, url: str = PDEBENCH_URL, md5: str = PDEBENCH_MD5,
                      name: str = PDEBENCH_FILE) -> Path:
    """Download one PDEBench file into out_dir (skipped if already present and valid)."""
    out = Path(out_dir) / name
    if out.exists() and md5sum(out) == md5:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    print(f"downloading {url} -> {out}")
    with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    got = md5sum(part)
    if got != md5:
        part.unlink()
        raise ValueError(f"MD5 mismatch for {name}: expected {md5}, got {got}")
    part.replace(out)
    return out


def load_pdebench(path: Path, n_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read (a, u) from a PDEBench Darcy HDF5: 'nu' is a(x), 'tensor' is u(x)."""
    import h5py
    with h5py.File(path, "r") as f:
        sl = slice(None) if n_samples is None else slice(0, n_samples)
        a = np.asarray(f["nu"][sl], dtype=np.float64)
        u = np.asarray(f["tensor"][sl, 0], dtype=np.float64)
    return a, u


def block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the last two axes by averaging factor x factor blocks."""
    *lead, h, w = x.shape
    return x.reshape(*lead, h // factor, factor, w // factor, factor).mean(axis=(-3, -1))


def field_stats(a: np.ndarray, u: np.ndarray) -> dict:
    """Summary statistics for a batch of (a, u) pairs, shape (N, n, n)."""
    mid = 0.5 * (a.min() + a.max())
    return {
        "n": int(a.shape[0]), "grid": int(a.shape[-1]),
        "a_min": float(a.min()), "a_max": float(a.max()),
        "a_unique": int(np.unique(a).size),
        "a_high_fraction": float((a > mid).mean()),
        "u_min": float(u.min()), "u_max": float(u.max()),
        "u_mean": float(u.mean()), "u_std": float(u.std()),
        "u_peak_mean": float(u.max(axis=(-2, -1)).mean()),
    }


def solver_agreement(a: np.ndarray, u_ref: np.ndarray, f: float = PDEBENCH_BETA) -> np.ndarray:
    """Relative L2 error of solve_darcy(a, f) against u_ref, one value per sample."""
    errs = []
    for ai, ui in zip(a, u_ref):
        u = solve_darcy(ai, f=f)
        errs.append(np.linalg.norm(u - ui) / np.linalg.norm(ui))
    return np.array(errs)


def plot_comparison(ours: tuple[np.ndarray, np.ndarray], theirs: tuple[np.ndarray, np.ndarray],
                    out: Path, k: int = 3) -> Path:
    """Save a PNG: k example (a, u) pairs from this generator and from PDEBench."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = min(k, len(ours[0]), len(theirs[0]))
    fig, axes = plt.subplots(4, k, figsize=(3 * k, 11), squeeze=False)
    rows = [("generated a", ours[0]), ("generated u", ours[1]),
            ("PDEBench a", theirs[0]), ("PDEBench u", theirs[1])]
    for r, (label, data) in enumerate(rows):
        for c in range(k):
            ax = axes[r, c]
            im = ax.imshow(data[c], origin="lower", cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[r, 0].set_ylabel(label)
    fig.tight_layout()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100); plt.close(fig)
    return out


def compare_pdebench(pdebench_path: Path, n_samples: int = 32, grid: int = 64, seed: int = 0,
                     plot: Path | None = None) -> dict:
    """Compare this generator with PDEBench beta=1.0 Darcy data.

    - solver: our solve_darcy run on PDEBench a(x) at PDEBench resolution vs PDEBench u(x)
    - stats: generated data at `grid` vs PDEBench block-averaged down to `grid`
    """
    a_pb, u_pb = load_pdebench(pdebench_path, n_samples)
    errs = solver_agreement(a_pb, u_pb)
    factor = a_pb.shape[-1] // grid
    a_pb_g, u_pb_g = block_mean(a_pb, factor), block_mean(u_pb, factor)

    rng = np.random.default_rng(seed)
    a_ours = np.stack([permeability_from_grf(gaussian_random_field(grid, rng=rng))
                       for _ in range(n_samples)])
    u_ours = np.stack([solve_darcy(a) for a in a_ours])

    result = {
        "solver_rel_l2_mean": float(errs.mean()),
        "solver_rel_l2_max": float(errs.max()),
        "pdebench_native": field_stats(a_pb, u_pb),
        "pdebench": field_stats(a_pb_g, u_pb_g),
        "generated": field_stats(a_ours, u_ours),
    }
    if plot is not None:
        plot_comparison((a_ours, u_ours), (a_pb_g, u_pb_g), plot)
    return result


def print_comparison(r: dict) -> None:
    print(f"solver agreement on PDEBench a(x): rel L2 mean {r['solver_rel_l2_mean']:.4f}, "
          f"max {r['solver_rel_l2_max']:.4f}")
    cols = ["pdebench_native", "pdebench", "generated"]
    print(f"{'stat':<16}" + "".join(f"{c:>18}" for c in cols))
    for key in r["generated"]:
        print(f"{key:<16}" + "".join(f"{r[c][key]:>18.4g}" for c in cols))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA_ROOT / "data" / "darcy_train.npz")
    p.add_argument("--compare-pdebench", type=Path, metavar="DIR", nargs="?", default=None,
                   const=DATA_ROOT / "data" / "pdebench",
                   help="download PDEBench beta=1.0 Darcy into DIR and compare instead of generating")
    p.add_argument("--compare-samples", type=int, default=32)
    p.add_argument("--plot", type=Path, default=DATA_ROOT / "data" / "pdebench" / "comparison.png")
    args = p.parse_args()
    if args.compare_pdebench is not None:
        path = download_pdebench(args.compare_pdebench)
        print_comparison(compare_pdebench(path, args.compare_samples, args.grid, args.seed, args.plot))
        print(f"wrote {args.plot}")
    else:
        generate(args.n_samples, args.grid, args.seed, args.out)
