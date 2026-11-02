"""Model wrapper for the demo app.

All model loading goes through load_model(), so a trained (or later quantized)
checkpoint can be swapped in without touching the API or UI. Preprocessing is
delegated to src.infer, so it always matches the upstream inference path.

    python -m app.model     # writes checkpoints/fno_small_random.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.benchmark import PRESETS
from src.data.darcy import generate
from src.infer import load_model as _infer_load_model
from src.infer import predict as _infer_predict
from src.models.fno import FNO2d
from src.train import load_data

GRID = 64
# Smallest upstream preset; benchmark.py does not pass --layers, so train.py's default (4) applies.
SMALL_CONFIG = {**PRESETS["small"], "layers": 4}
DEFAULT_CHECKPOINT = Path("checkpoints/fno_small_random.pt")
DEFAULT_DATA = Path("data/darcy_train.npz")
DEVICE = torch.device("cpu")


@dataclass
class LoadedModel:
    model: FNO2d
    stats: dict
    path: Path

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, a) -> np.ndarray:
        """a: (H, W) permeability in physical units -> u: (H, W) in physical units."""
        a_t = torch.as_tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)
        return _infer_predict(self.model, a_t, self.stats, DEVICE)[0].numpy()


def load_model(path: Path = DEFAULT_CHECKPOINT) -> LoadedModel:
    """Load a checkpoint in the format src.train saves / src.infer expects."""
    model, stats = _infer_load_model(Path(path), DEVICE)
    return LoadedModel(model=model, stats=stats, path=Path(path))


def build_random_checkpoint(path: Path = DEFAULT_CHECKPOINT, data_path: Path = DEFAULT_DATA,
                            n_samples: int = 1200, seed: int = 0) -> Path:
    """Save an untrained small FNO (torch.manual_seed(seed)) in src.train's checkpoint format.

    Normalisation stats come from src.train.load_data over a dataset produced by
    src.data.darcy.generate (its CLI defaults: 1200 samples, 64x64, seed 0), i.e.
    exactly what a real training run on the default data would store.
    """
    path, data_path = Path(path), Path(data_path)
    if not data_path.exists():
        generate(n_samples, GRID, 0, data_path)
    _, _, stats = load_data(data_path)
    torch.manual_seed(seed)
    model = FNO2d(**SMALL_CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": SMALL_CONFIG, "stats": stats}, path)
    return path


if __name__ == "__main__":
    out = build_random_checkpoint()
    m = load_model(out)
    print(f"wrote {out}  config={SMALL_CONFIG}  params={m.n_params:,}  stats={m.stats}")
