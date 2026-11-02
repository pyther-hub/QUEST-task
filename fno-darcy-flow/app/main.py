"""FastAPI demo: sample or draw Darcy problems and run the FNO surrogate on them.

    uvicorn app.main:app --reload       # http://127.0.0.1:8000

The checkpoint path comes from $FNO_CHECKPOINT (default $FNO_DATA_DIR/checkpoints/fno_darcy.pt,
$FNO_DATA_DIR defaults to ~/Desktop/QUEST-data).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from app.model import DEFAULT_CHECKPOINT, GRID, load_model
from src.data.darcy import split_sample, solve_darcy

STATIC = Path(__file__).parent / "static"
# Held-out test split of notebooks/01_generate_dataset.ipynb (SPLITS["test"] = (500, seed 2)).
TEST_SIZE, TEST_SEED = 500, 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("FNO_CHECKPOINT", DEFAULT_CHECKPOINT))
    app.state.model = load_model(path) if path.exists() else None
    app.state.checkpoint = str(path)
    yield


app = FastAPI(title="FNO Darcy demo", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # The default handler echoes the input back, which cannot serialise NaN/Inf.
    errors = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class PredictRequest(BaseModel):
    a: list[list[float]]

    @field_validator("a")
    @classmethod
    def check_grid(cls, a: list[list[float]]) -> list[list[float]]:
        if len(a) != GRID or any(len(row) != GRID for row in a):
            raise ValueError(f"a must be {GRID}x{GRID}")
        if not np.isfinite(np.asarray(a)).all():
            raise ValueError("a must contain only finite numbers")
        return a


class SolveRequest(PredictRequest):
    @field_validator("a")
    @classmethod
    def check_positive(cls, a: list[list[float]]) -> list[list[float]]:
        if not (np.asarray(a) > 0).all():
            raise ValueError("a must be positive everywhere")
        return a


def reference(a: np.ndarray) -> np.ndarray:
    """Exact steady solve with the discretisation the model was trained on (PDEBench's stencil)."""
    return solve_darcy(np.asarray(a, dtype=np.float64), h=1.0 / GRID, face="arithmetic")


@lru_cache(maxsize=64)
def sample(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Test-split sample `index` and its reference u, as float32 like the generated files."""
    a = split_sample(index, TEST_SIZE, TEST_SEED, GRID).astype(np.float32)
    return a, reference(a).astype(np.float32)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    m = app.state.model
    return {
        "model_loaded": m is not None,
        "trained": m.trained if m else None,
        "params": m.n_params if m else None,
        "grid": GRID,
        "checkpoint": app.state.checkpoint,
    }


@app.get("/sample")
def get_sample(index: int = Query(0, ge=0, lt=TEST_SIZE)) -> dict:
    a, u = sample(index)
    return {"index": index, "a": a.tolist(), "u": u.tolist()}


@app.post("/predict")
def post_predict(req: PredictRequest) -> dict:
    if app.state.model is None:
        raise HTTPException(503, f"no checkpoint at {app.state.checkpoint}")
    t = time.perf_counter()
    u = app.state.model.predict(req.a)
    return {"u": u.tolist(), "ms": (time.perf_counter() - t) * 1e3}


@app.post("/solve")
def post_solve(req: SolveRequest) -> dict:
    t = time.perf_counter()
    u = reference(req.a)
    return {"u": u.tolist(), "ms": (time.perf_counter() - t) * 1e3}
