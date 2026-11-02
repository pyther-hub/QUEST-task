"""FastAPI demo: sample Darcy problems and run the FNO surrogate on them.

    uvicorn app.main:app --reload       # http://127.0.0.1:8000

The checkpoint path comes from $FNO_CHECKPOINT (default checkpoints/fno_small_random.pt).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from app.model import DEFAULT_CHECKPOINT, GRID, load_model
from src.data.darcy import gaussian_random_field, permeability_from_grf, solve_darcy

STATIC = Path(__file__).parent / "static"


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


def sample(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One (a, u) pair from the upstream sampler + solver, as float32 like generate()."""
    a = permeability_from_grf(gaussian_random_field(GRID, rng=np.random.default_rng(seed)))
    return a.astype(np.float32), solve_darcy(a).astype(np.float32)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    m = app.state.model
    return {
        "model_loaded": m is not None,
        "params": m.n_params if m else None,
        "checkpoint": app.state.checkpoint,
    }


@app.get("/sample")
def get_sample(seed: int = Query(0, ge=0)) -> dict:
    a, u = sample(seed)
    return {"seed": seed, "a": a.tolist(), "u": u.tolist()}


@app.post("/predict")
def post_predict(req: PredictRequest) -> dict:
    if app.state.model is None:
        raise HTTPException(503, f"no checkpoint at {app.state.checkpoint}")
    return {"u": app.state.model.predict(req.a).tolist()}
