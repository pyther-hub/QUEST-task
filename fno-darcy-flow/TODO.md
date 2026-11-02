# TODO — FNO Darcy demo

Status (2026-11-02): the app serves the **trained** small FNO (run v1, Kaggle 2× T4) on CPU at
128×128 in the PDEBench setting. Deploy steps: `DEPLOY.md`.

## Done
- [x] `app/model.py`: `load_model(path)`, `.predict(a)`, `.trained`; preprocessing delegated to `src.infer`
- [x] `app/main.py`: `/health`, `/sample?index=N` (exact test-split fields), `/predict`, `/solve` (exact reference)
- [x] `app/static/index.html`: test sample / draw (two-phase brush, fill, undo) / function input
      (JS expression, evaluated in the browser), shared colour scale for u, error, timings, hover readout
- [x] Notebook 01 on Kaggle: train/val/test 9000/500/500 (seeds 0/1/2), 128×128, PDEBench setting
- [x] Notebook 02 run v1: 47 epochs (early stop, patience 3), test relL2 mean 2.52 %, median 2.00 %
      → `~/Desktop/QUEST-data/checkpoints/fno_darcy.pt` (+ `fno_darcy_v1.pt`)
- [x] `deploy/`: CPU Dockerfile, pinned requirements, `make_space.sh` (Docker context + HF Space)

## Next
- [ ] Run v2 (looser patience): if better, copy its best.pt to `fno_darcy.pt` (DEPLOY.md step 2)
- [ ] Deploy: DEPLOY.md step 4 (Docker) or 5 (Hugging Face Space)
- [ ] Evaluate on PDEBench's own data (v1, first 200 samples: median 2.14 %, mean 4.55 %;
      the tail is PDEBench samples that have not reached steady state)

## Adding your own models
- [ ] Decide the checkpoint contract for non-FNO2d models. Today `load_model()` →
      `src.infer.load_model`, which always builds `FNO2d(**config)`. Options:
      a `"arch"` key in the checkpoint plus a small registry in `app/model.py`, keeping
      `LoadedModel.predict(a) -> u` as the only interface the API uses.
- [ ] Keep the same normalisation stats keys (`a_mean`, `a_std`, `u_std`) or make preprocessing
      part of the model wrapper.
- [ ] Add a `GET /models` + model selector in the UI only once there are 2+ checkpoints.
- [ ] One test per new model: load, param count, output shape, determinism (copy `test_app_model.py`).

## Later (out of scope for now, per CLAUDE.md)
- [ ] Quantization / edge export — swap in through `load_model()` only.
- [ ] Deployment.

## Housekeeping
- [ ] Starlette warns that `httpx` with TestClient is deprecated in favour of `httpx2`.
      Needs approval before changing deps.
- [ ] `/sample` solves the PDE per request (~0.14 s measured); cache by seed if it gets slow.
- [ ] `data/darcy_train.npz` (1200 samples) and `checkpoints/` are gitignored; on a fresh clone
      `python -m app.model` regenerates both (~15 s measured).

## Later phase: bigger models
- [ ] Add a function to switch model size by name, e.g. `build_model(preset)` with presets
      `small` (modes 12, width 32, layers 4; 2,368,001 params) and `large` (modes 24, width 64,
      layers 4; 37,774,081 params), plus custom (modes, width, layers). Use it in notebook 02's
      config (`MODEL = "small"`) and record the preset in the checkpoint `config`, so
      `load_model()` and the app pick up any size unchanged.
- [ ] Re-run `MODE="precheck"` per preset: bigger models need a smaller BATCH_SIZE per GPU.
