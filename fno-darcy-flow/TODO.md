# TODO — FNO Darcy demo

Status (2026-11-02): demo works end to end with the **untrained** small FNO
(modes 12, width 32, layers 4; 2,368,001 params). Branch `demo`.

## Done
- [x] venv + deps; upstream tests: 22 passed, 1 skipped (FSDP needs CUDA)
- [x] `checkpoints/fno_small_random.pt` via `python -m app.model` (seed 0, src.train format)
- [x] `app/model.py` — `load_model(path)`, `.predict(a)`; preprocessing delegated to `src.infer`
- [x] `app/main.py` — `/health`, `/sample?seed=N`, `/predict` (422 on bad shape / NaN / Inf)
- [x] `app/static/index.html` — a, reference u, predicted u, |error| heatmaps
- [x] `tests/test_app_model.py`, `tests/test_app_api.py`

## Kaggle training (2× T4, DDP)
- [x] `notebooks/01_generate_dataset.ipynb` — train/val/test (4000/500/500, seeds 0/1/2), parallel solve
- [x] `notebooks/02_train_ddp.ipynb` — config on top; `MODE="precheck"` (batch-size sweep: samples/s,
      GPU util %, peak mem, recommended batch) and `MODE="train"` (logs, metrics.csv, curves,
      eval of best.pt on train/val/test). Tested on CPU only (`tests/test_notebooks.py`).
- [ ] Run 01 on Kaggle, then 02 precheck → set BATCH_SIZE → 02 train; record the numbers here
- [ ] Download `best.pt` → `checkpoints/fno_small.pt`, serve with `FNO_CHECKPOINT=...`

## Next: UI input
- [ ] Let the user supply a(x) three ways, all feeding `POST /predict`:
      upload a 64×64 field (.npy / .csv / grayscale .png), generate a random sample,
      or hand-draw on the canvas (two-phase brush: a=3 / a=12, plus clear / fill).

## Next: train the small model
- [ ] Train on CPU (or GPU) with the data already generated:
      `python -m src.train --data data/darcy_train.npz --modes 12 --width 32 --layers 4 --out checkpoints/fno_small.pt`
      (try `--epochs 5` first to time an epoch on CPU)
- [ ] Serve it: `FNO_CHECKPOINT=checkpoints/fno_small.pt uvicorn app.main:app --reload`
- [ ] Record val relL2 from the training log; compare with `python -m src.infer --checkpoint ... --data ...`
- [ ] Make the UI label depend on the checkpoint instead of always saying
      "untrained (random weights)": e.g. add an optional `"label"` key to the checkpoint
      (`src.infer.load_model` ignores extra keys), return it from `/health`, show it in the UI.
      Update `test_index` accordingly.
- [ ] Evaluate on held-out data generated with a different seed (`src.data.darcy --seed 1`),
      not only the train split.

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
