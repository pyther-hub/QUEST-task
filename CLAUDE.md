# FNO Darcy demo (practice run)

Repo: `./fno-darcy-flow`, cloned from https://github.com/AbdullahRasheed45/fno-darcy-flow. All paths below are relative to it.

## Scope now
- Smallest FNO preset (2,368,001 params), trained on Kaggle with notebook 02 (run v1: test relL2 2.52%).
  The app serves `$FNO_DATA_DIR/checkpoints/fno_darcy.pt`; `fno_small_random.pt` is the untrained fallback.
- CPU only for serving. 128x128 grid, PDEBench setting (a in {0.1, 1}), the grid the model was trained on.
- Deploy = CPU Docker image / Hugging Face Docker Space, steps in `fno-darcy-flow/DEPLOY.md`.
- Any file in this repo may be edited, including `src/` and upstream tests. Keep existing tests green.

## Stack
- Python venv, PyTorch from `requirements.txt`
- h5py (read PDEBench `.hdf5` files)
- FastAPI + uvicorn, pytest (+ httpx for TestClient)
- Frontend: one `app/static/index.html`, vanilla JS, canvas heatmaps
- Deploy: `deploy/Dockerfile` (python:3.13-slim + CPU torch), assembled by `deploy/make_space.sh`
- Nothing else: no frontend framework, no build step, no database

## Layout
```
app/model.py          load_model(path), predict(a) -> u
app/main.py           FastAPI app, serves static/index.html
app/static/index.html sample / draw / function input, heatmaps, error + timings
deploy/               Dockerfile, requirements.txt (pinned), make_space.sh, space_README.md
~/Desktop/QUEST-data/  outside the repo, override with $FNO_DATA_DIR
  checkpoints/         fno_darcy.pt (served), fno_darcy_v1.pt, fno_small_random.pt
  space/               deploy folder from make_space.sh (Docker context + HF Space)
  data/                darcy_train.npz, pdebench/*.hdf5
tests/test_app_*.py   one test file per piece
```

## API
- `GET /health` -> model loaded, trained flag, param count, grid
- `GET /sample?index=N` (0..499) -> test-split sample N of notebook 01 (`split_sample`) and its reference u
- `POST /predict` -> body `{"a": 128x128 floats}`, returns `{"u": 128x128 floats, "ms"}`; bad shape or NaN -> 422
- `POST /solve` -> same body, exact solve (`solve_darcy(h=1/n, face="arithmetic")`); also 422 if any a <= 0

## Rules
- Read the repo code before using it. Take model config, preprocessing and checkpoint format from `src/`, never guess.
- Every new function gets a test. Run `python -m pytest -q` after each change; all green before saying done.
- Report only numbers from commands you ran.
- Ask before adding any dependency beyond fastapi, uvicorn, httpx, h5py.
- Keep it small. Commit after each green step.
- UI must label predictions "untrained (random weights)" whenever the loaded checkpoint is untrained
  (`"trained": False` in the checkpoint; training checkpoints have no such key).
- User-written a(x) functions are evaluated in the browser only, never on the server.

## Commands
```
cd fno-darcy-flow && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt fastapi uvicorn httpx
python -m src.data.darcy --compare-pdebench   # downloads ~1.3 GB once into $FNO_DATA_DIR/data/pdebench
python -m pytest -q
uvicorn app.main:app --reload    # http://127.0.0.1:8000
bash deploy/make_space.sh        # deploy folder -> $FNO_DATA_DIR/space, then DEPLOY.md
```

## PDEBench comparison
- Only `2D_DarcyFlow_beta1.0_Train.hdf5` (beta = 1.0 matches our f = 1). URL + MD5 from PDEBench's `pdebench_data_urls.csv`.
- Lives in `src/data/darcy.py`; downloads into `$FNO_DATA_DIR/data/pdebench/` (default `~/Desktop/QUEST-data`), never inside the repo. Tests must not download; use a tiny fake HDF5.
- Compare: solver agreement (our `solve_darcy` on PDEBench a vs PDEBench u), data stats, side-by-side PNG.
- Notebook 01 generates training data in the PDEBench setting: 128x128 cell centres, a = 5 Gaussian bumps -> {1, 0.1}, beta = 1, exact steady solve with `solve_darcy(h=1/n, face="arithmetic")` (PDEBench's stencil). Output `darcy_{train,val,test}.hdf5` in PDEBench layout (`nu`, `tensor` (N,1,n,n)).
- Notebook 02 reads that layout via `DATA_SOURCE = "generated" | "pdebench"` (PDEBench file split by index).
- The app uses the same setting: `/sample` returns the exact test-split fields (bit-identical to `darcy_test.hdf5`).

## Not now
Quantization / edge export, GPU serving. Keep all model loading inside `load_model()` so a bigger or quantized model can be swapped in later.