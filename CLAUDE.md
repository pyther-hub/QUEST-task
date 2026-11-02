# FNO Darcy demo (practice run)

Repo: `./fno-darcy-flow`, cloned from https://github.com/AbdullahRasheed45/fno-darcy-flow. All paths below are relative to it.

## Scope now
- Smallest FNO preset from the repo, random weights (`torch.manual_seed(0)`). No training.
- CPU only. 64x64 grid.
- Any file in this repo may be edited, including `src/` and upstream tests. Keep existing tests green.

## Stack
- Python venv, PyTorch from `requirements.txt`
- h5py (read PDEBench `.hdf5` files)
- FastAPI + uvicorn, pytest (+ httpx for TestClient)
- Frontend: one `app/static/index.html`, vanilla JS, canvas heatmaps
- Nothing else: no frontend framework, no build step, no database, no Docker

## Layout
```
app/model.py          load_model(path), predict(a) -> u
app/main.py           FastAPI app, serves static/index.html
app/static/index.html
~/Desktop/QUEST-data/  outside the repo, override with $FNO_DATA_DIR
  checkpoints/         fno_small_random.pt
  data/                darcy_train.npz, pdebench/*.hdf5
tests/test_app_*.py   one test file per piece
```

## API
- `GET /health` -> model loaded, param count
- `GET /sample?seed=N` -> a(x) and reference u(x) from `src/data/darcy.py` sampler + solver
- `POST /predict` -> body `{"a": 64x64 floats}`, returns `{"u": 64x64 floats}`; bad shape or NaN -> 422

## Rules
- Read the repo code before using it. Take model config, preprocessing and checkpoint format from `src/`, never guess.
- Every new function gets a test. Run `python -m pytest -q` after each change; all green before saying done.
- Report only numbers from commands you ran.
- Ask before adding any dependency beyond fastapi, uvicorn, httpx, h5py.
- Keep it small. Commit after each green step.
- UI must label predictions "untrained (random weights)".

## Commands
```
cd fno-darcy-flow && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt fastapi uvicorn httpx
python -m src.data.darcy --compare-pdebench   # downloads ~1.3 GB once into $FNO_DATA_DIR/data/pdebench
python -m pytest -q
uvicorn app.main:app --reload    # http://127.0.0.1:8000
```

## PDEBench comparison
- Only `2D_DarcyFlow_beta1.0_Train.hdf5` (beta = 1.0 matches our f = 1). URL + MD5 from PDEBench's `pdebench_data_urls.csv`.
- Lives in `src/data/darcy.py`; downloads into `$FNO_DATA_DIR/data/pdebench/` (default `~/Desktop/QUEST-data`), never inside the repo. Tests must not download; use a tiny fake HDF5.
- Compare: solver agreement (our `solve_darcy` on PDEBench a vs PDEBench u), data stats, side-by-side PNG.
- Notebook 01 generates training data in the PDEBench setting: 128x128 cell centres, a = 5 Gaussian bumps -> {1, 0.1}, beta = 1, exact steady solve with `solve_darcy(h=1/n, face="arithmetic")` (PDEBench's stencil). Output `darcy_{train,val,test}.hdf5` in PDEBench layout (`nu`, `tensor` (N,1,n,n)).
- Notebook 02 reads that layout via `DATA_SOURCE = "generated" | "pdebench"` (PDEBench file split by index).
- The app (`/sample`, 64x64, a in {3, 12}) is still the original setting.

## Not now
Training, quantization / edge export, deployment. Keep all model loading inside `load_model()` so a trained or quantized model can be swapped in later.