# FNO Darcy demo (practice run)

Repo: `./fno-darcy-flow`, cloned from https://github.com/AbdullahRasheed45/fno-darcy-flow. All paths below are relative to it.

## Scope now
- Smallest FNO preset from the repo, random weights (`torch.manual_seed(0)`). No training.
- CPU only. 64x64 grid.
- Use upstream code in `src/` as is. Do not edit `src/` or upstream tests.

## Stack
- Python venv, PyTorch from `requirements.txt`
- FastAPI + uvicorn, pytest (+ httpx for TestClient)
- Frontend: one `app/static/index.html`, vanilla JS, canvas heatmaps
- Nothing else: no frontend framework, no build step, no database, no Docker

## Layout
```
app/model.py          load_model(path), predict(a) -> u
app/main.py           FastAPI app, serves static/index.html
app/static/index.html
checkpoints/          fno_small_random.pt (gitignored)
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
- Ask before adding any dependency beyond fastapi, uvicorn, httpx.
- Keep it small. Commit after each green step.
- UI must label predictions "untrained (random weights)".

## Commands
```
cd fno-darcy-flow && python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt fastapi uvicorn httpx
python -m pytest -q
uvicorn app.main:app --reload    # http://127.0.0.1:8000
```

## Not now
Training, quantization / edge export, deployment. Keep all model loading inside `load_model()` so a trained or quantized model can be swapped in later.