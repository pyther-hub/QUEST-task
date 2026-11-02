# Deploying the FNO Darcy demo (CPU)

The app is one FastAPI process that serves the UI, runs the trained FNO on CPU, and runs the exact
solver for comparison. No GPU is needed: one prediction takes about 30–40 ms on a laptop CPU at
128×128.

All commands run from `fno-darcy-flow/` unless stated. Large files (checkpoints, the deploy folder)
stay in `~/Desktop/QUEST-data/`, never in the git repo.

```
~/Desktop/QUEST-data/
  checkpoints/fno_darcy.pt      the model the app serves (copy of a run's best.pt)
  checkpoints/fno_darcy_v1.pt   run v1, kept for reference
  space/                        deploy folder made by deploy/make_space.sh (step 3)
```

---

## 1. Run it locally (no Docker)

```bash
cd ~/Desktop/QUEST-task/fno-darcy-flow
. .venv/bin/activate
uvicorn app.main:app            # open http://127.0.0.1:8000
```

The badge at the top should read **"Predictions: trained FNO · 2.37M params · CPU"**. If it reads
"no model loaded", `~/Desktop/QUEST-data/checkpoints/fno_darcy.pt` is missing.

The UI has three ways to set the permeability a(x):

- **Test sample**: one of the 500 held-out test fields the model never saw (`/sample?index=0..499`).
- **Draw**: paint a = 1 or a = 0.1 with the mouse or a finger. The prediction updates on release, and Undo / Ctrl+Z works.
- **Function**: type an expression in `x` and `y`, for example `hypot(x - 0.5, y - 0.5) < 0.3`. It is evaluated in the
  browser only; the server never runs user code.

## 2. Swap in a new checkpoint (for example run v2)

```bash
cp "<path to run v2>/best.pt" ~/Desktop/QUEST-data/checkpoints/fno_darcy_v2.pt
cp ~/Desktop/QUEST-data/checkpoints/fno_darcy_v2.pt ~/Desktop/QUEST-data/checkpoints/fno_darcy.pt
```

Restart uvicorn. To check it, open test sample **123** and compare the "relative L2 error" with
`per_sample_relL2.npz["test"][123]` from that run. They must match: the app's test samples are the exact
fields in `darcy_test.hdf5`. For run v1 it is 10.36 %, and sample 0 is 2.86 %.

To try a checkpoint without replacing the default:
`FNO_CHECKPOINT=/path/to/best.pt uvicorn app.main:app`.

## 3. Build the deploy folder

```bash
rm -rf ~/Desktop/QUEST-data/space                # only if an old one exists
bash deploy/make_space.sh                       # -> ~/Desktop/QUEST-data/space (about 9 MB)
```

This copies `app/`, `src/`, the Dockerfile, the pinned `requirements.txt`, a Space README and
`fno_darcy.pt` into one self-contained folder. That folder is both the Docker build context (step 4)
and what gets uploaded to Hugging Face (step 5). Re-run this step every time the code or the
checkpoint changes.

Other locations: `bash deploy/make_space.sh OUT_DIR [CHECKPOINT]`.

## 4. Option A: Docker (your Mac, or any Linux server / VM)

One-time setup: install Docker Desktop from https://www.docker.com/products/docker-desktop/ (or
`brew install --cask docker`), start it, and check with `docker version`.

```bash
docker build -t fno-darcy ~/Desktop/QUEST-data/space
docker run --rm -p 7860:7860 fno-darcy           # open http://127.0.0.1:7860
```

The first build downloads the CPU torch wheel (about 200 MB), so it takes a few minutes. Later builds
reuse the cached layers. Stop the container with Ctrl+C.

To run it on a server: the image is built for your Mac's CPU (arm64). Most cloud VMs are x86, so either
build on the server itself (copy the folder over with `scp -r ~/Desktop/QUEST-data/space server:`) or
build for x86 on the Mac:

```bash
docker build --platform linux/amd64 -t fno-darcy ~/Desktop/QUEST-data/space
```

On the server: `docker run -d --restart unless-stopped -p 80:7860 fno-darcy`. The app is then at
`http://<server-ip>/`. Put HTTPS in front of it (Caddy, nginx, or the cloud's load balancer) before
sharing the link widely.

## 5. Option B: Hugging Face Spaces (free CPU hosting, public URL)

The Space builds the same Dockerfile on Hugging Face's servers, so Docker isn't needed on the Mac.

One-time setup:

1. Create an account at https://huggingface.co and a **write** token at
   https://huggingface.co/settings/tokens.
2. Install the Hugging Face CLI in its own environment (it isn't a dependency of the project):

   ```bash
   python3 -m venv ~/.hf-cli
   ~/.hf-cli/bin/pip install -U huggingface_hub
   ~/.hf-cli/bin/hf auth login                     # paste the write token
   ```

Create the Space and upload the folder. Replace `USERNAME` with your Hugging Face username:

```bash
~/.hf-cli/bin/hf repos create USERNAME/fno-darcy --type space --space-sdk docker
~/.hf-cli/bin/hf upload USERNAME/fno-darcy ~/Desktop/QUEST-data/space . --type space
```

- Build logs: https://huggingface.co/spaces/USERNAME/fno-darcy (click "Logs"). The first build takes a few minutes.
- App URL: https://huggingface.co/spaces/USERNAME/fno-darcy, or directly `https://USERNAME-fno-darcy.hf.space`.
- The Space is public by default, including `model/fno_darcy.pt`. Add `--private` to `repos create` if
  only you should see it.
- Free CPU Spaces go to sleep after a period without visitors. The next visit wakes the Space, and the first page load is slow while it does.

To update after a code change or a new checkpoint: repeat step 3, then run the same `hf upload` command again.

## 6. Check a deployment

```bash
curl -s https://USERNAME-fno-darcy.hf.space/health     # or http://127.0.0.1:7860/health
```

It should print `"model_loaded":true,"trained":true,"params":2368001,"grid":128`. Then, in the UI:

- [ ] test sample 123 shows the same error as the run's `per_sample_relL2.npz` (v1: 10.36 %)
- [ ] drawing a stroke updates the prediction on release
- [ ] Function → "wavy channel" applies, and a typo such as `x +* y` shows a syntax error
- [ ] timings are shown (FNO and solver are both tens of ms on CPU)

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Badge says "no model loaded" | No checkpoint at `$FNO_CHECKPOINT`. Locally: step 2. In Docker or a Space: re-run step 3 with the checkpoint present. |
| `make_space.sh: ... is not empty` | It refuses to overwrite. Delete the old `~/Desktop/QUEST-data/space` first. |
| Space stuck on "Building" or showing an error | Open "Logs". A wrong `app_port` or a missing file shows up there. The README in the folder must keep `sdk: docker` and `app_port: 7860`. |
| `exec format error` on a server | The image was built for arm64. Rebuild with `--platform linux/amd64` (step 4). |
| `/predict` returns 422 | The input must be 128×128 finite numbers. `/solve` also needs a > 0 everywhere. |
