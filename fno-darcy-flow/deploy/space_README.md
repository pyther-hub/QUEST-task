---
title: FNO Darcy Flow
emoji: 🌊
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fourier Neural Operator surrogate for 2D Darcy flow
---

# FNO Darcy-flow surrogate

Draw, write or sample a permeability field a(x) and compare a trained Fourier Neural Operator's
pressure prediction u(x) against an exact finite-volume solve. Runs on CPU.

Built from `deploy/` in the project repository with `deploy/make_space.sh`.
