# CPU-only image for the FNO Darcy demo: FastAPI + static UI + the trained checkpoint.
# Build it from the folder that deploy/make_space.sh assembles (see DEPLOY.md):
#   docker build -t fno-darcy ~/Desktop/QUEST-data/space && docker run -p 7860:7860 fno-darcy
# The same folder is what gets uploaded to a Hugging Face Docker Space.
FROM python:3.13-slim

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FNO_CHECKPOINT=/home/user/srv/model/fno_darcy.pt \
    PORT=7860
WORKDIR /home/user/srv

# CPU wheel of torch (about 200 MB instead of the multi-GB CUDA build), then the rest.
RUN pip install --no-cache-dir --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user model/fno_darcy.pt model/fno_darcy.pt

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
