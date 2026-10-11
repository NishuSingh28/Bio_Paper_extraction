# One-command setup:
#   docker build -t paper-extractor .
#   docker run --rm -p 8501:8501 --env-file .env \
#     -v "$HOME/.paper-extractor:/data" paper-extractor
#
# The paper cache and the runs live in the /data volume, so they survive the
# container. To reach Ollama on the host, set OLLAMA_HOST in .env to
# http://host.docker.internal:11434

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PAPER_CACHE_DIR=/data/cache \
    PAPER_RUNS_DIR=/data/runs \
    HF_HOME=/data/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller than the default build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home app && mkdir -p /data && chown -R app /data /app
USER app
VOLUME ["/data"]
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
