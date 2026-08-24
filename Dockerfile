FROM python:3.12-slim

# The neural guidance detector is what lets the adversarial pass move trained
# classifiers (GPTZero, Copyleaks) rather than only surface-feature checkers.
# It is a real resource commitment, so it is a build arg:
#
#   WITH_DETECTOR=true   (default)  ~1.4GB image, needs ~2GB RAM at runtime
#   WITH_DETECTOR=false             ~250MB image, ~512MB RAM, proxy scoring only
#
# BAKE_MODEL=true copies the weights into the image so the first request does
# not pay a multi-minute download. Set it false to fetch at runtime instead,
# which needs a writable and ideally persistent HF cache.
ARG WITH_DETECTOR=true
ARG BAKE_MODEL=true
ARG GUIDANCE_MODEL=desklib/ai-text-detector-v1.01

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt requirements-detector.txt ./

# The CPU wheel is installed from PyTorch's own index on purpose: the default
# PyPI wheel drags in several GB of CUDA libraries this service never touches.
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_DETECTOR" = "true" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
     && pip install --no-cache-dir -r requirements-detector.txt; \
    fi

# Prefetch the weights. GUIDANCE_MODEL is passed as an argument to the script
# rather than read from the environment, so the value cannot drift from the ARG.
RUN if [ "$WITH_DETECTOR" = "true" ] && [ "$BAKE_MODEL" = "true" ]; then \
        python -c "import sys; from huggingface_hub import snapshot_download; \
snapshot_download(sys.argv[1], allow_patterns=['*.json','*.safetensors','*.model','*.txt','tokenizer*'])" \
        "$GUIDANCE_MODEL"; \
    fi

COPY app ./app

# Carried into the runtime environment so the app defaults to the model that was
# actually baked in.
ENV GUIDANCE_MODEL=${GUIDANCE_MODEL}

# Non-root: EasyPanel runs the container as-is, and there is no reason for this
# process to have write access to its own code. The HF cache lives under /app,
# so this chown also makes a runtime model download possible.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Longer start period than a plain web app: loading the detector weights on a
# small droplet can take the better part of a minute.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health').read()"

# One worker: the workload is I/O-bound on the Anthropic API and already
# concurrent inside the process, and a second worker would load a second copy of
# the detector weights. Scale with replicas, not workers.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]
