FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip uv

WORKDIR /app

# Copy only dependency metadata first so this layer is cached
# when application code changes.
COPY pyproject.toml README.md /app/
RUN uv sync --no-install-project

# Application code and runtime config go in later layers.
COPY src /app/src
COPY scripts /app/scripts
COPY config.json /app/config.json

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "--no-sync", "python", "/app/scripts/generator_cli.py", "--output-dir", "/app/output", "--num-prompts", "8", "--trajectories-per-prompt", "4", "--num-shards", "2"]
