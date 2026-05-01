FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── System dependencies ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ─────────────────────────────────────
RUN pip install --upgrade pip

# ── FORCE CPU PyTorch (CRITICAL FIX) ────────────────
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# ── Install Python deps ─────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── App code ────────────────────────────────────────
COPY . .

# ── Environment ─────────────────────────────────────
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

EXPOSE 8000

# ── Run server ──────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]