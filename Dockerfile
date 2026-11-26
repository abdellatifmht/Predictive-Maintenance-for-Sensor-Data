FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV PM_PROJECT_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --locked --no-dev --no-editable \
    && useradd --create-home appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

USER appuser
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "predictive_maintenance.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
