FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Runtime dependencies — the serving subset of pyproject.toml.
# scikit-learn is pinned to match the version the model artifacts were
# serialized with, preventing InconsistentVersionWarning at load time.
RUN pip install --no-cache-dir \
    "pandas>=2.0,<4" \
    "numpy>=1.24,<3" \
    "scikit-learn==1.9.0" \
    "pyarrow>=14,<25" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "python-multipart>=0.0.6"

# Application code and trained artifacts required to serve predictions
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
