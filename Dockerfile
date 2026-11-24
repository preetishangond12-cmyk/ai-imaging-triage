FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements-cloud.txt .
RUN python -m pip install --no-cache-dir -r requirements-cloud.txt \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.14.0" "torchvision>=0.29.0"

COPY backend ./backend
COPY frontend ./frontend
COPY ["ai model/inference.py", "ai model/labels.py", "ai model/triage.py", "./ai model/"]
COPY ["ai model/checkpoints/", "ai model/checkpoints/"]

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
