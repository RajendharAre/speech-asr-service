FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /srv
COPY requirements.txt requirements-asr.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-asr.txt

COPY app ./app
RUN useradd -m appuser && mkdir -p /models && chown -R appuser /models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
