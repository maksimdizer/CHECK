FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==26.3 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 qa
COPY --chown=qa qa ./qa
COPY --chown=qa cards ./cards
USER qa
CMD ["uvicorn", "qa.api:app", "--host", "0.0.0.0", "--port", "8000"]
