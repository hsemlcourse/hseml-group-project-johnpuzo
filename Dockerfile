FROM python:3.12-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10

COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 300 --retries 10 -r requirements-docker.txt

COPY src ./src
COPY models ./models

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]