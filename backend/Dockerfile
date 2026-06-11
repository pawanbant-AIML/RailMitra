FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev libpq5 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
COPY backend/.env .

RUN pip install --no-cache-dir -r requirements.txt

# Install the best spaCy large model directly from GitHub (no PyPI issues)
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0.tar.gz

COPY backend/app ./app

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
