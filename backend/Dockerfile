FROM python:3.12-slim

WORKDIR /app

# Install system deps (psycopg2 needs libpq)
RUN apt-get update && apt-get install -y gcc libpq-dev libpq5 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app ./app

# Copy the SQLite DB as fallback (Render PostgreSQL is the primary store)
COPY train_ticket.db /app/train_ticket.db

EXPOSE 10000

# Render injects PORT env var; fall back to 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
