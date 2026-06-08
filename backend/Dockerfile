FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev libpq5 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
COPY backend/.env .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

EXPOSE 10000

# Use PORT environment variable, default to 10000
# Uses DATABASE_URL from .env (PostgreSQL on Render)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
