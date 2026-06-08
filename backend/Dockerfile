FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev libpq5 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
COPY backend/.env .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/train_ticket.db ./train_ticket.db
COPY backend/app ./app
COPY populate_db.py ./populate_db.py

EXPOSE 10000

# Run population script before starting server
CMD ["sh", "-c", "python populate_db.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
