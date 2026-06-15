from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv("backend/.env")

engine = create_engine(os.getenv("DATABASE_URL"))

tables = [
    "stations",
    "trains",
    "routes",
    "fares",
    "bookings"
]

with engine.connect() as conn:
    for table in tables:
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()

        print(f"{table}: {count:,}")