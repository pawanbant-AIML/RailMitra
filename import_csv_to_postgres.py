import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv("backend/.env")

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

print("Connecting to PostgreSQL...")

# Clear existing sample data
with engine.begin() as conn:
    print("Clearing existing data...")

    for table in [
        "routes",
        "fares",
        "trains",
        "stations"
    ]:
        try:
            conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            print(f"Cleared {table}")
        except Exception as e:
            print(f"Skip {table}: {e}")

base = Path("data")

files = [
    ("stations", base / "cleaned" / "stations_clean.csv"),
    ("trains", base / "cleaned" / "trains_clean.csv"),
    ("routes", base / "cleaned" / "schedules_clean.csv"),
    ("fares", base / "processed" / "fares_clean.csv"),
]

for table, file_path in files:

    print(f"\nLoading {table}...")
    print(file_path)

    total = 0

    for chunk in pd.read_csv(file_path, chunksize=5000):

        chunk.to_sql(
            table,
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )

        total += len(chunk)

        print(f"  Imported {total:,} rows", end="\r")

    print(f"\n✅ {table}: {total:,} rows imported")

print("\nVerification:")

with engine.connect() as conn:

    for table in [
        "stations",
        "trains",
        "routes",
        "fares"
    ]:

        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()

        print(f"{table}: {count:,}")

print("\n🎉 Import completed")