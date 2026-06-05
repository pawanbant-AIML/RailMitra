import sys
from pathlib import Path
import os

# 1. Add the backend folder to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# 2. Load environment variables from backend/.env if it exists
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback: use SQLite for demo if .env not found
    os.environ.setdefault("DATABASE_URL", "sqlite:///./train_ticket.db")

# 3. Now import the backend modules
try:
    from app.core.config import settings
    from app.models.train_models import Base
except ImportError as e:
    print("ERROR: Could not import backend modules.")
    print("Make sure you have installed the backend dependencies:")
    print("    pip install -r backend/requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

from sqlalchemy import create_engine
import pandas as pd

def import_data():
    # Create engine using the URL from settings (already loaded from .env or environment)
    engine = create_engine(settings.DATABASE_URL, echo=True)

    # Drop and recreate tables (clean slate for demo)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
    PROC = Path(__file__).parent.parent / "data" / "processed"

    # Stations
    stations_path = CLEAN / "stations_clean.csv"
    if stations_path.exists():
        df = pd.read_csv(stations_path)
        df.to_sql("stations", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} stations")
    else:
        print("⚠ stations_clean.csv not found – run clean_data.py first")

    # Trains
    trains_path = CLEAN / "trains_clean.csv"
    if trains_path.exists():
        df = pd.read_csv(trains_path)
        df.to_sql("trains", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} trains")
    else:
        print("⚠ trains_clean.csv not found")

    # Routes (schedules)
    routes_path = CLEAN / "schedules_clean.csv"
    if routes_path.exists():
        df = pd.read_csv(routes_path)
        df.to_sql("routes", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} route entries")
    else:
        print("⚠ schedules_clean.csv not found")

    # Mock fares (if exists)
    fares_path = PROC / "fares_clean.csv"
    if fares_path.exists():
        df = pd.read_csv(fares_path)
        df.to_sql("fares", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} mock fare entries")
    else:
        print("No fare file found – skipping fares")

    print("✅ Database import complete!")

if __name__ == "__main__":
    import_data()