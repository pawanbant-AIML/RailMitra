import sys
from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

env_path = BACKEND_DIR / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set.")
    sys.exit(1)

try:
    from app.core.config import settings
    from app.models.train_models import Base
except ImportError as e:
    print("ERROR: Could not import backend modules.")
    print(f"Details: {e}")
    sys.exit(1)

from sqlalchemy import create_engine
import pandas as pd

def import_data():
    engine = create_engine(DATABASE_URL, echo=True)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
    PROC = Path(__file__).parent.parent / "data" / "processed"

    stations_path = CLEAN / "stations_clean.csv"
    if stations_path.exists():
        df = pd.read_csv(stations_path)
        df.to_sql("stations", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} stations")
    else:
        print("stations_clean.csv not found")

    trains_path = CLEAN / "trains_clean.csv"
    if trains_path.exists():
        df = pd.read_csv(trains_path)
        df.to_sql("trains", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} trains")
    else:
        print("trains_clean.csv not found")

    routes_path = CLEAN / "schedules_clean.csv"
    if routes_path.exists():
        df = pd.read_csv(routes_path)
        df.to_sql("routes", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} route entries")
    else:
        print("schedules_clean.csv not found")

    fares_path = PROC / "fares_clean.csv"
    if fares_path.exists():
        df = pd.read_csv(fares_path)
        df.to_sql("fares", engine, if_exists="replace", index=False)
        print(f"Imported {len(df)} fare entries")
    else:
        print("fares_clean.csv not found")

    print("Database import complete!")

if __name__ == "__main__":
    import_data()
