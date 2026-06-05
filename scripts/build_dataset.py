import pandas as pd
from pathlib import Path

CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def build():
    stations = pd.read_csv(CLEAN / "stations_clean.csv")
    trains = pd.read_csv(CLEAN / "trains_clean.csv")
    routes = pd.read_csv(CLEAN / "schedules_clean.csv")

    stations.to_parquet(PROC / "stations.parquet", index=False)
    trains.to_parquet(PROC / "trains.parquet", index=False)
    routes.to_parquet(PROC / "routes.parquet", index=False)

    fares_path = PROC / "fares_clean.csv"
    if fares_path.exists():
        fares = pd.read_csv(fares_path)
        fares.to_parquet(PROC / "fares.parquet", index=False)
        print("Fares parquet written")

    print("Processed parquet files written to:", PROC)

if __name__ == "__main__":
    build()