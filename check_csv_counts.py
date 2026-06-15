from pathlib import Path
import pandas as pd

clean = Path("data/cleaned")
proc = Path("data/processed")

files = {
    "stations": clean / "stations_clean.csv",
    "trains": clean / "trains_clean.csv",
    "routes": clean / "schedules_clean.csv",
    "fares": proc / "fares_clean.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"{name}: {len(df):,} rows")
