import pandas as pd
import pandera as pa
from pathlib import Path

CLEAN = Path(__file__).parent.parent / "data" / "cleaned"

StationSchema = pa.DataFrameSchema(
    {
        "station_code": pa.Column(str, pa.Check(lambda s: s.str.strip().ne("").all()), nullable=False),
        "station_name": pa.Column(str, pa.Check(lambda s: s.str.strip().ne("").all()), nullable=False),
        "city": pa.Column(str, nullable=True),
    }
)

TrainSchema = pa.DataFrameSchema(
    {
        "train_number": pa.Column(str, pa.Check(lambda s: s.str.strip().ne("").all()), nullable=False),
        "train_name": pa.Column(str, pa.Check(lambda s: s.str.strip().ne("").all()), nullable=False),
        "source_station_code": pa.Column(str, nullable=False),
        "destination_station_code": pa.Column(str, nullable=False),
    }
)

def validate():
    # Read as string, preserving empty fields (not converting to NaN)
    stations = pd.read_csv(CLEAN / "stations_clean.csv", keep_default_na=False, dtype="string")
    trains = pd.read_csv(CLEAN / "trains_clean.csv", keep_default_na=False, dtype="string")

    print("Validating stations …")
    StationSchema.validate(stations)
    print("✅ Stations OK")

    print("Validating trains …")
    TrainSchema.validate(trains)
    print("✅ Trains OK")

if __name__ == "__main__":
    validate()