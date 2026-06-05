import json
import pandas as pd
from pathlib import Path
import hashlib

RAW = Path(__file__).parent.parent / "data" / "raw"
CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
CLEAN.mkdir(parents=True, exist_ok=True)

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def clean_stations(json_path):
    data = _load_json(json_path)
    features = data.get("features", [])
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        code = props.get("code")
        name = props.get("name")
        state = props.get("state")
        address = props.get("address")

        # Skip if essential fields are missing or NaN
        if pd.isna(code) or pd.isna(name):
            continue

        code = str(code).upper().strip()
        name = str(name).title().strip()
        state = "" if pd.isna(state) else str(state).title().strip()
        address = "" if pd.isna(address) else str(address).title().strip()

        # Skip dummy codes or empty strings
        if not code or not name or code.startswith(("XX-", "YY-")):
            continue

        city = state if state else address
        rows.append({
            "station_code": code,
            "station_name": name,
            "city": city,
        })

    df = pd.DataFrame(rows)
    # Final safety net: drop any row where station_code or station_name is empty
    df = df.dropna(subset=["station_code", "station_name"])
    df = df[df["station_name"].str.strip() != ""]
    return df

def clean_trains(json_path):
    data = _load_json(json_path)
    if isinstance(data, dict):
        if "features" in data:
            data = data["features"]
        elif "trains" in data:
            data = data["trains"]
        else:
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = []
    elif not isinstance(data, list):
        data = []

    rows = []
    for item in data:
        if isinstance(item, dict) and "properties" in item:
            props = item["properties"]
        else:
            props = item
        rows.append({
            "train_number": str(props.get("number", props.get("train_number", ""))).strip(),
            "train_name": str(props.get("name", props.get("train_name", ""))).title().strip(),
            "source_station_code": str(props.get("from_station_code", props.get("source_station", ""))).upper().strip(),
            "destination_station_code": str(props.get("to_station_code", props.get("destination_station", ""))).upper().strip(),
        })
    df = pd.DataFrame(rows)
    # Drop rows missing train_number or train_name, and drop rows with empty train_name
    df = df.dropna(subset=["train_number", "train_name"])
    df = df[df["train_name"].str.strip() != ""]
    return df

def clean_schedules(json_path):
    data = _load_json(json_path)
    if not isinstance(data, list):
        raise ValueError("schedules.json must be a list")
    rows = []
    for stop in data:
        arr = stop.get("arrival")
        dep = stop.get("departure")
        arr = None if arr == "None" else arr
        dep = None if dep == "None" else dep
        day = stop.get("day")
        if day is None:
            day = 1
        else:
            day = int(day)
        rows.append({
            "train_number": str(stop.get("train_number", "")).strip(),
            "station_code": str(stop.get("station_code", "")).upper().strip(),
            "arrival_time": arr,
            "departure_time": dep,
            "day": day,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def to_minutes(t):
        try:
            parts = t.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    df["dep_min"] = df["departure_time"].apply(to_minutes)
    df = df.sort_values(["train_number", "day", "dep_min"])
    df["sequence"] = df.groupby("train_number").cumcount() + 1
    df = df.drop(columns=["day", "dep_min"])
    df["distance_km"] = None
    return df[["train_number", "sequence", "station_code", "arrival_time", "departure_time", "distance_km"]]

def checksum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    stations = clean_stations(RAW / "stations.json")
    stations.to_csv(CLEAN / "stations_clean.csv", index=False)
    print("Stations checksum:", checksum(CLEAN / "stations_clean.csv"))

    trains = clean_trains(RAW / "trains.json")
    trains.to_csv(CLEAN / "trains_clean.csv", index=False)
    print("Trains checksum:", checksum(CLEAN / "trains_clean.csv"))

    schedules = clean_schedules(RAW / "schedules.json")
    schedules.to_csv(CLEAN / "schedules_clean.csv", index=False)
    print("Schedules checksum:", checksum(CLEAN / "schedules_clean.csv"))

if __name__ == "__main__":
    main()