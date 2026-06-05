import os
import json
import requests
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DATA_URLS = {
    "stations": "https://raw.githubusercontent.com/datameet/railways/master/stations.json",
    "trains": "https://raw.githubusercontent.com/datameet/railways/master/trains.json",
    "schedules": "https://raw.githubusercontent.com/datameet/railways/master/schedules.json",
}

def download_file(name: str, url: str) -> None:
    dest = DATA_ROOT / f"{name}.json"
    if dest.exists():
        print(f"{dest.name} already exists — skipping")
        return
    print(f"Downloading {name} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {dest}")

def download():
    for name, url in DATA_URLS.items():
        download_file(name, url)

if __name__ == "__main__":
    download()