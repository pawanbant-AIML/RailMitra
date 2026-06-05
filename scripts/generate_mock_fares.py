import pandas as pd
from pathlib import Path

CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

RATES = {
    "SL": 0.50,
    "3A": 0.80,
    "2A": 1.00,
    "1A": 1.50,
}

def generate():
    trains = pd.read_csv(CLEAN / "trains_clean.csv")
    fares = []
    for _, row in trains.iterrows():
        dist = 800   # demo placeholder
        for class_type, rate in RATES.items():
            amount = round(dist * rate, 2)
            fares.append({
                "train_number": row["train_number"],
                "class_type": class_type,
                "amount": amount,
            })
    df = pd.DataFrame(fares)
    df.to_csv(PROC / "fares_clean.csv", index=False)
    print(f"Mock fares generated: {len(df)} rows → {PROC / 'fares_clean.csv'}")

if __name__ == "__main__":
    generate()