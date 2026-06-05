import pandas as pd
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
from spacy.pipeline import EntityRuler

CLEAN = Path(__file__).parent.parent / "data" / "cleaned"
MODEL_DIR = Path(__file__).parent.parent / "models" / "nlp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_dataset():
    stations = pd.read_csv(CLEAN / "stations_clean.csv")
    trains = pd.read_csv(CLEAN / "trains_clean.csv")

    texts = []
    intents = []

    # Search intent
    for _, row in stations.sample(min(200, len(stations)), random_state=42).iterrows():
        dest = stations.sample(1).iloc[0]["station_name"]
        txt = f"Find trains from {row['station_name']} to {dest} tomorrow"
        texts.append(txt)
        intents.append("search_train")

    # Booking intent
    for _, row in trains.sample(min(100, len(trains)), random_state=1).iterrows():
        src = stations.sample(1).iloc[0]["station_name"]
        dst = stations.sample(1).iloc[0]["station_name"]
        txt = f"Book 2 sleeper tickets from {src} to {dst} on 2023-12-25"
        texts.append(txt)
        intents.append("book_ticket")

    # Cancel intent
    for i in range(50):
        txt = f"Cancel booking id {i+1000}"
        texts.append(txt)
        intents.append("cancel_ticket")

    # History intent
    texts.append("Show my booking history")
    intents.append("booking_history")

    return pd.DataFrame({"text": texts, "intent": intents})

def train_intent():
    df = build_dataset()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    pipeline.fit(df["text"], df["intent"])
    with open(MODEL_DIR / "intent_classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("Intent classifier saved")

def train_entity_extractor():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = []

    stations = pd.read_csv(CLEAN / "stations_clean.csv")
    # Add station codes
    for code in stations["station_code"].dropna().unique():
        patterns.append({"label": "STATION", "pattern": code})
    # Add station names (case‑insensitive)
    for name in stations["station_name"].dropna().unique():
        # Lowercase pattern for robust matching
        patterns.append({"label": "STATION", "pattern": [{"LOWER": name.lower()}]})

    # Date pattern
    patterns.append({"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{4}-\d{2}-\d{2}"}}]})
    # Passenger count words
    for word in ["one", "two", "three", "four", "five", "six"]:
        patterns.append({"label": "PASSENGER_COUNT", "pattern": [{"LOWER": word}]})
    # Class types
    for cl in ["SL", "3A", "2A", "1A", "CC", "EC"]:
        patterns.append({"label": "CLASS_TYPE", "pattern": cl})

    ruler.add_patterns(patterns)
    nlp.to_disk(MODEL_DIR / "entity_extractor")
    with open(MODEL_DIR / "entity_extractor.pkl", "wb") as f:
        pickle.dump(nlp, f)
    print("Entity extractor saved")

if __name__ == "__main__":
    train_intent()
    train_entity_extractor()