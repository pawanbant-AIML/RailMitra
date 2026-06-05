import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
from train_nlp import build_dataset   # ← works when run from scripts/

MODEL_DIR = Path(__file__).parent.parent / "models" / "nlp"

def load_models():
    with open(MODEL_DIR / "intent_classifier.pkl", "rb") as f:
        intent_clf = pickle.load(f)
    with open(MODEL_DIR / "entity_extractor.pkl", "rb") as f:
        entity_extractor = pickle.load(f)
    return intent_clf, entity_extractor

def evaluate():
    df = build_dataset()
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    intent_clf, _ = load_models()
    preds = intent_clf.predict(test_df["text"])
    report = classification_report(test_df["intent"], preds, digits=3)
    print("Intent Classification Report:\n")
    print(report)

if __name__ == "__main__":
    evaluate()