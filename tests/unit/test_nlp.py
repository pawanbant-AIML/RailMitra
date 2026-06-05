import pytest
from app.services.nlp_service import NLPService

def test_intent_prediction():
    nlp = NLPService()
    intent, ents = nlp.predict("Find trains from Bangalore to Chennai tomorrow")
    assert intent == "search_train"
    # at least one of source/destination should be found
    assert "source_station" in ents or "destination_station" in ents