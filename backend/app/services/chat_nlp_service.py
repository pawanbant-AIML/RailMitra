"""
NLP Service for train booking conversation analysis.
Uses SpaCy for entity extraction and rule-based intent detection.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pydantic import BaseModel
import spacy
from difflib import get_close_matches

# ==================== SCHEMAS ====================

class Entity(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    via_stations: Optional[List[str]] = None
    date: Optional[str] = None  # ISO-8601: YYYY-MM-DD
    time: Optional[str] = None  # HH:MM in 24-hour format
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    budget: Optional[float] = None
    preference: Optional[str] = None  # shortest, cheapest, fastest


class ChatAnalysisRequest(BaseModel):
    user_message: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatAnalysisResponse(BaseModel):
    intent: str
    confidence: float
    entities: Entity
    resolved_context: Dict[str, Any]
    missing_required_slots: List[str]
    clarification_needed: bool
    clarification_question: Optional[str] = None
    next_action: str  # SEARCH_ROUTE, ESTIMATE_FARE, BOOK, COMPARE_ROUTES, ROUTE_ANALYSIS, ASK_CLARIFICATION, UNKNOWN
    action_payload: Optional[Dict[str, Any]] = None
    memory_patch: Optional[Dict[str, Any]] = None


# ==================== STATION MAPPING ====================

CITY_TO_CODES = {
    "bangalore": ["SBC", "YPR"],
    "bengaluru": ["SBC", "YPR"],
    "mumbai": ["CSTM", "BCT", "BDTS"],
    "bombay": ["CSTM", "BCT"],
    "delhi": ["NDLS", "DLI", "NZM"],
    "new delhi": ["NDLS"],
    "chennai": ["MAS", "MS"],
    "madras": ["MAS"],
    "kolkata": ["HWH", "SDAH"],
    "calcutta": ["HWH"],
    "hyderabad": ["SC", "HYB"],
    "secunderabad": ["SC"],
    "pune": ["PUNE", "PNA"],
    "jaipur": ["JP", "JPI"],
    "lucknow": ["LKO", "LJN"],
    "patna": ["PNBE", "PPTA"],
    "bhopal": ["BPL"],
    "nagpur": ["NGP"],
    "surat": ["ST"],
    "ahmedabad": ["ADI"],
    "varanasi": ["BSB", "VARANASI"],
    "kochi": ["ERS"],
    "trivandrum": ["TVC"],
    "guwahati": ["GHY"],
    "amritsar": ["ASR"],
    "chandigarh": ["CDG"],
}

CLASS_ALIASES = {
    "sleeper": "SL",
    "sl": "SL",
    "ac": "2A",  # Default AC
    "2ac": "2A",
    "3ac": "3A",
    "1ac": "1A",
    "first": "1A",
    "second": "2A",
    "third": "3A",
    "chair": "CC",
    "cc": "CC",
    "2s": "2S",
    "second sitting": "2S",
    "general": "GN",
    "gn": "GN",
}

PREFERENCE_KEYWORDS = {
    "shortest": "shortest",
    "shortest route": "shortest",
    "minimum stops": "shortest",
    "cheapest": "cheapest",
    "cheapest route": "cheapest",
    "budget": "cheapest",
    "fastest": "fastest",
    "fastest route": "fastest",
    "quick": "fastest",
    "minimum time": "fastest",
    "compare": "compare",
}

# ==================== NLP SERVICE ====================

class ChatNLPService:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def analyze(self, request: ChatAnalysisRequest) -> ChatAnalysisResponse:
        """Main entry point for chat analysis."""
        user_message = request.user_message.strip()
        
        # Get conversation memory
        memory = self._build_memory(request.conversation_history)
        
        # Extract intent
        intent = self._detect_intent(user_message)
        
        # Extract entities
        entities = self._extract_entities(user_message, memory)
        
        # Merge with memory (prefer current message over history)
        resolved_context = {**memory, **entities.dict(exclude_none=True)}
        
        # Check for missing required slots
        missing_slots = self._check_missing_slots(intent, resolved_context)
        
        # Generate clarification if needed
        clarification_needed = len(missing_slots) > 0
        clarification_question = self._generate_clarification(missing_slots) if clarification_needed else None
        
        # Determine next action
        next_action = self._determine_next_action(intent, clarification_needed, resolved_context)
        
        # Build action payload
        action_payload = self._build_action_payload(intent, resolved_context, next_action)
        
        # Confidence score
        confidence = self._calculate_confidence(intent, entities, missing_slots)
        
        return ChatAnalysisResponse(
            intent=intent,
            confidence=confidence,
            entities=entities,
            resolved_context=resolved_context,
            missing_required_slots=missing_slots,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            next_action=next_action,
            action_payload=action_payload,
            memory_patch=entities.dict(exclude_none=True),
        )

    def _build_memory(self, history: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Build context from conversation history."""
        memory = {}
        if not history:
            return memory
        
        # Get last few relevant messages
        for msg in history[-3:]:  # Last 3 turns
            if msg.get("role") == "assistant" and "entities" in msg:
                try:
                    entities = msg["entities"]
                    for key in ["source", "destination", "date", "travel_class"]:
                        if entities.get(key) and not memory.get(key):
                            memory[key] = entities[key]
                except:
                    pass
        
        return memory

    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message."""
        msg_lower = message.lower()
        
        # Check for specific intents
        if any(word in msg_lower for word in ["book", "reserve", "ticket", "confirm"]):
            return "BOOK_TICKET"
        
        if any(word in msg_lower for word in ["how much", "cost", "fare", "price", "charge"]):
            return "FARE_ESTIMATE"
        
        if any(word in msg_lower for word in ["shortest", "minimum stops", "route"]):
            return "SHORTEST_ROUTE"
        
        if any(word in msg_lower for word in ["cheapest", "budget", "affordable", "cheap"]):
            return "CHEAPEST_ROUTE"
        
        if any(word in msg_lower for word in ["fastest", "quick", "minimum time"]):
            return "FASTEST_ROUTE"
        
        if any(word in msg_lower for word in ["compare", "which is", "better"]):
            return "COMPARE_ROUTES"
        
        if any(word in msg_lower for word in ["what", "which", "how", "tell me", "show"]):
            return "CLARIFY"
        
        # Default: assume route search if has "to" or "from"
        if " to " in msg_lower or " from " in msg_lower:
            return "ROUTE_SEARCH"
        
        return "UNKNOWN"

    def _extract_entities(self, message: str, memory: Dict) -> Entity:
        """Extract entities from message."""
        msg_lower = message.lower()
        entities = Entity()
        
        # Extract source and destination
        source, destination = self._extract_stations(msg_lower)
        if source:
            entities.source = source
        if destination:
            entities.destination = destination
        
        # Extract via stations
        via = self._extract_via_stations(msg_lower)
        if via:
            entities.via_stations = via
        
        # Extract date
        date = self._extract_date(msg_lower)
        if date:
            entities.date = date
        
        # Extract time
        time = self._extract_time(msg_lower)
        if time:
            entities.time = time
        
        # Extract travel class
        travel_class = self._extract_class(msg_lower)
        if travel_class:
            entities.travel_class = travel_class
        
        # Extract passengers
        passengers = self._extract_passengers(msg_lower)
        if passengers:
            entities.passengers = passengers
        
        # Extract budget
        budget = self._extract_budget(msg_lower)
        if budget:
            entities.budget = budget
        
        # Extract preference
        preference = self._extract_preference(msg_lower)
        if preference:
            entities.preference = preference
        
        return entities

    def _extract_stations(self, message: str) -> tuple:
        """Extract source and destination stations."""
        source = None
        destination = None
        
        # Handle "from X to Y" or "X to Y"
        if " to " in message:
            parts = message.split(" to ")
            if len(parts) >= 2:
                # Extract source from first part
                source_text = parts[0].strip().split()[-1]  # Last word before "to"
                destination_text = parts[1].strip().split()[0]  # First word after "to"
                
                source = self._normalize_station(source_text)
                destination = self._normalize_station(destination_text)
        
        # Handle "from X via Y to Z"
        if " from " in message and " to " in message:
            match = re.search(r"from\s+(\w+)\s+to\s+(\w+)", message)
            if match:
                source = self._normalize_station(match.group(1))
                destination = self._normalize_station(match.group(2))
        
        return source, destination

    def _extract_via_stations(self, message: str) -> Optional[List[str]]:
        """Extract via stations."""
        via_stations = []
        
        match = re.search(r"via\s+([^,]+?)(?:\s+to|\s+arrive|\s+on|$)", message)
        if match:
            via_text = match.group(1).strip()
            stations = [s.strip() for s in via_text.split(",")]
            for station in stations:
                normalized = self._normalize_station(station)
                if normalized:
                    via_stations.append(normalized)
        
        return via_stations if via_stations else None

    def _extract_date(self, message: str) -> Optional[str]:
        """Extract and normalize date."""
        today = datetime.now().date()
        
        # Relative dates
        if "today" in message:
            return today.isoformat()
        if "tomorrow" in message:
            return (today + timedelta(days=1)).isoformat()
        if "next" in message:
            # "next monday", "next friday"
            for day_name in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                if day_name in message:
                    # Calculate next occurrence of that day
                    days_ahead = (["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(day_name) - today.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    return (today + timedelta(days=days_ahead)).isoformat()
        
        # Absolute dates
        # Try YYYY-MM-DD format
        match = re.search(r"(\d{4}-\d{2}-\d{2})", message)
        if match:
            return match.group(1)
        
        # Try DD/MM or DD-MM format
        match = re.search(r"(\d{1,2})[/-](\d{1,2})", message)
        if match:
            day, month = int(match.group(1)), int(match.group(2))
            year = today.year
            try:
                date_obj = datetime(year, month, day).date()
                if date_obj >= today:
                    return date_obj.isoformat()
            except:
                pass
        
        # Try written dates like "10 june"
        match = re.search(r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)", message)
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }
            month = month_map.get(month_name)
            year = today.year
            try:
                date_obj = datetime(year, month, day).date()
                if date_obj >= today:
                    return date_obj.isoformat()
            except:
                pass
        
        return None

    def _extract_time(self, message: str) -> Optional[str]:
        """Extract and normalize time."""
        # Try HH:MM format
        match = re.search(r"(\d{1,2}):(\d{2})", message)
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            if 0 <= hour < 24 and 0 <= minute < 60:
                return f"{hour:02d}:{minute:02d}"
        
        # Try "morning", "afternoon", "evening"
        if "morning" in message:
            return "08:00"
        if "afternoon" in message:
            return "14:00"
        if "evening" in message:
            return "18:00"
        if "night" in message:
            return "21:00"
        
        return None

    def _extract_class(self, message: str) -> Optional[str]:
        """Extract and normalize travel class."""
        msg_lower = message.lower()
        
        for alias, canonical in CLASS_ALIASES.items():
            if alias in msg_lower:
                return canonical
        
        return None

    def _extract_passengers(self, message: str) -> Optional[int]:
        """Extract number of passengers."""
        # Match "2 passengers", "for 3", "3 people", etc.
        match = re.search(r"(\d+)\s*(passenger|person|people|traveler)", message.lower())
        if match:
            return int(match.group(1))
        
        # Match just numbers with adult/child context
        match = re.search(r"(\d+)\s*(adult|child|kids|senior)", message.lower())
        if match:
            return int(match.group(1))
        
        # Match standalone numbers at start like "2 to delhi"
        match = re.search(r"^(\d+)\s+", message)
        if match:
            num = int(match.group(1))
            if num <= 10:  # Reasonable max
                return num
        
        return None

    def _extract_budget(self, message: str) -> Optional[float]:
        """Extract budget."""
        # Match currency amounts
        match = re.search(r"[₹$]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", message)
        if match:
            amount_str = match.group(1).replace(",", "")
            return float(amount_str)
        
        # Match "5000 rupees" style
        match = re.search(r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(rupees|inr|dollars|usd)", message.lower())
        if match:
            amount_str = match.group(1).replace(",", "")
            return float(amount_str)
        
        return None

    def _extract_preference(self, message: str) -> Optional[str]:
        """Extract travel preference."""
        msg_lower = message.lower()
        
        for keyword, preference in PREFERENCE_KEYWORDS.items():
            if keyword in msg_lower:
                return preference
        
        return None

    def _normalize_station(self, station: str) -> Optional[str]:
        """Normalize city/station to canonical code."""
        station = station.lower().strip()
        
        # Direct city to code lookup
        if station in CITY_TO_CODES:
            return CITY_TO_CODES[station][0]
        
        # Fuzzy match
        city_matches = get_close_matches(station, CITY_TO_CODES.keys(), n=1, cutoff=0.6)
        if city_matches:
            return CITY_TO_CODES[city_matches[0]][0]
        
        # Return as-is if could be a station code
        if len(station) <= 4 and station.isalpha():
            return station.upper()
        
        return None

    def _check_missing_slots(self, intent: str, context: Dict) -> List[str]:
        """Check for missing required slots based on intent."""
        missing = []
        
        if intent in ["ROUTE_SEARCH", "SHORTEST_ROUTE", "CHEAPEST_ROUTE", "FASTEST_ROUTE", "COMPARE_ROUTES"]:
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")
            if not context.get("date"):
                missing.append("date")
        
        if intent == "BOOK_TICKET":
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")
            if not context.get("date"):
                missing.append("date")
            if not context.get("travel_class"):
                missing.append("travel_class")
            if not context.get("passengers"):
                missing.append("passengers")
        
        if intent == "FARE_ESTIMATE":
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")
        
        return missing

    def _generate_clarification(self, missing_slots: List[str]) -> str:
        """Generate a single clarification question."""
        if not missing_slots:
            return None
        
        slot = missing_slots[0]  # Ask about first missing slot
        
        questions = {
            "source": "Where are you traveling from?",
            "destination": "Where are you traveling to?",
            "date": "When do you want to travel?",
            "travel_class": "What class would you prefer (Sleeper, 2AC, 3AC, 1AC)?",
            "passengers": "How many passengers?",
            "budget": "What's your budget?",
        }
        
        return questions.get(slot, f"Please provide {slot}.")

    def _determine_next_action(self, intent: str, clarification_needed: bool, context: Dict) -> str:
        """Determine next action based on intent and context."""
        if clarification_needed:
            return "ASK_CLARIFICATION"
        
        if intent == "UNKNOWN":
            return "UNKNOWN"
        
        if intent == "ROUTE_SEARCH":
            return "SEARCH_ROUTE"
        
        if intent in ["SHORTEST_ROUTE", "CHEAPEST_ROUTE", "FASTEST_ROUTE"]:
            return "ROUTE_ANALYSIS"
        
        if intent == "COMPARE_ROUTES":
            return "COMPARE_ROUTES"
        
        if intent == "FARE_ESTIMATE":
            return "ESTIMATE_FARE"
        
        if intent == "BOOK_TICKET":
            return "BOOK"
        
        if intent == "CLARIFY":
            return "CLARIFY"
        
        return "UNKNOWN"

    def _build_action_payload(self, intent: str, context: Dict, next_action: str) -> Optional[Dict]:
        """Build payload for next action."""
        if next_action == "ASK_CLARIFICATION" or next_action == "UNKNOWN":
            return None
        
        payload = {}
        
        if next_action in ["SEARCH_ROUTE", "ROUTE_ANALYSIS", "ESTIMATE_FARE", "COMPARE_ROUTES"]:
            payload = {
                "source": context.get("source"),
                "destination": context.get("destination"),
                "date": context.get("date"),
                "via_stations": context.get("via_stations"),
                "travel_class": context.get("travel_class"),
                "passengers": context.get("passengers", 1),
                "preference": context.get("preference"),  # For ROUTE_ANALYSIS
            }
        
        if next_action == "BOOK":
            payload = {
                "source": context.get("source"),
                "destination": context.get("destination"),
                "date": context.get("date"),
                "travel_class": context.get("travel_class"),
                "passengers": context.get("passengers"),
            }
        
        return payload if payload else None

    def _calculate_confidence(self, intent: str, entities: Entity, missing_slots: List[str]) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base
        
        # Reduce confidence for unknown intent
        if intent == "UNKNOWN":
            confidence = 0.2
        
        # Increase for high-confidence intents
        if intent in ["BOOK_TICKET", "ROUTE_SEARCH"]:
            confidence = 0.8
        
        # Reduce for missing slots
        confidence -= len(missing_slots) * 0.15
        
        # Increase if multiple entities extracted
        entity_count = sum(1 for v in entities.dict().values() if v is not None)
        confidence += min(entity_count * 0.05, 0.2)
        
        return max(0.1, min(1.0, confidence))
