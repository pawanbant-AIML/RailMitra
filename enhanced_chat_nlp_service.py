from dataclasses import dataclass
from typing import List, Any, Optional, Dict

# Compatibility shim to satisfy legacy tests that import enhanced_chat_nlp_service
# Wraps QueryUnderstanding and adapts its output to the legacy test interface.

try:
    from app.agent.query_understanding import QueryUnderstanding
except Exception:
    from backend.app.agent.query_understanding import QueryUnderstanding


@dataclass
class ChatAnalysisRequest:
    user_message: str
    conversation_history: List[dict]


@dataclass
class Entity:
    source: Optional[str] = None
    destination: Optional[str] = None
    travel_class: Optional[str] = None
    date: Optional[str] = None
    passengers: Optional[int] = None
    booking_id: Optional[int] = None
    train_number: Optional[str] = None
    budget: Optional[float] = None
    preference: Optional[str] = None


@dataclass
class ChatAnalysisResponse:
    entities: Entity
    intent: str
    clarification_needed: bool = False
    missing_required_slots: List[str] = None
    missing_slots: List[str] = None
    next_action: Optional[str] = None
    confidence: float = 0.0
    normalized_text: Optional[str] = None


_INTENT_MAP = {
    "train_search": "ROUTE_SEARCH",
    "booking_create": "BOOK_TICKET",
    "booking_history": "BOOKING_HISTORY",
    "booking_cancel": "CANCEL_BOOKING",
    "fare_query": "FARE_ESTIMATE",
    "route_query": "CHECK_ROUTE",
    "station_query": "STATION_QUERY",
    "train_info": "TRAIN_INFO",
    "greeting": "GREETING",
}


def _map_intent(interp) -> str:
    base = interp.intent or "train_search"
    sub = set(interp.sub_intents or [])
    if base == "train_search":
        if "cheapest" in sub:
            return "CHEAPEST_ROUTE"
        if "fastest" in sub:
            return "FASTEST_ROUTE"
        if "compare" in sub:
            return "COMPARE_ROUTES"
        return _INTENT_MAP.get(base, "ROUTE_SEARCH")
    return _INTENT_MAP.get(base, base.upper())


def _determine_next_action(intent_label: str, clarification_needed: bool) -> str:
    if clarification_needed:
        return "ASK_CLARIFICATION"
    if intent_label in ("ROUTE_SEARCH", "CHEAPEST_ROUTE", "FASTEST_ROUTE", "COMPARE_ROUTES"):
        return "SEARCH_ROUTE"
    if intent_label == "BOOK_TICKET":
        return "BOOK_TICKET"
    if intent_label == "CHECK_ROUTE":
        return "CHECK_ROUTE"
    return "RESPOND"


class ChatNLPService:
    def __init__(self) -> None:
        self._qu = QueryUnderstanding()

    def analyze(self, req: ChatAnalysisRequest) -> ChatAnalysisResponse:
        text = (req.user_message or "").strip()
        # Find last user turn in history (simple heuristic)
        # Build memory by interpreting user turns in history sequentially so that
        # context accumulates (source/destination/date/passengers/class etc.).
        memory: Dict[str, Any] = {}
        previous_result: Dict[str, Any] = {}
        for msg in (req.conversation_history or []):
            if msg.get("role") == "user" and msg.get("content"):
                part = msg.get("content")
                interp_part = self._qu.interpret(part, memory=memory, previous_result=previous_result)
                # merge slots into memory
                try:
                    memory.update({k: v for k, v in (interp_part.slots.__dict__ if interp_part and interp_part.slots else {}).items() if v not in (None, "", [])})
                except Exception:
                    pass
                previous_result = interp_part.to_dict() if interp_part else {}

        interp = self._qu.interpret(text, memory=memory, previous_result=previous_result)
        slots = interp.slots

        pref_raw = slots.preference
        pref_map = {
            "low_cost": "cheapest",
            "fastest": "fastest",
            "shortest": "shortest",
            "comfort": "comfort",
            "direct_only": "direct_only",
            "overnight": "overnight",
        }
        ent = Entity(
            source=slots.source,
            destination=slots.destination,
            travel_class=slots.travel_class,
            date=slots.travel_date,
            passengers=slots.passengers,
            booking_id=(int(slots.booking_id) if slots.booking_id else None),
            train_number=slots.train_number,
            budget=(float(slots.budget_max) if slots.budget_max is not None else None),
            preference=pref_map.get(pref_raw, pref_raw),
        )

        intent_label = _map_intent(interp)
        clarification = bool(interp.clarification_needed)
        missing = list(interp.missing_slots or [])
        next_action = _determine_next_action(intent_label, clarification)

        return ChatAnalysisResponse(
            entities=ent,
            intent=intent_label,
            clarification_needed=clarification,
            missing_required_slots=missing,
            missing_slots=missing,
            next_action=next_action,
            confidence=float(getattr(interp, "confidence", 0.0)),
            normalized_text=getattr(interp, "normalized_text", None),
        )
