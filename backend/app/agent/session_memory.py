"""
agent/session_memory.py

Conversation memory for RailMitra.

Goals:
- Track the last known route, train, fare search, booking, station, and user preference.
- Support follow-up queries like "that one", "what about AC?", "show fare for it".
- Keep the structure serializable so it can later be replaced with Redis / DB storage.
- Stay thread-safe for in-process demo usage.

This module does not perform persistence by itself.
It gives you a memory object you can keep per chat session and/or
serialize to your database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional


@dataclass
class ConversationMemory:
    session_id: str
    user_id: Optional[int] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    train_number: Optional[str] = None
    train_name: Optional[str] = None
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    travel_date: Optional[str] = None
    station: Optional[str] = None
    booking_id: Optional[str] = None
    last_intent: Optional[str] = None
    last_user_message: Optional[str] = None
    last_assistant_message: Optional[str] = None
    selected_train_number: Optional[str] = None
    selected_option_index: Optional[int] = None
    previous_results: Dict[str, Any] = field(default_factory=dict)
    previous_results_full: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    clarification_pending: bool = False
    clarification_question: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        # Tolerantly construct the dataclass from a dict that may contain extra keys
        from dataclasses import fields as _dc_fields
        allowed = {f.name for f in _dc_fields(cls)}
        init_kwargs = {k: v for k, v in (data or {}).items() if k in allowed}
        return cls(**init_kwargs)


class SessionMemoryStore:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str, user_id: Optional[int] = None) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationMemory(session_id=session_id, user_id=user_id)
            elif user_id is not None and self._sessions[session_id].user_id is None:
                self._sessions[session_id].user_id = user_id
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session_id: str, **kwargs: Any) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self._sessions.get(session_id)
            if memory is None:
                memory = ConversationMemory(session_id=session_id)
                self._sessions[session_id] = memory
            for key, value in kwargs.items():
                if value is None:
                    continue
                if hasattr(memory, key):
                    setattr(memory, key, value)
            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def merge_result(self, session_id: str, result: Dict[str, Any]) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self.get_or_create(session_id)
            if not isinstance(result, dict):
                return memory

            # Keep both compacted results (for quick context) and the full raw result
            memory.previous_results = self._compact_result(result)
            try:
                memory.previous_results_full = dict(result) if isinstance(result, dict) else {}
            except Exception:
                memory.previous_results_full = {}

            mapping = {
                "source": ["source", "source_station", "origin"],
                "destination": ["destination", "destination_station", "target"],
                "train_number": ["train_number"],
                "train_name": ["train_name"],
                "travel_class": ["travel_class", "class", "class_code"],
                "passengers": ["passengers", "passenger_count"],
                "travel_date": ["travel_date", "date"],
                "station": ["station", "station_code"],
                "booking_id": ["booking_id"],
                "selected_train_number": ["selected_train_number", "chosen_train_number"],
            }

            for memory_key, result_keys in mapping.items():
                value = self._first_present(result, result_keys)
                if value is not None:
                    setattr(memory, memory_key, value)

            if "source_resolved" in result and not memory.source:
                memory.source = self._string_or_none(result.get("source_resolved"))
            if "destination_resolved" in result and not memory.destination:
                memory.destination = self._string_or_none(result.get("destination_resolved"))

            if "selected_option_index" in result:
                try:
                    memory.selected_option_index = int(result["selected_option_index"])
                except Exception:
                    pass

            if "clarification_question" in result:
                memory.clarification_pending = True
                memory.clarification_question = self._string_or_none(result.get("clarification_question"))
            elif result.get("status") in {"confirmed", "ok", "success"}:
                memory.clarification_pending = False
                memory.clarification_question = None

            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def set_selection(self, session_id: str, train_number: Optional[str] = None, option_index: Optional[int] = None) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self.get_or_create(session_id)
            if train_number:
                memory.selected_train_number = train_number
                memory.train_number = train_number
            if option_index is not None:
                memory.selected_option_index = option_index
            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def set_clarification(self, session_id: str, question: str) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self.get_or_create(session_id)
            memory.clarification_pending = True
            memory.clarification_question = question
            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def clear_clarification(self, session_id: str) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self.get_or_create(session_id)
            memory.clarification_pending = False
            memory.clarification_question = None
            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def reset(self, session_id: str) -> None:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def as_context(self, session_id: str) -> Dict[str, Any]:
        memory = self.get(session_id)
        if memory is None:
            return {}
        return {
            "source": memory.source,
            "destination": memory.destination,
            "train_number": memory.train_number,
            "train_name": memory.train_name,
            "travel_class": memory.travel_class,
            "passengers": memory.passengers,
            "travel_date": memory.travel_date,
            "station": memory.station,
            "booking_id": memory.booking_id,
            "selected_train_number": memory.selected_train_number,
            "selected_option_index": memory.selected_option_index,
            "last_intent": memory.last_intent,
            "preferences": dict(memory.preferences) if isinstance(memory.preferences, dict) else {},
            "clarification_pending": memory.clarification_pending,
            "clarification_question": memory.clarification_question,
            "previous_results": dict(memory.previous_results) if isinstance(memory.previous_results, dict) else {},
            "previous_results_full": dict(memory.previous_results_full) if hasattr(memory, 'previous_results_full') and isinstance(memory.previous_results_full, dict) else {},
        }

    def build_memory_summary(self, session_id: str) -> str:
        memory = self.get(session_id)
        if memory is None:
            return "none"
        parts: List[str] = []
        if memory.source:
            parts.append(f"source={memory.source}")
        if memory.destination:
            parts.append(f"destination={memory.destination}")
        if memory.train_number:
            parts.append(f"train={memory.train_number}")
        if memory.travel_class:
            parts.append(f"class={memory.travel_class}")
        if memory.passengers:
            parts.append(f"passengers={memory.passengers}")
        if memory.travel_date:
            parts.append(f"date={memory.travel_date}")
        if memory.station:
            parts.append(f"station={memory.station}")
        if memory.booking_id:
            parts.append(f"booking={memory.booking_id}")
        if memory.last_intent:
            parts.append(f"intent={memory.last_intent}")
        if memory.selected_option_index is not None:
            parts.append(f"selected_option={memory.selected_option_index}")
        if memory.clarification_pending and memory.clarification_question:
            parts.append("clarification_pending=true")
        return "; ".join(parts) if parts else "none"

    def update_from_interpretation(self, session_id: str, interpretation: Dict[str, Any]) -> ConversationMemory:
        session_id = self._normalize_session_id(session_id)
        with self._lock:
            memory = self.get_or_create(session_id)
            slots = interpretation.get("slots", {}) if isinstance(interpretation, dict) else {}
            if isinstance(slots, dict):
                self.update(
                    session_id,
                    source=slots.get("source") or memory.source,
                    destination=slots.get("destination") or memory.destination,
                    train_number=slots.get("train_number") or memory.train_number,
                    travel_class=slots.get("travel_class") or memory.travel_class,
                    passengers=slots.get("passengers") or memory.passengers,
                    travel_date=slots.get("travel_date") or memory.travel_date,
                    station=slots.get("station") or memory.station,
                )
            if isinstance(interpretation, dict):
                if interpretation.get("intent"):
                    memory.last_intent = str(interpretation["intent"])
                if interpretation.get("clarification_needed"):
                    memory.clarification_pending = True
                    memory.clarification_question = self._string_or_none(interpretation.get("clarification_question"))
                else:
                    memory.clarification_pending = False
                    memory.clarification_question = None
            memory.updated_at = datetime.now(timezone.utc).isoformat()
            return memory

    def export(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {session_id: memory.to_dict() for session_id, memory in self._sessions.items()}

    def import_data(self, payload: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            self._sessions.clear()
            for session_id, data in (payload or {}).items():
                try:
                    self._sessions[self._normalize_session_id(session_id)] = ConversationMemory.from_dict(data)
                except Exception:
                    # Fallback: try to pick known keys only
                    try:
                        from dataclasses import fields as _dc_fields
                        allowed = {f.name for f in _dc_fields(ConversationMemory)}
                        filtered = {k: v for k, v in (data or {}).items() if k in allowed}
                        self._sessions[self._normalize_session_id(session_id)] = ConversationMemory(**filtered)
                    except Exception:
                        continue

    def _normalize_session_id(self, session_id: str) -> str:
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError("session_id cannot be empty")
        return sid

    def _compact_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        allowed_keys = {
            "status", "message", "source", "destination", "source_resolved", "destination_resolved",
            "train_number", "train_name", "class", "class_code", "travel_class",
            "passengers", "passenger_count", "travel_date", "booking_id", "count",
            "selected_train_number", "selected_option_index", "clarification_question",
        }
        return {k: v for k, v in result.items() if k in allowed_keys}

    def _first_present(self, data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for key in keys:
            if key in data and data[key] not in (None, "", [], {}):
                return data[key]
        return None

    def _string_or_none(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


default_session_memory_store = SessionMemoryStore()


def get_memory_context(session_id: str) -> Dict[str, Any]:
    return default_session_memory_store.as_context(session_id)


def summarize_memory(session_id: str) -> str:
    return default_session_memory_store.build_memory_summary(session_id)
