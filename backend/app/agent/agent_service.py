"""backend/app/agent/agent_service.py"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from sqlalchemy.orm import Session

from app.agent.query_understanding import QueryInterpretation, QueryUnderstanding
from app.agent.session_memory import SessionMemoryStore, default_session_memory_store
from app.agent.tools import AgentTools
from app.core.logger import logger
from app.services.recommendation_engine import RecommendationEngine

try:
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    BaseTool = Any  # type: ignore

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions"
)
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SUPPORTED_CLASSES = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]

# ---- Optimized defaults for free tier ----
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_TOOL_ITERATIONS = 2
DEFAULT_MAX_HISTORY_TURNS = 2
DEFAULT_MAX_OUTPUT_TOKENS = 250
DEFAULT_TEMPERATURE = 0.25
DEFAULT_RETRIES = 2

# ---- New LLM-first system prompt ----
SYSTEM_PROMPT = """You are RailMitra, an Indian Railways assistant.

You have these tools:
- search_trains(source, destination, date=None, time_hint=None, limit=5)
- get_fare(train_number, source, destination, travel_class="ALL", passengers=1)
- get_train_route(train_number)
- book_ticket(source, destination, travel_class, passengers, date=None, train_number=None)
- cancel_booking(booking_id)
- get_booking_history(user_id=1)
- get_station_info(station)

Instructions:
- For every user request, first decide which tool to call (if any) or if you need to ask a clarifying question.
- If the user's request is complete (e.g., "cancel booking 17"), output a JSON object: {"tool": "cancel_booking", "args": {"booking_id": 17}}.
- If the request is incomplete (e.g., "cancel my booking"), ask for the missing info: "Please provide the booking ID."
- If the user asks for a train search, extract source and destination. If missing, ask.
- Keep your responses concise and helpful.
- Do NOT invent any data; always use the tools.
- Only output the JSON for tool calls. For everything else, output plain text.

Examples:
- User: "cancel #17" → {"tool": "cancel_booking", "args": {"booking_id": 17}}
- User: "I want to cancel" → "Please provide the booking ID."
- User: "book 2 sleeper tickets from Delhi to Chennai tomorrow" → {"tool": "book_ticket", "args": {"source": "Delhi", "destination": "Chennai", "travel_class": "SL", "passengers": 2, "date": "2026-06-20"}}
- User: "show trains from Bangalore to Mumbai" → {"tool": "search_trains", "args": {"source": "Bangalore", "destination": "Mumbai"}}
"""


@dataclass
class ConversationContext:
    source: Optional[str] = None
    destination: Optional[str] = None
    train_number: Optional[str] = None
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    travel_date: Optional[str] = None
    time_hint: Optional[str] = None
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    booking_id: Optional[str] = None
    station: Optional[str] = None
    preference: Optional[str] = None
    intent: Optional[str] = None
    budget_max: Optional[int] = None
    selected_option_index: Optional[int] = None


@dataclass
class ParsedRequest:
    intent: str
    source: Optional[str] = None
    destination: Optional[str] = None
    train_number: Optional[str] = None
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    travel_date: Optional[str] = None
    time_hint: Optional[str] = None
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    booking_id: Optional[str] = None
    station: Optional[str] = None
    preference: Optional[str] = None
    budget_max: Optional[int] = None
    direct_only: bool = False
    selected_option_index: Optional[int] = None
    raw: str = ""


class AgentService:
    def __init__(self, session_store: Optional[SessionMemoryStore] = None) -> None:
        self.session_store = session_store or default_session_memory_store
        self.query_understanding = QueryUnderstanding()
        self.recommendation_engine = RecommendationEngine()
        self.hf_token = (
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HF_TOKEN")
            or ""
        )
        self.allow_llm = os.environ.get("RAILMITRA_ALLOW_LLM", "1").strip() not in {"0", "false", "False"}
        self.timeout = int(os.environ.get("HF_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        self.max_tool_iterations = int(os.environ.get("AGENT_MAX_TOOL_ITERATIONS", str(DEFAULT_MAX_TOOL_ITERATIONS)))
        self.max_history_turns = int(os.environ.get("AGENT_MAX_HISTORY_TURNS", str(DEFAULT_MAX_HISTORY_TURNS)))
        self.max_output_tokens = int(os.environ.get("HF_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))
        self.temperature = float(os.environ.get("HF_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.retries = int(os.environ.get("HF_RETRIES", str(DEFAULT_RETRIES)))

    def run(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        db: Session,
        session_id: str = "default",
        user_id: Optional[int] = None,
    ) -> str:
        cleaned_message = self._normalize_text(user_message)
        logger.info("[agent] incoming=%r", cleaned_message[:240])

        memory = self._get_or_create_memory(session_id, user_id)
        previous_result = self._memory_previous_result(memory)

        interpretation = self.query_understanding.interpret(
            cleaned_message,
            memory=self._memory_to_dict(memory),
            previous_result=previous_result,
        )
        self._update_memory_from_interpretation(session_id, interpretation, memory)
        context = self._build_context(conversation_history, interpretation, memory)
        parsed = self._parsed_from_interpretation(interpretation, memory, cleaned_message)

        tools = AgentTools(db)

        # ALWAYS try LLM first for all non-greeting intents
        if self.allow_llm and self.hf_token and parsed.intent != "greeting":
            llm_answer = self._run_tool_agent(
                cleaned_message, conversation_history, context, tools, parsed, session_id
            )
            if llm_answer:
                self._remember_last_turn(session_id, user_message, llm_answer)
                return llm_answer

        # Fallback to local handler if LLM fails or for greetings
        local_answer = self._handle_locally(parsed, tools, context, session_id, memory)
        if local_answer is not None:
            self._remember_last_turn(session_id, user_message, local_answer)
            return local_answer

        # Ultimate fallback
        answer = self._fallback_help_message()
        self._remember_last_turn(session_id, user_message, answer)
        return answer

    # ---------- Memory helpers ----------
    # (These are unchanged from the previous version)
    def _get_or_create_memory(self, session_id: str, user_id: Optional[int]) -> Any:
        store = self.session_store
        if hasattr(store, "get_or_create"):
            return store.get_or_create(session_id, user_id)
        if hasattr(store, "get"):
            mem = store.get(session_id)
            if mem is not None:
                return mem
        return type("Memory", (), {})()

    def _memory_to_dict(self, memory: Any) -> Dict[str, Any]:
        if isinstance(memory, dict):
            return memory
        if hasattr(memory, "to_dict"):
            try:
                d = memory.to_dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        out: Dict[str, Any] = {}
        for attr in (
            "source", "destination", "train_number", "train_name", "travel_class",
            "passengers", "travel_date", "station", "booking_id", "last_intent",
            "selected_train_number", "selected_option_index", "previous_results",
            "previous_results_full", "preferences", "slots", "budget_max",
            "time_hint", "departure_after", "departure_before",
        ):
            if hasattr(memory, attr):
                out[attr] = getattr(memory, attr)
        return out

    def _memory_previous_result(self, memory: Any) -> Dict[str, Any]:
        if isinstance(memory, dict):
            return memory.get("previous_results_full") or memory.get("previous_results") or {}
        for attr in ("previous_results_full", "previous_results"):
            if hasattr(memory, attr):
                val = getattr(memory, attr)
                if isinstance(val, dict):
                    return val
        return {}

    def _update_memory_from_interpretation(self, session_id: str, interpretation: QueryInterpretation, memory: Any) -> None:
        s = interpretation.slots
        payload = {
            "source": s.source,
            "destination": s.destination,
            "travel_date": s.travel_date,
            "travel_class": s.travel_class,
            "passengers": s.passengers,
            "station": s.station,
            "booking_id": s.booking_id,
            "last_intent": interpretation.intent,
            "selected_option_index": s.selected_option_index,
            "budget_max": s.budget_max,
            "time_hint": s.time_hint,
            "departure_after": s.departure_after,
            "departure_before": s.departure_before,
            "slots": {
                "source": s.source,
                "destination": s.destination,
                "travel_date": s.travel_date,
                "travel_class": s.travel_class,
                "passengers": s.passengers,
                "time_hint": s.time_hint,
                "departure_after": s.departure_after,
                "departure_before": s.departure_before,
                "budget_max": s.budget_max,
            },
        }
        try:
            if hasattr(self.session_store, "update_from_interpretation"):
                self.session_store.update_from_interpretation(session_id, interpretation.to_dict())
            elif hasattr(self.session_store, "update"):
                self.session_store.update(session_id, **payload)
        except Exception as exc:
            logger.warning("[agent] memory update failed: %s", exc)

    def _remember_last_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        try:
            if hasattr(self.session_store, "update"):
                self.session_store.update(session_id, last_user_message=user_message, last_assistant_message=assistant_message)
        except Exception:
            pass

    def _remember_selected_results(self, session_id: str, parsed: ParsedRequest, ranked: List[Dict[str, Any]], selected_index: int = 0) -> None:
        if not ranked:
            return
        selected_index = max(0, min(selected_index, len(ranked) - 1))
        selected = ranked[selected_index]
        train_number = selected.get("train_number") if isinstance(selected, dict) else getattr(selected, "train_number", None)
        train_name = selected.get("train_name") if isinstance(selected, dict) else getattr(selected, "train_name", None)
        payload = {
            "source": parsed.source,
            "destination": parsed.destination,
            "train_number": train_number or parsed.train_number,
            "train_name": train_name,
            "travel_class": parsed.travel_class,
            "passengers": parsed.passengers,
            "travel_date": parsed.travel_date,
            "last_intent": parsed.intent,
            "selected_train_number": train_number,
            "selected_option_index": selected_index,
            "previous_results": ranked[:10],
            "previous_results_full": {
                "intent": parsed.intent,
                "entities": {
                    "source": parsed.source,
                    "destination": parsed.destination,
                    "date": parsed.travel_date,
                    "travel_class": parsed.travel_class,
                    "passengers": parsed.passengers,
                },
                "results": ranked[:10],
                "selected_train_number": train_number,
                "selected_option_index": selected_index,
            },
            "slots": {
                "source": parsed.source,
                "destination": parsed.destination,
                "travel_date": parsed.travel_date,
                "travel_class": parsed.travel_class,
                "passengers": parsed.passengers,
                "time_hint": parsed.time_hint,
                "departure_after": parsed.departure_after,
                "departure_before": parsed.departure_before,
                "budget_max": parsed.budget_max,
            },
        }
        try:
            if hasattr(self.session_store, "merge_result"):
                self.session_store.merge_result(session_id, payload)
            elif hasattr(self.session_store, "update"):
                self.session_store.update(session_id, **payload)
        except Exception as exc:
            logger.warning("[agent] remember-selected failed: %s", exc)

    def _remember_context(self, session_id: str, parsed: ParsedRequest) -> None:
        payload = {
            "source": parsed.source,
            "destination": parsed.destination,
            "travel_class": parsed.travel_class,
            "passengers": parsed.passengers,
            "travel_date": parsed.travel_date,
            "last_intent": parsed.intent,
            "time_hint": parsed.time_hint,
            "departure_after": parsed.departure_after,
            "departure_before": parsed.departure_before,
            "budget_max": parsed.budget_max,
            "slots": {
                "source": parsed.source,
                "destination": parsed.destination,
                "travel_date": parsed.travel_date,
                "travel_class": parsed.travel_class,
                "passengers": parsed.passengers,
                "time_hint": parsed.time_hint,
                "departure_after": parsed.departure_after,
                "departure_before": parsed.departure_before,
                "budget_max": parsed.budget_max,
            },
        }
        try:
            if hasattr(self.session_store, "update"):
                self.session_store.update(session_id, **payload)
        except Exception as exc:
            logger.warning("[agent] remember-context failed: %s", exc)

    def _build_context(self, conversation_history: List[Dict[str, str]], interpretation: QueryInterpretation, memory: Any) -> ConversationContext:
        slots = getattr(interpretation, "slots", None) or None
        def mem_attr(name: str):
            try:
                return getattr(memory, name)
            except Exception:
                return None
        return ConversationContext(
            source = getattr(slots, 'source', None) or mem_attr('source'),
            destination = getattr(slots, 'destination', None) or mem_attr('destination'),
            train_number = getattr(slots, 'train_number', None) or mem_attr('train_number'),
            travel_class = getattr(slots, 'travel_class', None) or mem_attr('travel_class'),
            passengers = getattr(slots, 'passengers', None) or mem_attr('passengers'),
            travel_date = getattr(slots, 'travel_date', None) or mem_attr('travel_date'),
            time_hint = getattr(slots, 'time_hint', None) or mem_attr('time_hint'),
            departure_after = getattr(slots, 'departure_after', None) or mem_attr('departure_after'),
            departure_before = getattr(slots, 'departure_before', None) or mem_attr('departure_before'),
            sort_by = getattr(slots, 'sort_by', None) or mem_attr('sort_by'),
            limit = getattr(slots, 'limit', None) or mem_attr('limit'),
            booking_id = getattr(slots, 'booking_id', None) or mem_attr('booking_id'),
            station = getattr(slots, 'station', None) or mem_attr('station'),
            preference = getattr(slots, 'preference', None) or mem_attr('preference'),
            intent = getattr(interpretation, 'intent', None),
            budget_max = getattr(slots, 'budget_max', None) or mem_attr('budget_max'),
            selected_option_index = getattr(slots, 'selected_option_index', None) or mem_attr('selected_option_index'),
        )

    def _parsed_from_interpretation(self, interpretation: QueryInterpretation, memory: Any, raw_text: str) -> ParsedRequest:
        slots = getattr(interpretation, "slots", None)
        mem = self._memory_to_dict(memory)
        direct_only = False
        try:
            direct_only = "direct_only" in (interpretation.sub_intents or [])
        except Exception:
            direct_only = False
        return ParsedRequest(
            intent=getattr(interpretation, "intent", "train_search"),
            source=getattr(slots, "source", None) or mem.get("source"),
            destination=getattr(slots, "destination", None) or mem.get("destination"),
            train_number=getattr(slots, "train_number", None) or mem.get("train_number"),
            travel_class=getattr(slots, "travel_class", None) or mem.get("travel_class"),
            passengers=getattr(slots, "passengers", None) or mem.get("passengers"),
            travel_date=getattr(slots, "travel_date", None) or mem.get("travel_date"),
            time_hint=getattr(slots, "time_hint", None) or mem.get("time_hint"),
            departure_after=getattr(slots, "departure_after", None) or mem.get("departure_after"),
            departure_before=getattr(slots, "departure_before", None) or mem.get("departure_before"),
            sort_by=getattr(slots, "sort_by", None) or mem.get("sort_by"),
            limit=getattr(slots, "limit", None) or mem.get("limit"),
            booking_id=getattr(slots, "booking_id", None) or mem.get("booking_id"),
            station=getattr(slots, "station", None) or mem.get("station"),
            preference=getattr(slots, "preference", None) or mem.get("preference"),
            budget_max=getattr(slots, "budget_max", None) or mem.get("budget_max"),
            direct_only=bool(direct_only),
            selected_option_index=getattr(slots, "selected_option_index", None) or mem.get("selected_option_index"),
            raw=raw_text or getattr(interpretation, "raw_text", ""),
        )

    # ---------- Local handlers (fallback) ----------
    # Kept only for when LLM fails or for greetings
    def _handle_locally(
        self,
        parsed: ParsedRequest,
        tools: AgentTools,
        context: ConversationContext,
        session_id: str,
        memory: Any,
    ) -> Optional[str]:
        intent = parsed.intent
        self._remember_context(session_id, parsed)

        if intent == "greeting":
            return self._greeting_message()

        if intent == "booking_history":
            result = self._safe_tool_json(self._invoke_compat(tools.get_booking_history))
            if result.get("status") == "empty":
                return "You do not have any bookings yet. Search for trains to get started."
            if result.get("status") == "error":
                return self._friendly_tool_error("booking history", result)
            return self._format_booking_history(result)

        if intent == "booking_cancel":
            booking_id = parsed.booking_id
            if not booking_id:
                # Fallback extraction
                booking_id = self._extract_booking_id_fallback(parsed.raw)
                if booking_id:
                    parsed.booking_id = booking_id
                    self._remember_context(session_id, parsed)
            if not booking_id:
                return "Please share the booking ID so I can cancel it."
            result = self._safe_tool_json(self._invoke_compat(tools.cancel_booking, booking_id=booking_id))
            if result.get("status") == "error":
                return self._friendly_tool_error("cancellation", result)
            return result.get("message", f"Booking #{booking_id} has been processed.")

        # For other intents, we keep the previous fallback logic
        # (but they will rarely be used because LLM handles them)
        return self._fallback_help_message()

    def _extract_booking_id_fallback(self, text: str) -> Optional[str]:
        m = re.search(r'\b(?:cancel|#|id)\s*#?\s*(\d+)\b', text, re.IGNORECASE)
        if m:
            return m.group(1)
        if re.search(r'\bcancel\b', text, re.IGNORECASE):
            m = re.search(r'\b(\d+)\b', text)
            if m:
                return m.group(1)
        return None

    # ---------- Formatting helpers ----------
    # (Kept the same as before)
    def _format_booking_history(self, result: Dict[str, Any]) -> str:
        bookings = result.get("bookings", [])
        lines = [f"📋 **Your Bookings ({result.get('count', len(bookings))} total):**\n"]
        for booking in bookings:
            icon = "✅" if str(booking.get("status", "")).upper() == "CONFIRMED" else "❌"
            lines.append(
                f"{icon} **#{booking.get('booking_id', '-') }** – Train {booking.get('train_number', '-') } | "
                f"{booking.get('class', '-') } × {booking.get('passengers', '-') } | "
                f"{str(booking.get('travel_date', ''))[:10]} | {booking.get('status', '-') }"
            )
        return "\n".join(lines)

    def _friendly_tool_error(self, action: str, payload: Dict[str, Any]) -> str:
        msg = payload.get("message") or payload.get("error") or "unknown error"
        return f"❌ Could not complete {action}: {msg}"

    def _greeting_message(self) -> str:
        return (
            "👋 Hello! I’m **RailMitra**. I can help you search trains, estimate fares, view routes, and handle demo bookings.\n\n"
            "Try asking:\n"
            "• Show trains from Bangalore to Mangalore after 8 PM\n"
            "• What is the sleeper fare for train 16585?\n"
            "• Tell me the route of train 12627"
        )

    def _fallback_help_message(self) -> str:
        return (
            "I can help with train search, fares, routes, station info, and demo bookings.\n\n"
            "Examples:\n"
            "• Show trains from Bangalore to Mangalore after 8 PM\n"
            "• Find the sleeper fare for train 16585\n"
            "• Tell me the route of train 12627\n"
            "• Book 2 sleeper seats from Bangalore to Mangalore tomorrow"
        )

    # ---------- Other helper methods ----------
    # (We keep the existing ones for compatibility, but many are unused now)
    def _safe_tool_json(self, raw: Any) -> Dict[str, Any]:
        parsed = self._safe_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"status": "ok", "result": parsed}
        return {"status": "ok", "result": parsed}

    def _safe_json_loads(self, raw: Any) -> Any:
        if raw is None:
            return {}
        if isinstance(raw, (dict, list, int, float, bool)):
            return raw
        text = str(raw).strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return {"status": "error", "message": text}

    def _invoke_compat(self, fn: Any, **kwargs: Any) -> Any:
        try:
            sig = inspect.signature(fn)
            allowed = {k: v for k, v in kwargs.items() if k in sig.parameters and v not in (None, "", [], {})}
            return fn(**allowed)
        except Exception:
            try:
                filtered = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
                return fn(**filtered)
            except TypeError:
                if "source" in kwargs and "destination" in kwargs and "date" in kwargs:
                    return fn(kwargs["source"], kwargs["destination"], kwargs.get("date", ""))
                raise

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _sanitize_output(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"hf_[A-Za-z0-9]{20,}", "[redacted]", cleaned)
        return cleaned

    def _context_summary(self, context: ConversationContext) -> str:
        parts = []
        for key in ("source", "destination", "train_number", "travel_class", "passengers", "travel_date", "time_hint", "departure_after", "departure_before", "preference", "intent", "budget_max", "selected_option_index"):
            value = getattr(context, key, None)
            if value not in (None, "", []):
                parts.append(f"{key}={value}")
        return "; ".join(parts) if parts else "none"

    # ---------- LLM agent loop ----------
    def _build_messages(self, user_message: str, history: List[Dict[str, str]], context: ConversationContext) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if any([context.source, context.destination, context.train_number, context.travel_class, context.time_hint]):
            messages.append({"role": "system", "content": f"Context: {self._context_summary(context)}"})

        # Keep only the last user and assistant messages (for cost)
        last_user = None
        last_assistant = None
        for item in reversed(history):
            role = item.get("role")
            if role == "user" and last_user is None:
                last_user = item
            elif role == "assistant" and last_assistant is None:
                last_assistant = item
            if last_user and last_assistant:
                break
        if last_user:
            messages.append(last_user)
        if last_assistant:
            messages.append(last_assistant)

        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_tool_schemas(self, tools: Sequence[Any]) -> List[Dict[str, Any]]:
        return []  # Not needed; we use a prompt-based approach

    def _get_tool_args_schema(self, tool: Any) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def _build_tool_map(self, tools: Sequence[Any]) -> Dict[str, Any]:
        return {getattr(tool, "name", f"tool_{i}"): tool for i, tool in enumerate(tools)}

    def _invoke_tool(self, tool: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        raw = tool.invoke(args)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            parsed = self._safe_json_loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"status": "ok", "result": raw}
        return {"status": "ok", "result": raw}

    def _parse_tool_call(self, text: str, tool_map: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        # First, try to find JSON in code block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data and "args" in data and isinstance(data["args"], dict):
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        json_match = re.search(r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data and "args" in data and isinstance(data["args"], dict):
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text as JSON
        try:
            data = json.loads(text.strip())
            if "tool" in data and "args" in data and isinstance(data["args"], dict):
                return data["tool"], data["args"]
        except json.JSONDecodeError:
            pass

        return None

    def _run_tool_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        context: ConversationContext,
        tools: AgentTools,
        parsed: ParsedRequest,
        session_id: str,
    ) -> Optional[str]:
        if not self.allow_llm or not self.hf_token:
            return None

        try:
            tool_objects = tools.build()
            tool_map = self._build_tool_map(tool_objects)

            messages = self._build_messages(user_message, conversation_history, context)

            for iteration in range(self.max_tool_iterations):
                payload = {
                    "model": HF_MODEL_NAME,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_output_tokens,
                    "stream": False,
                }

                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json",
                }

                logger.info("[agent] LLM request (iter %d): %s", iteration + 1, messages[-1]["content"][:100])

                response = requests.post(HF_API_URL, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code != 200:
                    logger.error("[agent] HF API error: %s %s", response.status_code, response.text)
                    return None

                data = response.json()
                assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not assistant_message:
                    logger.error("[agent] Empty response from HF")
                    return None

                logger.info("[agent] LLM response: %s", assistant_message[:200])

                # Check if LLM wants to call a tool
                tool_call = self._parse_tool_call(assistant_message, tool_map)
                if tool_call:
                    tool_name, tool_args = tool_call
                    logger.info("[agent] Tool call: %s with args: %s", tool_name, tool_args)

                    tool = tool_map.get(tool_name)
                    if not tool:
                        messages.append({"role": "assistant", "content": assistant_message})
                        messages.append({"role": "user", "content": f"Error: Tool '{tool_name}' not found. Please use one of: {list(tool_map.keys())}."})
                        continue

                    result = self._invoke_tool(tool, tool_args)
                    logger.info("[agent] Tool result: %s", str(result)[:200])

                    # Store in memory if it's a train search
                    if tool_name == "search_trains" and result.get("status") == "ok":
                        trains = result.get("trains", [])
                        if trains:
                            self._remember_selected_results(session_id, parsed, trains, selected_index=0)

                    # Add assistant's JSON and tool result to conversation
                    messages.append({"role": "assistant", "content": assistant_message})
                    messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, ensure_ascii=False)}"})
                    continue
                else:
                    # No tool call – this is the final answer (could be clarification or direct response)
                    return assistant_message

            return "I'm sorry, I couldn't complete your request in the allowed number of steps."

        except requests.exceptions.Timeout:
            logger.error("[agent] HF API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error("[agent] HF API request error: %s", e)
            return None
        except json.JSONDecodeError as e:
            logger.error("[agent] Failed to parse HF response: %s", e)
            return None
        except Exception as e:
            logger.exception("[agent] Unexpected error in _run_tool_agent: %s", e)
            return None


def build_agent_service() -> AgentService:
    return AgentService()
