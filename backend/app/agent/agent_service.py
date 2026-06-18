"""backend/app/agent/agent_service.py"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
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

# ---------- FORCE IPv4 (fixes DNS on Render) ----------
import requests.packages.urllib3.util.connection as urllib3_cn

def _allowed_gateways():
    return socket.AF_INET

urllib3_cn.allowed_gateways = _allowed_gateways
# -----------------------------------------------------

# Try to import huggingface_hub; if not installed, fall back to requests
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False
    InferenceClient = None

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions"
)
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SUPPORTED_CLASSES = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]
BOOKING_REQUIRED_FIELDS = (
    "source",
    "destination",
    "travel_date",
    "travel_class",
    "passenger_count",
    "train_number",
)

# ---- Optimized defaults for free tier ----
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_TOOL_ITERATIONS = 2
DEFAULT_MAX_HISTORY_TURNS = 2
DEFAULT_MAX_OUTPUT_TOKENS = 250
DEFAULT_TEMPERATURE = 0.25
DEFAULT_RETRIES = 2

# ---- LLM system prompt ----
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
- For booking requests in chat, do not create the booking directly. Ask the user to review the booking form.
- If the user's request is complete, output a JSON object: {"tool": "tool_name", "args": {...}}.
- If the request is incomplete, ask for the missing info in plain text.
- Keep responses concise.
- Only output JSON for tool calls. For everything else, output plain text.

Examples:
- User: "cancel #17" → {"tool": "cancel_booking", "args": {"booking_id": 17}}
- User: "I want to cancel" → "Please provide the booking ID."
- User: "book 2 sleeper tickets from Delhi to Chennai tomorrow" → {"tool": "book_ticket", "args": {"source": "Delhi", "destination": "Chennai", "travel_class": "SL", "passengers": 2, "date": "2026-06-20"}}
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


@dataclass
class AgentRunResult:
    answer: str
    action: Optional[str] = None
    booking_draft: Optional[Dict[str, Any]] = None
    missing_required_fields: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


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
        return self.run_structured(
            user_message=user_message,
            conversation_history=conversation_history,
            db=db,
            session_id=session_id,
            user_id=user_id,
        ).answer

    def run_structured(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        db: Session,
        session_id: str = "default",
        user_id: Optional[int] = None,
    ) -> AgentRunResult:
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

        context = self._build_context(conversation_history, interpretation, memory, cleaned_message, previous_result)
        parsed = self._parsed_from_interpretation(interpretation, memory, cleaned_message, previous_result)

        tools = AgentTools(db)
        diagnostics: Dict[str, Any] = {
            "intent": parsed.intent,
            "route": "unknown",
            "llm_attempted": False,
            "llm_used": False,
            "local_handler_used": False,
            "fallback_used": False,
            "llm_error": None,
            "local_error": None,
        }

        # Booking intent is draft-only in chat. Final creation must happen from
        # a validated booking form/submit endpoint.
        if parsed.intent == "booking_create":
            result = self._handle_booking_draft(parsed, context, session_id, memory, diagnostics)
            self._remember_last_turn(session_id, user_message, result.answer)
            self._log_run_diagnostics(session_id, result.diagnostics)
            return result

        # If the request is incomplete, keep it local so we preserve the clarification flow.
        if interpretation.clarification_needed:
            local_answer = self._call_local_handler(parsed, tools, context, session_id, memory, diagnostics)
            if local_answer is not None:
                diagnostics["route"] = "local_handler"
                diagnostics["local_handler_used"] = True
                self._remember_last_turn(session_id, user_message, local_answer)
                self._log_run_diagnostics(session_id, diagnostics)
                return AgentRunResult(answer=local_answer, diagnostics=diagnostics)

        # Only let the LLM handle clear, non-follow-up requests.
        llm_allowed = (
            self.allow_llm
            and self.hf_token
            and parsed.intent != "greeting"
            and not self._should_bypass_llm(cleaned_message, interpretation, memory, previous_result)
        )
        if llm_allowed:
            diagnostics["llm_attempted"] = True
            llm_answer = self._run_tool_agent(
                cleaned_message, conversation_history, context, tools, parsed, session_id
            )
            if llm_answer:
                diagnostics["route"] = "llm"
                diagnostics["llm_used"] = True
                self._remember_last_turn(session_id, user_message, llm_answer)
                self._log_run_diagnostics(session_id, diagnostics)
                return AgentRunResult(answer=llm_answer, diagnostics=diagnostics)
            diagnostics["llm_error"] = "no_response_or_error"
            diagnostics["fallback_used"] = True
            logger.warning("[agent] LLM unavailable or empty; falling back to local handler")
        else:
            if not self.allow_llm:
                diagnostics["llm_error"] = "disabled"
            elif not self.hf_token:
                diagnostics["llm_error"] = "missing_token"
            elif parsed.intent == "greeting":
                diagnostics["llm_error"] = "bypassed_for_greeting"
            else:
                diagnostics["llm_error"] = "bypassed_for_context"

        # Fallback to full local handler if LLM fails
        local_answer = self._call_local_handler(parsed, tools, context, session_id, memory, diagnostics)
        if local_answer is not None:
            diagnostics["route"] = "local_after_llm" if diagnostics["llm_attempted"] else "local_handler"
            diagnostics["local_handler_used"] = True
            self._remember_last_turn(session_id, user_message, local_answer)
            self._log_run_diagnostics(session_id, diagnostics)
            return AgentRunResult(answer=local_answer, diagnostics=diagnostics)

        # Ultimate fallback
        answer = self._fallback_help_message()
        diagnostics["route"] = "fallback_help"
        diagnostics["fallback_used"] = True
        self._remember_last_turn(session_id, user_message, answer)
        self._log_run_diagnostics(session_id, diagnostics)
        return AgentRunResult(answer=answer, diagnostics=diagnostics)

    def _call_local_handler(
        self,
        parsed: ParsedRequest,
        tools: AgentTools,
        context: ConversationContext,
        session_id: str,
        memory: Any,
        diagnostics: Dict[str, Any],
    ) -> Optional[str]:
        try:
            return self._handle_locally_full(parsed, tools, context, session_id, memory)
        except Exception as exc:
            diagnostics["local_error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("[agent] Local handler failed: %s", exc)
            return None

    def _log_run_diagnostics(self, session_id: str, diagnostics: Dict[str, Any]) -> None:
        logger.info(
            "[agent] session=%s route=%s intent=%s llm_attempted=%s llm_used=%s local_used=%s fallback=%s llm_error=%s local_error=%s",
            session_id,
            diagnostics.get("route"),
            diagnostics.get("intent"),
            diagnostics.get("llm_attempted"),
            diagnostics.get("llm_used"),
            diagnostics.get("local_handler_used"),
            diagnostics.get("fallback_used"),
            diagnostics.get("llm_error"),
            diagnostics.get("local_error"),
        )

    # ---------- Memory helpers ----------
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

    def _should_reuse_memory_context(
        self,
        raw_text: str,
        memory: Any,
        previous_result: Optional[Dict[str, Any]],
    ) -> bool:
        text = self._normalize_text(raw_text)
        if not text:
            return False

        if self._has_fresh_route_request(text):
            return False

        if getattr(memory, "clarification_pending", False):
            return True

        follow_up_markers = (
            "what about", "that one", "this one", "first one", "second one", "third one",
            "book it", "book the first", "book the second", "cheapest one", "fastest one",
            "fare for it", "route for it", "show fare", "show route", "cancel it",
        )
        if any(marker in text for marker in follow_up_markers):
            return True

        if previous_result:
            # Short replies after a search/clarification are usually follow-ups.
            if len(text.split()) <= 4 and re.search(r"\b(first|second|third|that|it|one)\b", text):
                return True

        return False

    def _has_fresh_route_request(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if re.search(r"\bfrom\s+\S.+\bto\s+\S", normalized):
            return True
        if re.search(r"\bbetween\s+\S.+\band\s+\S", normalized):
            return True
        try:
            src, dst = self.query_understanding._extract_stations(normalized)
            if src or dst:
                return True
        except Exception:
            pass
        return False

    def _should_bypass_llm(
        self,
        raw_text: str,
        interpretation: QueryInterpretation,
        memory: Any,
        previous_result: Dict[str, Any],
    ) -> bool:
        if interpretation.clarification_needed:
            return True
        if getattr(memory, "clarification_pending", False):
            return True
        if self._should_reuse_memory_context(raw_text, memory, previous_result):
            return True
        # Very short messages are usually fragments like "delhi" / "tomorrow" / "3a"
        if len(self._normalize_text(raw_text).split()) <= 2:
            return True
        return False

    def _build_context(
        self,
        conversation_history: List[Dict[str, str]],
        interpretation: QueryInterpretation,
        memory: Any,
        raw_text: str,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> ConversationContext:
        slots = getattr(interpretation, "slots", None) or None

        def mem_attr(name: str):
            try:
                return getattr(memory, name)
            except Exception:
                return None

        use_memory = self._should_reuse_memory_context(raw_text, memory, previous_result)
        return ConversationContext(
            source=getattr(slots, "source", None) or (mem_attr("source") if use_memory else None),
            destination=getattr(slots, "destination", None) or (mem_attr("destination") if use_memory else None),
            train_number=getattr(slots, "train_number", None) or (mem_attr("train_number") if use_memory else None),
            travel_class=getattr(slots, "travel_class", None) or (mem_attr("travel_class") if use_memory else None),
            passengers=getattr(slots, "passengers", None) or (mem_attr("passengers") if use_memory else None),
            travel_date=getattr(slots, "travel_date", None) or (mem_attr("travel_date") if use_memory else None),
            time_hint=getattr(slots, "time_hint", None) or (mem_attr("time_hint") if use_memory else None),
            departure_after=getattr(slots, "departure_after", None) or (mem_attr("departure_after") if use_memory else None),
            departure_before=getattr(slots, "departure_before", None) or (mem_attr("departure_before") if use_memory else None),
            sort_by=getattr(slots, "sort_by", None) or (mem_attr("sort_by") if use_memory else None),
            limit=getattr(slots, "limit", None) or (mem_attr("limit") if use_memory else None),
            booking_id=getattr(slots, "booking_id", None) or (mem_attr("booking_id") if use_memory else None),
            station=getattr(slots, "station", None) or (mem_attr("station") if use_memory else None),
            preference=getattr(slots, "preference", None) or (mem_attr("preference") if use_memory else None),
            intent=getattr(interpretation, "intent", None),
            budget_max=getattr(slots, "budget_max", None) or (mem_attr("budget_max") if use_memory else None),
            selected_option_index=getattr(slots, "selected_option_index", None) or (mem_attr("selected_option_index") if use_memory else None),
        )

    def _parsed_from_interpretation(
        self,
        interpretation: QueryInterpretation,
        memory: Any,
        raw_text: str,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> ParsedRequest:
        slots = getattr(interpretation, "slots", None)
        mem = self._memory_to_dict(memory)
        direct_only = "direct_only" in (interpretation.sub_intents or [])
        use_memory = self._should_reuse_memory_context(raw_text, memory, previous_result)
        return ParsedRequest(
            intent=getattr(interpretation, "intent", "train_search"),
            source=getattr(slots, "source", None) or (mem.get("source") if use_memory else None),
            destination=getattr(slots, "destination", None) or (mem.get("destination") if use_memory else None),
            train_number=getattr(slots, "train_number", None) or (mem.get("train_number") if use_memory else None),
            travel_class=getattr(slots, "travel_class", None) or (mem.get("travel_class") if use_memory else None),
            passengers=getattr(slots, "passengers", None) or (mem.get("passengers") if use_memory else None),
            travel_date=getattr(slots, "travel_date", None) or (mem.get("travel_date") if use_memory else None),
            time_hint=getattr(slots, "time_hint", None) or (mem.get("time_hint") if use_memory else None),
            departure_after=getattr(slots, "departure_after", None) or (mem.get("departure_after") if use_memory else None),
            departure_before=getattr(slots, "departure_before", None) or (mem.get("departure_before") if use_memory else None),
            sort_by=getattr(slots, "sort_by", None) or (mem.get("sort_by") if use_memory else None),
            limit=getattr(slots, "limit", None) or (mem.get("limit") if use_memory else None),
            booking_id=getattr(slots, "booking_id", None) or (mem.get("booking_id") if use_memory else None),
            station=getattr(slots, "station", None) or (mem.get("station") if use_memory else None),
            preference=getattr(slots, "preference", None) or (mem.get("preference") if use_memory else None),
            budget_max=getattr(slots, "budget_max", None) or (mem.get("budget_max") if use_memory else None),
            direct_only=bool(direct_only),
            selected_option_index=getattr(slots, "selected_option_index", None) or (mem.get("selected_option_index") if use_memory else None),
            raw=raw_text or getattr(interpretation, "raw_text", ""),
        )

    # ---------- FULL LOCAL HANDLER ----------
    def _handle_locally_full(
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

        if intent == "station_query":
            station_query = parsed.station or parsed.source or parsed.destination or self._guess_station_from_text(parsed.raw)
            if not station_query:
                return "Please provide a station name or code."
            result = self._safe_tool_json(self._invoke_compat(tools.get_station_info, station=station_query))
            if result.get("status") != "ok":
                return result.get("message", "Station not found.")
            return self._format_station_info(result)

        if intent == "train_info":
            if not parsed.train_number:
                return self._clarify_train_number()
            result = self._safe_tool_json(self._invoke_compat(tools.get_train_route, train_number=parsed.train_number))
            if result.get("status") != "ok":
                return result.get("message", "Route not found.")
            return self._format_route(result, parsed.train_number)

        if intent in {"route_query", "fare_query", "train_search", "multi_intent", "booking_create"}:
            src = parsed.source or context.source
            dst = parsed.destination or context.destination

            if not (src and dst):
                prev = self._memory_previous_result(memory)
                if self._is_contextual_follow_up(parsed.raw) and prev:
                    route_ctx = self._previous_route_context(prev)
                    src = src or route_ctx.get("source")
                    dst = dst or route_ctx.get("destination")
                    parsed.train_number = parsed.train_number or route_ctx.get("train_number")
                    parsed.travel_class = parsed.travel_class or route_ctx.get("travel_class")
                    parsed.passengers = parsed.passengers or route_ctx.get("passengers")
                    parsed.travel_date = parsed.travel_date or route_ctx.get("travel_date")
                if not (src and dst):
                    if intent in {"fare_query", "multi_intent"} and not prev:
                        return "Please tell me the source and destination first so I can help with that."
                    return self._clarify_missing_route(parsed.raw)

            parsed.source = src
            parsed.destination = dst

            if intent == "fare_query":
                return self._handle_fare_query(parsed, tools, context, session_id, memory)
            if intent == "booking_create":
                return self._handle_booking_query(parsed, tools, context, session_id, memory)

            trains = self._search_trains(parsed, tools)
            if not trains:
                return f"😔 No trains found from **{src}** to **{dst}**. Please check the station names or try nearby stations."

            ranked = self._rank_trains(trains, parsed, src, dst)
            self._remember_selected_results(session_id, parsed, ranked, selected_index=parsed.selected_option_index or 0)

            if intent == "multi_intent":
                return self._format_train_search(src, dst, ranked, parsed, multi=True)
            return self._format_train_search(src, dst, ranked, parsed)

        if intent == "booking_modify":
            return self._booking_modify_message()

        # Fallback: if source & destination are present, try to search
        if parsed.source and parsed.destination:
            trains = self._search_trains(parsed, tools)
            if trains:
                ranked = self._rank_trains(trains, parsed, parsed.source, parsed.destination)
                self._remember_selected_results(session_id, parsed, ranked, selected_index=parsed.selected_option_index or 0)
                return self._format_train_search(parsed.source, parsed.destination, ranked, parsed)

        # Handle "cheapest", "fastest" etc.
        if any(w in parsed.raw.lower() for w in ("cheapest", "fastest", "best balance")):
            prev = self._memory_previous_result(memory)
            if not prev:
                return "Please first ask me to show trains for a route, then I can compare the options."
            src = prev.get("entities", {}).get("source") or context.source
            dst = prev.get("entities", {}).get("destination") or context.destination
            if not (src and dst):
                return "Please first share the route so I can compare trains."
            ranked = prev.get("results") or prev.get("trains") or []
            if ranked:
                ranked = self._rank_trains(ranked, parsed, src, dst)
                return self._format_train_search(src, dst, ranked, parsed)
            return "Please first ask for train options so I can compare them."

        return None

    # ---------- Helper methods for local fallback ----------
    def _handle_fare_query(self, parsed: ParsedRequest, tools: AgentTools, context: ConversationContext, session_id: str, memory: Any) -> Optional[str]:
        src = parsed.source or context.source
        dst = parsed.destination or context.destination
        pax = parsed.passengers or context.passengers or 1
        travel_class = parsed.travel_class or context.travel_class
        train_number = parsed.train_number or context.train_number

        if not src or not dst:
            prev = self._memory_previous_result(memory)
            if self._is_contextual_follow_up(parsed.raw) and prev:
                route_ctx = self._previous_route_context(prev)
                src = src or route_ctx.get("source")
                dst = dst or route_ctx.get("destination")
                train_number = train_number or route_ctx.get("train_number")
                travel_class = travel_class or route_ctx.get("travel_class")
                pax = pax or route_ctx.get("passengers") or 1
            if not (src and dst):
                return "Please tell me the source and destination first so I can estimate fare."

        trains = self._search_trains(parsed, tools) if not train_number else []
        if not train_number and trains:
            ranked = self._rank_trains(trains, parsed, src, dst)
            if ranked:
                self._remember_selected_results(session_id, parsed, ranked, selected_index=0)
                train_number = ranked[0].get("train_number")
            if not train_number and trains:
                first_train = trains[0]
                train_number = first_train.get("train_number") if isinstance(first_train, dict) else getattr(first_train, "train_number", None)

        if not train_number:
            fare_est = self._estimate_fare_by_distance(src, dst, travel_class or "SL", pax)
            if fare_est:
                return (
                    f"💰 **Approximate Fare Estimate** ({src} → {dst})\n"
                    f"Class: **{travel_class or 'SL'}**\n"
                    f"Per passenger: **₹{fare_est['per_passenger']:,.0f}**\n"
                    f"Total ({pax} pax): **₹{fare_est['total']:,.0f}**\n"
                    "_Estimated based on corridor distance (no direct train found)._"
                )
            return f"I could not find a suitable train from **{src}** to **{dst}** for fare estimation. Try using station codes or nearby cities."

        fare = self._safe_tool_json(
            self._invoke_compat(
                tools.get_fare,
                train_number=train_number,
                source=src,
                destination=dst,
                travel_class=travel_class or "ALL",
                passengers=pax,
                travel_date=parsed.travel_date or context.travel_date or "",
                departure_after=parsed.departure_after or "",
                departure_before=parsed.departure_before or "",
                time_hint=parsed.time_hint or "",
            )
        )
        if fare.get("status") == "error":
            return self._friendly_tool_error("fare lookup", fare)
        if travel_class and travel_class != "ALL":
            return self._format_single_fare(fare, src, dst, pax)
        return self._format_fare_table(fare, src, dst, pax)

    def _handle_booking_query(self, parsed: ParsedRequest, tools: AgentTools, context: ConversationContext, session_id: str, memory: Any) -> Optional[str]:
        draft = self._build_booking_draft(parsed, context, memory)
        self._remember_context(session_id, parsed)
        logger.info(
            "[agent] booking_create handled as draft action=open_booking_form missing=%s",
            draft.get("missing_required_fields", []),
        )
        return self._format_booking_draft_message(draft)

    def _handle_booking_draft(
        self,
        parsed: ParsedRequest,
        context: ConversationContext,
        session_id: str,
        memory: Any,
        diagnostics: Dict[str, Any],
    ) -> AgentRunResult:
        draft = self._build_booking_draft(parsed, context, memory)
        answer = self._format_booking_draft_message(draft)
        self._remember_context(session_id, parsed)
        diagnostics["route"] = "booking_draft"
        diagnostics["local_handler_used"] = True
        logger.info(
            "[agent] booking draft prepared session=%s missing=%s ready=%s",
            session_id,
            draft.get("missing_required_fields", []),
            draft.get("ready_for_submit"),
        )
        return AgentRunResult(
            answer=answer,
            action="open_booking_form",
            booking_draft=draft,
            missing_required_fields=list(draft.get("missing_required_fields", [])),
            diagnostics=dict(diagnostics),
        )

    def _build_booking_draft(
        self,
        parsed: ParsedRequest,
        context: Optional[ConversationContext] = None,
        memory: Any = None,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tool_args = tool_args or {}
        follow_up = self._is_contextual_follow_up(parsed.raw)
        route_ctx = self._previous_route_context(self._memory_previous_result(memory)) if follow_up and memory is not None else {}

        def contextual_value(name: str) -> Any:
            if not follow_up or context is None:
                return None
            return getattr(context, name, None)

        passenger_count = (
            parsed.passengers
            if parsed.passengers is not None
            else tool_args.get("passenger_count")
            or tool_args.get("passengers")
            or contextual_value("passengers")
            or route_ctx.get("passengers")
        )
        try:
            passenger_count = int(passenger_count) if passenger_count not in (None, "") else None
        except Exception:
            passenger_count = None

        draft: Dict[str, Any] = {
            "source": parsed.source or tool_args.get("source") or contextual_value("source") or route_ctx.get("source"),
            "destination": parsed.destination or tool_args.get("destination") or contextual_value("destination") or route_ctx.get("destination"),
            "travel_date": parsed.travel_date or tool_args.get("travel_date") or tool_args.get("date") or contextual_value("travel_date") or route_ctx.get("travel_date"),
            "travel_class": parsed.travel_class or tool_args.get("travel_class") or contextual_value("travel_class") or route_ctx.get("travel_class"),
            "passenger_count": passenger_count,
            "train_number": parsed.train_number or tool_args.get("train_number") or contextual_value("train_number") or route_ctx.get("train_number"),
            "time_preference": parsed.time_hint or tool_args.get("time_hint") or contextual_value("time_hint"),
            "departure_after": parsed.departure_after or tool_args.get("departure_after") or contextual_value("departure_after"),
            "departure_before": parsed.departure_before or tool_args.get("departure_before") or contextual_value("departure_before"),
            "berth_preference": tool_args.get("berth_preference"),
            "budget": parsed.budget_max or tool_args.get("budget") or tool_args.get("budget_max") or contextual_value("budget_max"),
            "direct_only": bool(parsed.direct_only or tool_args.get("direct_only", False)),
        }

        missing = [field_name for field_name in BOOKING_REQUIRED_FIELDS if not draft.get(field_name)]
        draft["missing_required_fields"] = missing
        draft["ready_for_submit"] = not missing
        return draft

    def _format_booking_draft_message(self, draft: Dict[str, Any]) -> str:
        labels = {
            "source": "source",
            "destination": "destination",
            "travel_date": "travel date",
            "travel_class": "travel class",
            "passenger_count": "passenger count",
            "train_number": "train selection",
        }
        prefilled = []
        for key in ("source", "destination", "travel_date", "travel_class", "passenger_count", "train_number"):
            value = draft.get(key)
            if value not in (None, "", [], {}):
                prefilled.append(f"{labels[key]}: **{value}**")

        lines = ["I prepared a booking draft. I will not create the booking until the form is reviewed and submitted."]
        if prefilled:
            lines.append("\nPrefilled from the request/context: " + ", ".join(prefilled) + ".")

        missing = draft.get("missing_required_fields", [])
        if missing:
            pretty = [labels.get(item, item) for item in missing]
            lines.append("Missing required fields: **" + ", ".join(pretty) + "**.")
        else:
            lines.append("All required fields are present. Please review the form before submitting.")

        return "\n".join(lines)

    def _search_trains(self, parsed: ParsedRequest, tools: AgentTools) -> List[Dict[str, Any]]:
        raw = self._safe_tool_json(
            self._invoke_compat(
                tools.search_trains,
                source=parsed.source or "",
                destination=parsed.destination or "",
                date=parsed.travel_date or "",
                departure_after=parsed.departure_after or "",
                departure_before=parsed.departure_before or "",
                time_hint=parsed.time_hint or "",
                direct_only=parsed.direct_only,
                limit=parsed.limit or 5,
            )
        )
        trains = raw.get("trains") or raw.get("results") or []
        return trains if isinstance(trains, list) else []

    # ---------- FIXED _rank_trains (handles objects, dicts, and nested train dicts) ----------
    def _rank_trains(self, trains: List[Dict[str, Any]], parsed: ParsedRequest, src: str, dst: str) -> List[Dict[str, Any]]:
        engine = self.recommendation_engine
        ranked: Any = None
        if hasattr(engine, "rank"):
            try:
                ranked = engine.rank(
                    trains,
                    source=src,
                    destination=dst,
                    time_hint=parsed.time_hint,
                    departure_after=parsed.departure_after,
                    departure_before=parsed.departure_before,
                    sort_by=parsed.sort_by,
                    preference=parsed.preference,
                    direct_only=parsed.direct_only,
                    travel_class=parsed.travel_class,
                    passengers=parsed.passengers or 1,
                    limit=parsed.limit or 5,
                    budget_max=parsed.budget_max,
                )
            except TypeError:
                try:
                    ranked = engine.rank(trains)
                except Exception:
                    ranked = None
            except Exception:
                ranked = None
        if ranked is None and hasattr(engine, "recommend"):
            try:
                ranked = engine.recommend(trains, src=src, dst=dst, preference=parsed.preference)
            except Exception:
                ranked = None
        if ranked is None:
            ranked = self._fallback_rank_trains(trains, parsed)

        out: List[Dict[str, Any]] = []
        for item in ranked:
            # If item is already a dict, use it directly
            if isinstance(item, dict):
                out.append(item)
                continue
            # If item has a 'train' attribute, get the train
            train_obj = getattr(item, "train", item)
            # If train_obj is a dict, use it
            if isinstance(train_obj, dict):
                out.append(train_obj)
                continue
            # Otherwise, extract attributes from the object
            out.append({
                "train_number": getattr(train_obj, "train_number", None),
                "train_name": getattr(train_obj, "train_name", ""),
                "departure": getattr(train_obj, "departure", None) or getattr(train_obj, "departure_time", None),
                "arrival": getattr(train_obj, "arrival", None) or getattr(train_obj, "arrival_time", None),
                "duration": getattr(train_obj, "duration", None) or getattr(train_obj, "journey_time", None) or getattr(train_obj, "travel_time", None),
                "stops": getattr(train_obj, "stops", None) or getattr(train_obj, "total_stops", None),
                "fare": getattr(train_obj, "fare", None) or getattr(train_obj, "estimated_fare", None),
            })
        return out[: (parsed.limit or 5)]

    def _fallback_rank_trains(self, trains: List[Dict[str, Any]], parsed: ParsedRequest) -> List[Dict[str, Any]]:
        ranked = list(trains)

        def duration_minutes(val: Any) -> int:
            if val is None:
                return 10**9
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val)
            m = re.search(r"(\d+)\s*h", s)
            n = re.search(r"(\d+)\s*m", s)
            total = 0
            if m:
                total += int(m.group(1)) * 60
            if n:
                total += int(n.group(1))
            return total if total else 10**9

        ranked = self._filter_by_time_hint(ranked, parsed.time_hint or parsed.departure_after or parsed.departure_before)

        if parsed.sort_by == "fare" or "cheapest" in parsed.raw.lower():
            ranked.sort(key=lambda x: duration_minutes(x.get("fare") or x.get("estimated_fare") or x.get("min_fare")))
        elif parsed.sort_by == "duration" or "fastest" in parsed.raw.lower():
            ranked.sort(key=lambda x: duration_minutes(x.get("duration") or x.get("journey_time") or x.get("travel_time")))
        elif parsed.sort_by == "stops" or "fewest stops" in parsed.raw.lower():
            ranked.sort(key=lambda x: duration_minutes(x.get("stops") or x.get("total_stops") or x.get("stop_count")))
        elif "overnight" in parsed.raw.lower() or parsed.time_hint == "night":
            ranked.sort(key=lambda x: self._departure_minutes(x.get("departure") or x.get("dep") or x.get("departure_time")))
        else:
            ranked.sort(key=lambda x: (
                duration_minutes(x.get("stops") or x.get("total_stops") or x.get("stop_count")),
                duration_minutes(x.get("duration") or x.get("journey_time") or x.get("travel_time")),
            ))
        return ranked

    def _filter_by_time_hint(self, trains: List[Dict[str, Any]], hint: Optional[str]) -> List[Dict[str, Any]]:
        if not hint:
            return trains
        if re.match(r"^\d{2}:\d{2}$", hint):
            target_hour = int(hint.split(":")[0])
            return [t for t in trains if self._departure_hour(t) == -1 or abs(self._departure_hour(t) - target_hour) <= 1]
        if hint == "morning":
            return [t for t in trains if self._departure_hour(t) == -1 or 5 <= self._departure_hour(t) <= 11]
        if hint == "afternoon":
            return [t for t in trains if self._departure_hour(t) == -1 or 12 <= self._departure_hour(t) <= 16]
        if hint == "evening":
            return [t for t in trains if self._departure_hour(t) == -1 or 17 <= self._departure_hour(t) <= 21]
        if hint == "night":
            return [t for t in trains if self._departure_hour(t) == -1 or self._departure_hour(t) >= 22 or self._departure_hour(t) <= 4]
        return trains

    def _departure_hour(self, train: Dict[str, Any]) -> int:
        dt = train.get("departure") or train.get("dep") or train.get("departure_time") or train.get("start_time")
        if not dt:
            return -1
        m = re.search(r"(\d{1,2}):(\d{2})", str(dt))
        return int(m.group(1)) if m else -1

    def _departure_minutes(self, value: Any, default: int = 10**9) -> int:
        if not value:
            return default
        m = re.search(r"(\d{1,2}):(\d{2})", str(value))
        return int(m.group(1)) * 60 + int(m.group(2)) if m else default

    def _estimate_fare_by_distance(self, src: str, dst: str, travel_class: str, passengers: int) -> Optional[Dict[str, float]]:
        try:
            from app.services.fare_calculator import FareCalculator
            calc = FareCalculator()
            corridor = getattr(calc, 'CORRIDOR_DISTANCES', {})
            key = (src.upper(), dst.upper())
            distance = corridor.get(key)
            if not distance:
                key_rev = (dst.upper(), src.upper())
                distance = corridor.get(key_rev)
            if distance is None:
                return None
            breakdown = calc.calculate(
                travel_class=travel_class,
                distance_km=distance,
                passengers=passengers,
                source_code=src,
                dest_code=dst,
            )
            return {
                "per_passenger": breakdown.per_passenger,
                "total": breakdown.total_fare,
                "distance_km": distance,
            }
        except Exception as e:
            logger.warning("Distance fallback failed: %s", e)
            return None

    def _extract_booking_id_fallback(self, text: str) -> Optional[str]:
        m = re.search(r'\b(?:cancel|#|id)\s*#?\s*(\d+)\b', text, re.IGNORECASE)
        if m:
            return m.group(1)
        if re.search(r'\bcancel\b', text, re.IGNORECASE):
            m = re.search(r'\b(\d+)\b', text)
            if m:
                return m.group(1)
        return None

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

    def _safe_tool_json(self, raw: Any) -> Dict[str, Any]:
        parsed = self._safe_json_loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"status": "ok", "result": parsed}
        return {"status": "ok", "result": parsed}

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


    def _is_contextual_follow_up(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        cues = (
            "what about fare",
            "fare for it",
            "cheapest one",
            "fastest one",
            "best one",
            "first one",
            "second one",
            "third one",
            "book it",
            "book the first one",
            "book that",
            "that train",
            "this train",
            "same train",
            "route of it",
            "route for it",
            "show route",
            "show fare",
            "compare",
            "price of it",
        )
        return any(cue in normalized for cue in cues) or bool(re.search(r"(first|second|third|that|this|it)", normalized))

    def _previous_route_context(self, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not previous_result:
            return {}
        entities = previous_result.get("entities") or {}
        results = previous_result.get("results") or previous_result.get("trains") or []
        selected_train_number = previous_result.get("selected_train_number")
        if not selected_train_number and isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                selected_train_number = first.get("train_number") or first.get("train_no")
        return {
            "source": entities.get("source"),
            "destination": entities.get("destination"),
            "travel_date": entities.get("date") or entities.get("travel_date"),
            "travel_class": entities.get("travel_class"),
            "passengers": entities.get("passengers"),
            "train_number": selected_train_number,
            "selected_option_index": previous_result.get("selected_option_index"),
        }

    # ---------- Formatting helpers ----------
    def _format_train_search(self, src: str, dst: str, trains: List[Dict[str, Any]], parsed: ParsedRequest, multi: bool = False) -> str:
        if not trains:
            return f"I couldn't find any trains from **{src}** to **{dst}**."
        lines = [f"🚆 Found **{len(trains)} train(s)** from **{src}** → **{dst}**:\n"]
        for idx, train in enumerate(trains, start=1):
            train_no = train.get("train_number", "-")
            train_name = train.get("train_name", "")
            dep = train.get("departure") or train.get("dep") or train.get("departure_time") or "--:--"
            arr = train.get("arrival") or train.get("arr") or train.get("arrival_time") or "--:--"
            duration = train.get("duration") or train.get("journey_time") or train.get("travel_time") or "N/A"
            stop_count = train.get("stops") or train.get("total_stops") or train.get("stop_count")
            note = train.get("note") or ""
            line = f"{idx}. **{train_no}** – {train_name} | {dep} → {arr}"
            if duration and duration != "N/A":
                line += f" | ⏱ {duration}"
            if stop_count is not None:
                line += f" | 🛑 {stop_count} stops"
            if note:
                line += f" | _{note}_"
            lines.append(line)
        if parsed.time_hint or parsed.departure_after or parsed.departure_before:
            lines.append("\nFiltered by your time preference.")
        if multi:
            lines.append("\nI can compare the top options, show fares, or help book one.")
        else:
            lines.append("\nAsk me for fares, route details, the cheapest option, or the fastest train.")
        return "\n".join(lines)

    def _format_single_fare(self, result: Dict[str, Any], src: str, dst: str, pax: int) -> str:
        if result.get("status") == "error":
            return f"❌ Could not get fare: {result.get('message', 'Unknown error')}"
        class_code = result.get("class_code") or result.get("class") or "SL"
        class_name = result.get("class_name", class_code)
        per_ticket = result.get("per_passenger", result.get("fare", 0))
        total = result.get("total_fare", result.get("total", per_ticket * pax))
        distance_km = result.get("distance_km")
        note = "~estimated" if result.get("is_estimated", True) else "from route data"
        distance_text = f" | {distance_km:.0f} km" if isinstance(distance_km, (int, float)) else ""
        return (
            f"💰 **Fare for {class_name}** ({src} → {dst})\n"
            f"🚆 Train: **{result.get('train_number', '-') }**{distance_text}\n"
            f"Per ticket: **₹{float(per_ticket):,.0f}**\n"
            f"Total ({pax} pax): **₹{float(total):,.0f}**\n"
            f"_{note}_"
        )

    def _format_fare_table(self, result: Dict[str, Any], src: str, dst: str, pax: int) -> str:
        if result.get("status") == "error":
            return f"❌ Could not get fare: {result.get('message', 'Unknown error')}"
        if "fares" not in result:
            return self._format_single_fare(result, src, dst, pax)
        fare_map = result.get("fares", {})
        lines = [
            f"💰 **Fare Estimates** – {src} → {dst}",
            f"🚆 Train: **{result.get('train_number', '-') }** ({result.get('train_name', '')})",
            f"👥 Passengers: **{pax}**\n",
            "| Class | Per Ticket | Total |",
            "|-------|-----------:|------:|",
        ]
        for code in SUPPORTED_CLASSES:
            if code in fare_map:
                fare = fare_map[code]
                lines.append(f"| {fare.get('class_name', code)} | ₹{fare.get('per_passenger', 0):,.0f} | ₹{fare.get('total', 0):,.0f} |")
        dist = self._extract_distance_from_result(result)
        if dist is not None:
            lines.append(f"\n_Fares are approximate ({dist:.0f} km, demo estimate)._")
        else:
            lines.append("\n_Fares are approximate (demo estimate)._")
        return "\n".join(lines)

    def _extract_distance_from_result(self, result: Dict[str, Any]) -> Optional[float]:
        if isinstance(result.get("distance_km"), (int, float)):
            return float(result["distance_km"])
        fares = result.get("fares", {})
        for _, item in fares.items():
            if isinstance(item, dict) and isinstance(item.get("distance_km"), (int, float)):
                return float(item["distance_km"])
        return None

    def _format_route(self, result: Dict[str, Any], train_number: str) -> str:
        route = result.get("route", [])
        total_stops = result.get("total_stops", len(route))
        if not route:
            return result.get("message", "Route not found.")
        lines = [f"🗺️ **Route for Train {train_number}** ({total_stops} stops):\n"]
        for stop in route[:20]:
            seq = stop.get("seq", stop.get("stop_no", "-"))
            station = stop.get("station_code", stop.get("station", "-"))
            arrival = stop.get("arrival", "--:--")
            departure = stop.get("departure", "--:--")
            dist = stop.get("distance_km")
            extra = f" ({dist} km)" if dist is not None else ""
            lines.append(f"{seq:>2}. **{station}**  arr:{arrival}  dep:{departure}{extra}")
        if total_stops > 20:
            lines.append(f"...and {total_stops - 20} more stops.")
        return "\n".join(lines)

    def _format_station_info(self, result: Dict[str, Any]) -> str:
        return (
            f"🏠 **{result.get('station_name', '-') }** ({result.get('station_code', '-')})\n"
            f"📍 City: {result.get('city', 'Unknown')}\n"
            f"ℹ️ Type: {result.get('station_type', 'Railway station')}"
        )

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

    def _format_booking_confirmation(self, result: Dict[str, Any]) -> str:
        return (
            f"✅ **Booking Confirmed!**\n\n"
            f"🆔 Booking ID: **{result.get('booking_id', '-') }**\n"
            f"🚆 Train: **{result.get('train_number', '-') }**\n"
            f"📍 {result.get('source', '-') } → {result.get('destination', '-') }\n"
            f"🎫 Class: **{result.get('class', '-') }** | 👥 {result.get('passengers', '-') } pax\n"
            f"📅 Date: **{result.get('travel_date', '-') }**\n"
            f"💰 Est. fare: ₹{float(result.get('estimated_total_fare', 0)):,.0f}"
        )

    def _booking_clarification_message(self, missing: Sequence[str], raw_text: str) -> str:
        mapping = {
            "source": "source station",
            "destination": "destination station",
            "travel_class": "class",
            "passengers": "number of passengers",
            "travel_date": "travel date",
            "booking_id": "booking ID",
            "train_number": "train number",
            "station": "station",
        }
        pretty = [mapping[m] for m in missing if m in mapping]
        if not pretty:
            return "Please share a little more detail so I can help."
        if len(pretty) == 1:
            return f"Please share the {pretty[0]} so I can proceed."
        return "Please share " + ", ".join(pretty) + " so I can proceed."

    def _friendly_tool_error(self, action: str, payload: Dict[str, Any]) -> str:
        msg = payload.get("message") or payload.get("error") or "unknown error"
        return f"❌ Could not complete {action}: {msg}"

    def _clarify_missing_route(self, raw_text: str) -> str:
        return (
            "Please tell me the source and destination stations. For example:\n"
            "• Show trains from Bangalore to Mangalore\n"
            "• Find trains between Mysore and Chennai"
        )

    def _clarify_train_number(self) -> str:
        return "Please provide the train number."

    def _greeting_message(self) -> str:
        return (
            "👋 Hello! I’m **RailMitra**. I can help you search trains, estimate fares, view routes, and handle demo bookings.\n\n"
            "Try asking:\n"
            "• Show trains from Bangalore to Mangalore after 8 PM\n"
            "• What is the sleeper fare for train 16585?\n"
            "• Tell me the route of train 12627"
        )

    def _booking_modify_message(self) -> str:
        return "I can understand booking changes, but your current booking backend only exposes create/cancel/history flows."

    def _default_travel_date(self) -> str:
        return datetime.now().date().isoformat()

    def _fallback_help_message(self) -> str:
        return (
            "I can help with train search, fares, routes, station info, and demo bookings.\n\n"
            "Examples:\n"
            "• Show trains from Bangalore to Mangalore after 8 PM\n"
            "• Find the sleeper fare for train 16585\n"
            "• Tell me the route of train 12627\n"
            "• Book 2 sleeper seats from Bangalore to Mangalore tomorrow"
        )

    def _guess_station_from_text(self, text: str) -> Optional[str]:
        # simple fallback: if text contains a known station code (uppercase 2-5 letters)
        match = re.search(r"\b([A-Z]{2,5})\b", text)
        return match.group(1) if match else None

    # ---------- LLM agent loop ----------
    def _build_messages(self, user_message: str, history: List[Dict[str, str]], context: ConversationContext) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if any([context.source, context.destination, context.train_number, context.travel_class, context.time_hint]):
            messages.append({"role": "system", "content": f"Context: {self._context_summary(context)}"})

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
            messages = self._build_messages(user_message, conversation_history, context)
            tool_objects = tools.build()
            tool_map = self._build_tool_map(tool_objects)

            # Use the official Hugging Face client if available
            if HF_CLIENT_AVAILABLE:
                client = InferenceClient(
                    model=HF_MODEL_NAME,
                    token=self.hf_token
                )
                response = client.chat.completions.create(
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature
                )
                assistant_message = response.choices[0].message.content
                if assistant_message:
                    tool_call = self._parse_tool_call(assistant_message, tool_map)
                    if tool_call:
                        tool_name, tool_args = tool_call
                        logger.info("[agent] Tool call: %s with args: %s", tool_name, tool_args)
                        if tool_name == "book_ticket":
                            logger.warning("[agent] Suppressed LLM book_ticket tool call; returning booking draft text")
                            draft = self._build_booking_draft(parsed, context, tool_args=tool_args)
                            return self._format_booking_draft_message(draft)
                        tool = tool_map.get(tool_name)
                        if tool:
                            result = self._invoke_tool(tool, tool_args)
                            if tool_name == "search_trains" and result.get("status") == "ok":
                                trains = result.get("trains", [])
                                if trains:
                                    self._remember_selected_results(session_id, parsed, trains, selected_index=0)
                            # Return the tool result as the final answer for simplicity
                            return json.dumps(result, ensure_ascii=False)
                        else:
                            logger.warning(f"Tool {tool_name} not found")
                    else:
                        return assistant_message

            # Fallback to raw requests with IPv4 fix
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

                tool_call = self._parse_tool_call(assistant_message, tool_map)
                if tool_call:
                    tool_name, tool_args = tool_call
                    logger.info("[agent] Tool call: %s with args: %s", tool_name, tool_args)

                    if tool_name == "book_ticket":
                        logger.warning("[agent] Suppressed LLM book_ticket tool call; returning booking draft text")
                        draft = self._build_booking_draft(parsed, context, tool_args=tool_args)
                        return self._format_booking_draft_message(draft)

                    tool = tool_map.get(tool_name)
                    if not tool:
                        messages.append({"role": "assistant", "content": assistant_message})
                        messages.append({"role": "user", "content": f"Error: Tool '{tool_name}' not found. Please use one of: {list(tool_map.keys())}."})
                        continue

                    result = self._invoke_tool(tool, tool_args)
                    logger.info("[agent] Tool result: %s", str(result)[:200])

                    if tool_name == "search_trains" and result.get("status") == "ok":
                        trains = result.get("trains", [])
                        if trains:
                            self._remember_selected_results(session_id, parsed, trains, selected_index=0)

                    messages.append({"role": "assistant", "content": assistant_message})
                    messages.append({"role": "user", "content": f"Tool result: {json.dumps(result, ensure_ascii=False)}"})
                    continue
                else:
                    # No tool call – this is the final answer
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

    def _parse_tool_call(self, text: str, tool_map: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data and "args" in data and isinstance(data["args"], dict):
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data and "args" in data and isinstance(data["args"], dict):
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                pass

        try:
            data = json.loads(text.strip())
            if "tool" in data and "args" in data and isinstance(data["args"], dict):
                return data["tool"], data["args"]
        except json.JSONDecodeError:
            pass

        return None

    def _build_tool_schemas(self, tools: Sequence[Any]) -> List[Dict[str, Any]]:
        return []  # Not used

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


def build_agent_service() -> AgentService:
    return AgentService()
