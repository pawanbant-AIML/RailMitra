"""
agent/agent_service.py — research-grade, production-oriented RailMitra agent.

Design goals:
- Deterministic local routing for the common railway tasks
- Optional HF tool-calling LLM fallback for unsupported queries
- Strong conversation context carryover
- Better ambiguity handling and follow-up questions
- Graceful degradation when Datameet/Railway data is incomplete
- No hallucinated train numbers, fares, routes, or station facts
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from sqlalchemy.orm import Session

from app.agent.tools import AgentTools
from app.core.logger import logger

try:
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    BaseTool = Any  # type: ignore


HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions"
)
HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_MAX_TOOL_ITERATIONS = 3
DEFAULT_MAX_HISTORY_TURNS = 8
DEFAULT_MAX_OUTPUT_TOKENS = 900
DEFAULT_TEMPERATURE = 0.25
DEFAULT_RETRIES = 2

SUPPORTED_CLASSES = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]
CLASS_ALIASES = {
    "general": "GN",
    "gn": "GN",
    "second sitting": "2S",
    "2s": "2S",
    "sleeper": "SL",
    "sl": "SL",
    "chair car": "CC",
    "cc": "CC",
    "3ac": "3A",
    "3a": "3A",
    "third ac": "3A",
    "2ac": "2A",
    "2a": "2A",
    "second ac": "2A",
    "1ac": "1A",
    "1a": "1A",
    "first ac": "1A",
    "executive": "EC",
    "ec": "EC",
}

STATION_ALIASES = {
    "bangalore": "SBC",
    "bengaluru": "SBC",
    "blr": "SBC",
    "sbc": "SBC",
    "yesvantpur": "YPR",
    "ypr": "YPR",
    "mysore": "MYS",
    "mysuru": "MYS",
    "mys": "MYS",
    "hubli": "UBL",
    "hubballi": "UBL",
    "ubl": "UBL",
    "mangalore": "MAQ",
    "mangaluru": "MAQ",
    "maq": "MAQ",
    "mumbai": "CSMT",
    "bombay": "CSMT",
    "csmt": "CSMT",
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "ndls": "NDLS",
    "chennai": "MAS",
    "madras": "MAS",
    "mas": "MAS",
    "kolkata": "HWH",
    "howrah": "HWH",
    "hwh": "HWH",
    "hyderabad": "HYB",
    "hyd": "HYB",
    "secunderabad": "SC",
    "pune": "PUNE",
    "goa": "MAO",
    "udupi": "UD",
    "hassan": "HAS",
    "shimoga": "SMET",
    "shivamogga": "SMET",
    "davangere": "DVG",
    "kochi": "ERS",
    "ernakulam": "ERS",
    "ers": "ERS",
    "trivandrum": "TVC",
    "thiruvananthapuram": "TVC",
    "tvc": "TVC",
    "coimbatore": "CBE",
    "madurai": "MDU",
    "ahmedabad": "ADI",
    "surat": "ST",
    "vadodara": "BRC",
    "jaipur": "JP",
    "jodhpur": "JU",
    "udaipur": "UDZ",
    "lucknow": "LKO",
    "varanasi": "BSB",
    "patna": "PNBE",
    "bhopal": "BPL",
    "nagpur": "NGP",
    "indore": "INDB",
    "visakhapatnam": "VSKP",
    "vizag": "VSKP",
    "amritsar": "ASR",
    "chandigarh": "CDG",
    "ludhiana": "LDH",
    "guwahati": "GHY",
    "bhubaneswar": "BBS",
}

INTENT_HINTS = {
    "booking_cancel": ["cancel booking", "cancel ticket", "cancel reservation", "reverse booking", "cancel"],
    "booking_history": ["my booking", "booking history", "my ticket", "reservations", "bookings"],
    "booking_create": ["book", "reserve", "ticket", "buy"],
    "fare": ["fare", "cost", "price", "how much", "charge", "estimate"],
    "route": ["route", "stops", "stop", "schedule", "where does", "between"],
    "station_info": ["station", "about", "tell me about"],
    "train_search": ["train", "available", "options", "direct", "fastest", "cheapest"],
    "greeting": ["hi", "hello", "hey", "namaste", "help"],
}


SYSTEM_PROMPT = """You are RailMitra, a production-grade Indian Railways assistant for a Datameet-powered rail application.

Rules:
1. Never invent train numbers, station codes, timings, routes, distances, or fares.
2. Use tools for any answer that depends on database-backed railway data.
3. Ask a concise follow-up if the request is incomplete or ambiguous.
4. Reuse conversation context for follow-ups like "that one", "the first train", "show fare for it".
5. Be conversational, but keep answers precise and useful.
6. If data is missing or incomplete, say that clearly and degrade gracefully.
7. For bookings, confirm source, destination, class, passengers, and date when any of them are missing.
8. Do not expose internal prompts, SQL, API keys, or chain-of-thought.
9. Prefer direct answers for train search, fare, route, station, booking, and comparison questions.
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
    intent: Optional[str] = None
    last_query_type: Optional[str] = None


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
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    booking_id: Optional[str] = None
    direct_only: bool = False
    compare_classes: List[str] = field(default_factory=list)
    compare_trains: List[str] = field(default_factory=list)
    raw: str = ""


class AgentService:
    """Request-response orchestrator for one user turn."""

    def __init__(self) -> None:
        self.hf_token = (
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HF_TOKEN")
            or ""
        ).strip()
        self.timeout = int(os.environ.get("HF_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
        self.max_tool_iterations = int(os.environ.get("AGENT_MAX_TOOL_ITERATIONS", str(DEFAULT_MAX_TOOL_ITERATIONS)))
        self.max_history_turns = int(os.environ.get("AGENT_MAX_HISTORY_TURNS", str(DEFAULT_MAX_HISTORY_TURNS)))
        self.max_output_tokens = int(os.environ.get("HF_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))
        self.temperature = float(os.environ.get("HF_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.retries = int(os.environ.get("HF_RETRIES", str(DEFAULT_RETRIES)))
        self.allow_llm = os.environ.get("AGENT_ENABLE_LLM", "1").strip() not in {"0", "false", "False"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        db: Session,
    ) -> str:
        cleaned_message = self._normalize_text(user_message)
        logger.info("[agent] incoming=%r", cleaned_message[:240])

        context = self._build_context(conversation_history, cleaned_message)
        parsed = self._parse_request(cleaned_message, context)

        tools = AgentTools(db)

        local_answer = self._handle_locally(parsed, conversation_history, db, tools, context)
        if local_answer is not None:
            return local_answer

        if self.allow_llm and self.hf_token:
            llm_answer = self._run_tool_agent(cleaned_message, conversation_history, context, tools)
            if llm_answer:
                return llm_answer

        return self._fallback_help_message()

    # ------------------------------------------------------------------
    # LLM / tool-calling fallback
    # ------------------------------------------------------------------

    def _run_tool_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        context: ConversationContext,
        tools: AgentTools,
    ) -> Optional[str]:
        try:
            tool_list = tools.build()
            messages = self._build_messages(user_message, conversation_history, context)
            tool_schemas = self._build_tool_schemas(tool_list)
            tool_map = self._build_tool_map(tool_list)

            for _ in range(self.max_tool_iterations):
                llm_message = self._call_llm(messages, tool_schemas)
                if not llm_message:
                    return None

                content = (llm_message.get("content") or "").strip()
                tool_calls = llm_message.get("tool_calls") or []

                if not tool_calls:
                    return self._sanitize_output(content) or None

                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    raw_args = tc.get("function", {}).get("arguments", "{}")
                    tc_id = tc.get("id", f"call_{tool_name}")
                    parsed_args = self._safe_json_loads(raw_args)

                    if tool_name not in tool_map:
                        tool_result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
                    else:
                        try:
                            tool_result = self._invoke_tool(tool_map[tool_name], parsed_args)
                        except Exception as exc:  # pragma: no cover
                            logger.exception("[agent] tool failed: %s", tool_name)
                            tool_result = {"status": "error", "message": f"Tool {tool_name} failed: {exc}"}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
            return None
        except Exception as exc:  # pragma: no cover
            logger.exception("[agent] LLM tool loop failed: %s", exc)
            return None

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "model": HF_MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = (data.get("choices") or [{}])[0]
                    return choice.get("message") or {}

                if resp.status_code == 429:
                    time.sleep(min(8, 2 * (attempt + 1)))
                    continue

                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(min(4, attempt + 1))
                    continue

                logger.error("[agent] hf status=%d body=%s", resp.status_code, resp.text[:400])
                return None
            except requests.Timeout:
                logger.warning("[agent] hf timeout attempt=%d", attempt + 1)
            except Exception as exc:  # pragma: no cover
                logger.exception("[agent] hf request exception: %s", exc)
                return None

        return None

    def _build_messages(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        context: ConversationContext,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if any([context.source, context.destination, context.train_number, context.travel_class]):
            messages.append(
                {
                    "role": "system",
                    "content": f"Conversation memory summary: {self._context_summary(context)}",
                }
            )

        recent = history[-(self.max_history_turns * 2) :]
        for item in recent:
            role = item.get("role", "user")
            content = self._sanitize_output(item.get("content", ""))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_tool_schemas(self, tools: Sequence[Any]) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for tool in tools:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", "unknown_tool"),
                        "description": getattr(tool, "description", ""),
                        "parameters": self._get_tool_args_schema(tool),
                    },
                }
            )
        return schemas

    def _get_tool_args_schema(self, tool: Any) -> Dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None:
            return {"type": "object", "properties": {}}
        try:
            if hasattr(args_schema, "model_json_schema"):
                return args_schema.model_json_schema()
            if hasattr(args_schema, "schema"):
                return args_schema.schema()
        except Exception:
            logger.exception("[agent] failed building schema for %s", getattr(tool, "name", "unknown_tool"))
        return {"type": "object", "properties": {}}

    def _build_tool_map(self, tools: Sequence[Any]) -> Dict[str, Any]:
        return {getattr(tool, "name", f"tool_{idx}"): tool for idx, tool in enumerate(tools)}

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

    # ------------------------------------------------------------------
    # Local orchestration
    # ------------------------------------------------------------------

    def _handle_locally(
        self,
        parsed: ParsedRequest,
        history: List[Dict[str, str]],
        db: Session,
        tools: AgentTools,
        context: ConversationContext,
    ) -> Optional[str]:
        intent = parsed.intent

        if intent == "greeting":
            return self._greeting_message()

        if intent == "booking_history":
            result = self._safe_tool_json(tools.get_booking_history())
            if result.get("status") == "empty":
                return "You do not have any bookings yet. Search for trains to get started."
            if result.get("status") == "error":
                return self._friendly_tool_error("booking history", result)
            return self._format_booking_history(result)

        if intent == "booking_cancel":
            if not parsed.booking_id:
                return "Please share the booking ID so I can cancel it."
            result = self._safe_tool_json(tools.cancel_booking(parsed.booking_id))
            if result.get("status") == "error":
                return self._friendly_tool_error("cancellation", result)
            return result.get("message", f"Booking #{parsed.booking_id} has been processed.")

        if intent == "station_info":
            station_query = parsed.source or parsed.destination or self._guess_station_from_text(parsed.raw) or context.source or context.destination
            if not station_query:
                return "Please provide a station name or code."
            result = self._safe_tool_json(tools.get_station_info(station_query))
            if result.get("status") != "ok":
                return result.get("message", "Station not found.")
            return self._format_station_info(result)

        if intent == "train_info":
            if not parsed.train_number:
                return "Please provide the train number."
            info = self._safe_tool_json(tools.get_train_info(parsed.train_number))
            route = self._safe_tool_json(tools.get_train_route(parsed.train_number))
            if info.get("status") == "error" and route.get("status") == "error":
                return self._friendly_tool_error("train info", route if route.get("status") == "error" else info)
            return self._format_train_info(info, route)

        if intent == "route":
            if parsed.train_number:
                route = self._safe_tool_json(tools.get_train_route(parsed.train_number))
                if route.get("status") != "ok":
                    return route.get("message", "Route not found.")
                return self._format_route(route, parsed.train_number)

            if parsed.source and parsed.destination:
                search = self._search_and_rank(parsed, tools)
                if search is not None:
                    return search
                return self._clarify_missing_route(parsed.raw)

            return self._clarify_missing_route(parsed.raw)

        if intent == "fare":
            return self._handle_fare_query(parsed, tools, context)

        if intent in {"train_search", "multi_intent", "recommendation"}:
            if parsed.source and parsed.destination:
                search = self._search_and_rank(parsed, tools)
                if search is not None:
                    return search
                return self._clarify_missing_route(parsed.raw)
            return self._clarify_missing_route(parsed.raw)

        if intent == "booking_create":
            return self._handle_booking_query(parsed, tools, context)

        return self._handle_general_request(parsed, tools, context)

    def _handle_general_request(self, parsed: ParsedRequest, tools: AgentTools, context: ConversationContext) -> Optional[str]:
        text = parsed.raw

        if parsed.compare_classes and (parsed.source or context.source) and (parsed.destination or context.destination):
            src = parsed.source or context.source
            dst = parsed.destination or context.destination
            if not src or not dst:
                return self._clarify_missing_route(text)

            train_number = parsed.train_number or context.train_number
            if not train_number:
                chosen = self._choose_best_train_candidate(self._search_trains(src, dst, tools, parsed.travel_date), parsed)
                train_number = chosen.get("train_number") if chosen else None
            if not train_number:
                return f"I could not find a train from {src} to {dst} to compare fares."

            fare_rows = []
            for cls in parsed.compare_classes[:4]:
                fare = self._safe_tool_json(
                    tools.get_fare(
                        train_number=train_number,
                        source=src,
                        destination=dst,
                        travel_class=cls,
                        passengers=parsed.passengers or context.passengers or 1,
                    )
                )
                if fare.get("status") == "ok":
                    fare_rows.append(fare)
            if not fare_rows:
                return "I could not compare those fare classes with the available data."
            return self._format_fare_comparison_table(fare_rows, src, dst)

        if parsed.source and parsed.destination:
            search = self._search_and_rank(parsed, tools)
            if search is not None:
                return search

        if parsed.train_number:
            info = self._safe_tool_json(tools.get_train_info(parsed.train_number))
            route = self._safe_tool_json(tools.get_train_route(parsed.train_number))
            if info.get("status") == "ok" or route.get("status") == "ok":
                return self._format_train_info(info, route)

        return None

    def _search_trains(
        self,
        source: str,
        destination: str,
        tools: AgentTools,
        travel_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        result = self._safe_tool_json(tools.search_trains(source, destination, travel_date or ""))
        trains = result.get("trains") if isinstance(result.get("trains"), list) else []
        return [t for t in trains if isinstance(t, dict)]

    def _search_and_rank(self, parsed: ParsedRequest, tools: AgentTools) -> Optional[str]:
        src = parsed.source
        dst = parsed.destination
        if not src or not dst:
            return None

        trains = self._search_trains(src, dst, tools, parsed.travel_date)
        if not trains:
            return f"No trains were found from **{src}** to **{dst}** in the available Datameet data."

        trains = self._apply_time_filter(trains, parsed.time_hint)
        if parsed.direct_only:
            direct = [t for t in trains if self._is_direct_or_low_stop(t)]
            if direct:
                trains = direct

        if parsed.sort_by:
            trains = self._sort_trains(trains, parsed.sort_by)
        else:
            # For search responses, a sensible default is a balanced ranking.
            trains = self._sort_trains(trains, "duration")

        if parsed.limit:
            trains = trains[:parsed.limit]
        else:
            trains = trains[:10]

        return self._format_train_search(src, dst, trains, parsed)

    def _handle_fare_query(self, parsed: ParsedRequest, tools: AgentTools, context: ConversationContext) -> Optional[str]:
        src = parsed.source or context.source
        dst = parsed.destination or context.destination
        train_number = parsed.train_number or context.train_number
        pax = parsed.passengers or context.passengers or 1

        if not src or not dst:
            return "Please tell me the source and destination first so I can estimate fare."

        requested_all = self._wants_all_classes(parsed.raw)
        requested_classes = parsed.compare_classes[:]

        if not train_number:
            trains = self._search_trains(src, dst, tools, parsed.travel_date)
            if not trains:
                return f"I could not find trains from **{src}** to **{dst}** to estimate fare."
            trains = self._apply_time_filter(trains, parsed.time_hint)
            chosen = self._choose_best_train_candidate(trains, parsed)
            if not chosen:
                return f"I found route data for **{src} → {dst}**, but I could not choose a train for fare estimation."
            train_number = chosen.get("train_number")
            if not train_number:
                return f"I found trains from **{src} → {dst}**, but the train number is missing in the available data."
            if not requested_all and not parsed.travel_class and not requested_classes:
                # For an underspecified fare request, choose the most natural class response.
                requested_all = True

        if requested_classes:
            fare_rows = []
            for cls in requested_classes[:4]:
                fare = self._safe_tool_json(
                    tools.get_fare(
                        train_number=train_number,
                        source=src,
                        destination=dst,
                        travel_class=cls,
                        passengers=pax,
                    )
                )
                if fare.get("status") == "ok":
                    fare_rows.append(fare)
            if not fare_rows:
                return "I could not compare the requested fare classes with the available data."
            return self._format_fare_comparison_table(fare_rows, src, dst)

        if requested_all:
            fare = self._safe_tool_json(
                tools.get_fare(
                    train_number=train_number,
                    source=src,
                    destination=dst,
                    travel_class="ALL",
                    passengers=pax,
                )
            )
            if fare.get("status") != "ok":
                return self._friendly_tool_error("fare lookup", fare)
            return self._format_fare_table(fare, src, dst, pax)

        travel_class = parsed.travel_class or context.travel_class
        if not travel_class:
            travel_class = "SL"

        if travel_class not in SUPPORTED_CLASSES:
            travel_class = self._normalize_class_code(travel_class)

        if travel_class not in SUPPORTED_CLASSES:
            return f"Please provide a valid class: {', '.join(SUPPORTED_CLASSES)}."

        fare = self._safe_tool_json(
            tools.get_fare(
                train_number=train_number,
                source=src,
                destination=dst,
                travel_class=travel_class,
                passengers=pax,
            )
        )
        if fare.get("status") != "ok":
            return self._friendly_tool_error("fare lookup", fare)
        return self._format_single_fare(fare, src, dst, pax)

    def _handle_booking_query(self, parsed: ParsedRequest, tools: AgentTools, context: ConversationContext) -> Optional[str]:
        src = parsed.source or context.source
        dst = parsed.destination or context.destination
        pax = parsed.passengers or context.passengers
        travel_class = parsed.travel_class or context.travel_class
        travel_date = parsed.travel_date or context.travel_date
        train_number = parsed.train_number or context.train_number

        missing = []
        if not src:
            missing.append("source")
        if not dst:
            missing.append("destination")
        if not travel_class:
            missing.append("class")
        if not pax:
            missing.append("passengers")
        if not travel_date:
            travel_date = self._default_travel_date()

        if missing:
            return self._booking_clarification_message(missing, parsed.raw)

        if not train_number:
            trains = self._search_trains(src, dst, tools, travel_date)
            if not trains:
                return f"I could not find any trains from **{src}** to **{dst}** for booking."
            chosen = self._choose_best_train_candidate(trains, parsed)
            train_number = chosen.get("train_number") if chosen else None

        if not train_number:
            return "I found route data, but I could not determine which train to book."

        book = self._safe_tool_json(
            tools.book_ticket(
                source=src,
                destination=dst,
                travel_class=travel_class,
                passengers=pax,
                travel_date=travel_date,
                train_number=train_number,
            )
        )
        if book.get("status") != "confirmed":
            return self._friendly_tool_error("booking", book)
        return self._format_booking_confirmation(book)

    # ------------------------------------------------------------------
    # Parsing / intent detection
    # ------------------------------------------------------------------

    def _parse_request(self, text: str, context: ConversationContext) -> ParsedRequest:
        lowered = text.lower()
        intent = self._detect_intent(lowered)
        source, destination = self._extract_stations(lowered)
        train_number = self._extract_train_number(lowered)
        travel_class = self._extract_class(lowered)
        passengers = self._extract_passengers(lowered)
        travel_date = self._extract_date(lowered)
        time_hint = self._extract_time_hint(lowered)
        sort_by = self._extract_sort_hint(lowered)
        limit = self._extract_limit_hint(lowered)
        booking_id = self._extract_booking_id(lowered)
        direct_only = any(x in lowered for x in ["direct only", "non stop", "non-stop", "fewest stops", "no stops"])
        compare_classes = self._extract_compare_classes(lowered)
        compare_trains = self._extract_compare_trains(lowered)

        source = source or context.source
        destination = destination or context.destination
        train_number = train_number or context.train_number
        travel_class = travel_class or context.travel_class
        passengers = passengers or context.passengers
        travel_date = travel_date or context.travel_date
        time_hint = time_hint or context.time_hint

        return ParsedRequest(
            intent=intent,
            source=source,
            destination=destination,
            train_number=train_number,
            travel_class=travel_class,
            passengers=passengers,
            travel_date=travel_date,
            time_hint=time_hint,
            sort_by=sort_by,
            limit=limit,
            booking_id=booking_id,
            direct_only=direct_only,
            compare_classes=compare_classes,
            compare_trains=compare_trains,
            raw=text,
        )

    def _detect_intent(self, text: str) -> str:
        score = {key: 0 for key in INTENT_HINTS}
        for intent, hints in INTENT_HINTS.items():
            for hint in hints:
                if hint in text:
                    score[intent] += 1

        if any(k in text for k in ["compare", "versus", "vs", "difference"]):
            if any(k in text for k in ["fare", "class", "sleeper", "3a", "2a", "1a", "cc", "ec"]):
                return "fare"
            if any(k in text for k in ["train", "first train", "second train", "third train"]):
                return "multi_intent"

        if any(k in text for k in ["fastest", "cheapest", "best balance", "best option", "recommend", "suggest"]):
            return "recommendation"

        if any(k in text for k in ["what is train", "tell me about train", "train number", "train info"]):
            return "train_info"

        for key in ["booking_cancel", "booking_history", "booking_create", "fare", "route", "station_info", "train_search", "greeting"]:
            if score.get(key, 0) > 0:
                return key

        if any(word in text for word in ["book", "reserve", "ticket"]):
            return "booking_create"
        if "cancel" in text:
            return "booking_cancel"
        if any(word in text for word in ["history", "bookings", "reservations"]):
            return "booking_history"
        if any(word in text for word in ["fare", "price", "cost", "how much"]):
            return "fare"
        if any(word in text for word in ["route", "stop", "stops", "schedule", "where does"]):
            return "route"
        if any(word in text for word in ["station", "station info", "about"]):
            return "station_info"
        if any(word in text for word in ["hi", "hello", "hey", "namaste", "help"]):
            return "greeting"
        return "train_search"

    def _extract_stations(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        found: List[str] = []
        aliases = sorted(STATION_ALIASES.keys(), key=len, reverse=True)
        padded = f" {text} "
        for alias in aliases:
            if f" {alias} " in padded:
                code = STATION_ALIASES[alias]
                if code not in found:
                    found.append(code)
                if len(found) == 2:
                    break

        if len(found) >= 2:
            return found[0], found[1]
        if len(found) == 1:
            return found[0], None

        for regex in (
            r"from\s+([a-zA-Z\s]{3,50})\s+to\s+([a-zA-Z\s]{3,50})",
            r"between\s+([a-zA-Z\s]{3,50})\s+and\s+([a-zA-Z\s]{3,50})",
            r"between\s+([a-zA-Z\s]{3,50})\s+to\s+([a-zA-Z\s]{3,50})",
        ):
            match = re.search(regex, text)
            if match:
                src_raw = match.group(1).strip()
                dst_raw = match.group(2).strip()
                return self._resolve_station_alias(src_raw), self._resolve_station_alias(dst_raw)

        return None, None

    def _resolve_station_alias(self, text: str) -> Optional[str]:
        cleaned = self._normalize_text(text)
        if cleaned in STATION_ALIASES:
            return STATION_ALIASES[cleaned]
        for alias, code in STATION_ALIASES.items():
            if alias in cleaned:
                return code
        if len(cleaned) >= 3:
            return cleaned.upper()[:5]
        return None

    def _guess_station_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        return self._resolve_station_alias(text)

    def _extract_train_number(self, text: str) -> Optional[str]:
        match = re.search(r"\b(1\d{4}|[2-9]\d{4})\b", text)
        return match.group(1) if match else None

    def _extract_class(self, text: str) -> Optional[str]:
        normalized = text.lower()
        for alias, code in sorted(CLASS_ALIASES.items(), key=lambda item: -len(item[0])):
            if alias in normalized:
                return code
        return None

    def _extract_compare_classes(self, text: str) -> List[str]:
        found = []
        for alias, code in sorted(CLASS_ALIASES.items(), key=lambda item: -len(item[0])):
            if alias in text and code not in found:
                found.append(code)
        return found[:4]

    def _extract_compare_trains(self, text: str) -> List[str]:
        numbers = re.findall(r"\b(1\d{4}|[2-9]\d{4})\b", text)
        seen: List[str] = []
        for n in numbers:
            if n not in seen:
                seen.append(n)
        return seen[:3]

    def _extract_passengers(self, text: str) -> Optional[int]:
        match = re.search(r"\b(\d+)\s*(?:passenger|pax|ticket|seat|person|people|traveller|traveler)s?\b", text)
        if match:
            return int(match.group(1))

        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
        }
        for word, value in word_map.items():
            if re.search(rf"\b{word}\b", text):
                return value
        return None

    def _extract_limit_hint(self, text: str) -> Optional[int]:
        for pattern in (r"\btop\s+(\d+)\b", r"\bshow\s+(\d+)\b", r"\bfirst\s+(\d+)\b"):
            match = re.search(pattern, text)
            if match:
                return max(1, min(15, int(match.group(1))))
        return None

    def _extract_sort_hint(self, text: str) -> Optional[str]:
        if any(x in text for x in ["cheapest", "lowest fare", "sort by fare", "by fare"]):
            return "fare"
        if any(x in text for x in ["fastest", "least time", "shortest journey", "earliest"]):
            return "duration"
        if any(x in text for x in ["fewest stops", "least stops", "fewer stops", "direct"]):
            return "stops"
        return None

    def _extract_booking_id(self, text: str) -> Optional[str]:
        match = re.search(r"\bbooking\s*(?:id|no\.?|number)?\s*[:#-]?\s*(\d{1,8})\b", text)
        if match:
            return match.group(1)
        if any(x in text for x in ["cancel", "booking"]):
            alt = re.search(r"\b(\d{1,8})\b", text)
            if alt:
                return alt.group(1)
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        today = datetime.now().date()
        if "today" in text:
            return today.isoformat()
        if "tomorrow" in text:
            return (today + timedelta(days=1)).isoformat()
        if "day after tomorrow" in text:
            return (today + timedelta(days=2)).isoformat()

        ymd = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
        if ymd:
            year, month, day = map(int, ymd.groups())
            try:
                return datetime(year, month, day).date().isoformat()
            except Exception:
                return None

        dmy = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
        if dmy:
            day, month, year = map(int, dmy.groups())
            try:
                return datetime(year, month, day).date().isoformat()
            except Exception:
                return None

        return None

    def _extract_time_hint(self, text: str) -> Optional[str]:
        if "morning" in text:
            return "morning"
        if "afternoon" in text:
            return "afternoon"
        if "evening" in text:
            return "evening"
        if "night" in text:
            return "night"

        match = re.search(r"\b(?:at|after|before)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
        if match:
            hh = int(match.group(1))
            mm = int(match.group(2) or 0)
            ampm = (match.group(3) or "").lower()
            if ampm == "pm" and hh != 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
            return f"{hh:02d}:{mm:02d}"
        return None

    # ------------------------------------------------------------------
    # History / context
    # ------------------------------------------------------------------

    def _build_context(self, history: List[Dict[str, str]], current_text: str) -> ConversationContext:
        ctx = ConversationContext()
        recent = history[-10:]
        for message in reversed(recent):
            text = self._normalize_text(message.get("content", ""))
            s, d = self._extract_stations(text)
            ctx.source = ctx.source or s
            ctx.destination = ctx.destination or d
            ctx.train_number = ctx.train_number or self._extract_train_number(text)
            ctx.travel_class = ctx.travel_class or self._extract_class(text)
            ctx.passengers = ctx.passengers or self._extract_passengers(text)
            ctx.travel_date = ctx.travel_date or self._extract_date(text)
            ctx.time_hint = ctx.time_hint or self._extract_time_hint(text)
            ctx.intent = ctx.intent or self._detect_intent(text)

        current = self._parse_request(current_text, ConversationContext())
        ctx.source = ctx.source or current.source
        ctx.destination = ctx.destination or current.destination
        ctx.train_number = ctx.train_number or current.train_number
        ctx.travel_class = ctx.travel_class or current.travel_class
        ctx.passengers = ctx.passengers or current.passengers
        ctx.travel_date = ctx.travel_date or current.travel_date
        ctx.time_hint = ctx.time_hint or current.time_hint
        ctx.intent = current.intent or ctx.intent
        ctx.last_query_type = current.intent
        return ctx

    def _context_summary(self, context: ConversationContext) -> str:
        parts = []
        if context.source:
            parts.append(f"source={context.source}")
        if context.destination:
            parts.append(f"destination={context.destination}")
        if context.train_number:
            parts.append(f"train={context.train_number}")
        if context.travel_class:
            parts.append(f"class={context.travel_class}")
        if context.passengers:
            parts.append(f"passengers={context.passengers}")
        if context.travel_date:
            parts.append(f"date={context.travel_date}")
        if context.time_hint:
            parts.append(f"time={context.time_hint}")
        return "; ".join(parts) if parts else "none"

    # ------------------------------------------------------------------
    # Search ranking / recommendation
    # ------------------------------------------------------------------

    def _choose_best_train_candidate(self, trains: List[Dict[str, Any]], parsed: ParsedRequest) -> Dict[str, Any]:
        if not trains:
            return {}
        ranked = list(trains)
        if parsed.time_hint:
            filtered = self._apply_time_filter(ranked, parsed.time_hint)
            if filtered:
                ranked = filtered
        if parsed.sort_by:
            ranked = self._sort_trains(ranked, parsed.sort_by)
        else:
            ranked = self._sort_trains(ranked, "duration")
        return ranked[0] if ranked else {}

    def _apply_time_filter(self, trains: List[Dict[str, Any]], time_hint: Optional[str]) -> List[Dict[str, Any]]:
        if not time_hint:
            return trains

        def hour_from_time(value: str) -> Optional[int]:
            if not value:
                return None
            match = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)?", value.lower())
            if not match:
                return None
            hh = int(match.group(1))
            ampm = match.group(3)
            if ampm == "pm" and hh != 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
            return hh

        def departure_hour(train: Dict[str, Any]) -> Optional[int]:
            for key in ("departure", "dep", "start_time", "departure_time"):
                if train.get(key):
                    return hour_from_time(str(train[key]))
            return None

        hint = time_hint.lower()
        if hint == "morning":
            return [t for t in trains if 5 <= (departure_hour(t) or -1) <= 11]
        if hint == "afternoon":
            return [t for t in trains if 12 <= (departure_hour(t) or -1) <= 16]
        if hint == "evening":
            return [t for t in trains if 17 <= (departure_hour(t) or -1) <= 21]
        if hint == "night":
            return [t for t in trains if (departure_hour(t) or -1) >= 22 or (departure_hour(t) or -1) <= 4]

        if re.match(r"^\d{2}:\d{2}$", hint):
            hh = int(hint.split(":")[0])
            return [t for t in trains if abs((departure_hour(t) or -100) - hh) <= 1]

        return trains

    def _sort_trains(self, trains: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        def to_float(value: Any, default: float = 10**9) -> float:
            try:
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    return float(value)
                text = str(value).strip().lower()
                minutes = 0.0
                h = re.search(r"(\d+)\s*h", text)
                m = re.search(r"(\d+)\s*m", text)
                if h:
                    minutes += float(h.group(1)) * 60
                if m:
                    minutes += float(m.group(1))
                if minutes > 0:
                    return minutes
                return float(text)
            except Exception:
                return default

        if sort_by == "fare":
            return sorted(trains, key=lambda x: to_float(x.get("fare") or x.get("estimated_fare") or x.get("min_fare")))
        if sort_by == "duration":
            return sorted(trains, key=lambda x: to_float(x.get("duration") or x.get("journey_time") or x.get("travel_time")))
        if sort_by == "stops":
            return sorted(trains, key=lambda x: to_float(x.get("stops") or x.get("total_stops") or x.get("stop_count")))
        return trains

    def _is_direct_or_low_stop(self, train: Dict[str, Any]) -> bool:
        stops = train.get("stops") or train.get("total_stops") or train.get("stop_count")
        try:
            if stops is None:
                return False
            return int(float(stops)) <= 2
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_train_search(
        self,
        src: str,
        dst: str,
        trains: List[Dict[str, Any]],
        parsed: ParsedRequest,
    ) -> str:
        if not trains:
            return f"No trains were found from **{src}** to **{dst}** in the available data."

        heading = f"Found {len(trains)} train(s) from **{src}** to **{dst}**"
        qualifiers = []
        if parsed.time_hint:
            qualifiers.append(f"time: {parsed.time_hint}")
        if parsed.sort_by:
            qualifiers.append(f"sorted by {parsed.sort_by}")
        if parsed.direct_only:
            qualifiers.append("direct/low-stop preference")
        if qualifiers:
            heading += f" ({', '.join(qualifiers)})"

        lines = [f"🚆 **{heading}**\n"]
        for idx, train in enumerate(trains, start=1):
            train_no = train.get("train_number", "-")
            train_name = train.get("train_name", "")
            dep = train.get("departure") or train.get("dep") or "--:--"
            arr = train.get("arrival") or train.get("arr") or "--:--"
            duration = train.get("duration") or train.get("journey_time") or "N/A"
            stop_count = train.get("stops") or train.get("total_stops") or train.get("stop_count")
            extra = []
            if duration and duration != "N/A":
                extra.append(f"⏱ {duration}")
            if stop_count is not None:
                extra.append(f"🛑 {stop_count} stops")
            extra_text = f" | {' | '.join(extra)}" if extra else ""
            lines.append(f"{idx}. **{train_no}** — {train_name} | {dep} → {arr}{extra_text}")

        lines.append("\nAsk me for the cheapest option, fastest option, route details, fare, or booking help.")
        return "\n".join(lines)

    def _format_single_fare(self, result: Dict[str, Any], src: str, dst: str, pax: int) -> str:
        class_code = result.get("class_code") or result.get("class") or "SL"
        class_name = result.get("class_name", class_code)
        per_ticket = result.get("per_passenger", result.get("fare", 0))
        total = result.get("total_fare", result.get("total", per_ticket * pax))
        distance_km = result.get("distance_km")
        note = "~Estimated fare" if result.get("is_estimated", True) else "From route data"

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
            return f"Could not get fare: {result.get('message', 'Unknown error')}"

        if "fares" in result:
            fare_map = result.get("fares", {})
            lines = [
                f"💰 **Fare estimates** — {src} → {dst}",
                f"🚆 Train: **{result.get('train_number', '-') }** ({result.get('train_name', '')})",
                f"👥 Passengers: **{pax}**\n",
                "| Class | Per Ticket | Total |",
                "|---|---:|---:|",
            ]
            for code in SUPPORTED_CLASSES:
                if code in fare_map:
                    fare = fare_map[code]
                    lines.append(
                        f"| {fare.get('class_name', code)} | ₹{fare.get('per_passenger', 0):,.0f} | ₹{fare.get('total', 0):,.0f} |"
                    )
            distance = self._extract_distance_from_result(result)
            if distance is not None:
                lines.append(f"\n_Fares are estimates calibrated for roughly {distance:.0f} km._")
            else:
                lines.append("\n_Fares are estimates based on route and class._")
            return "\n".join(lines)

        return self._format_single_fare(result, src, dst, pax)

    def _format_fare_comparison_table(self, fares: List[Dict[str, Any]], src: str, dst: str) -> str:
        lines = [
            f"💰 **Fare comparison** — {src} → {dst}",
            "",
            "| Class | Per Ticket | Total | Train |",
            "|---|---:|---:|---|",
        ]
        for item in fares:
            lines.append(
                f"| {item.get('class_name', item.get('class_code', '-'))} | "
                f"₹{float(item.get('per_passenger', 0)):,.0f} | "
                f"₹{float(item.get('total_fare', item.get('total', 0))):,.0f} | "
                f"{item.get('train_number', '-')} |"
            )
        return "\n".join(lines)

    def _extract_distance_from_result(self, result: Dict[str, Any]) -> Optional[float]:
        if isinstance(result.get("distance_km"), (int, float)):
            return float(result["distance_km"])
        fares = result.get("fares", {})
        for item in fares.values():
            if isinstance(item, dict) and isinstance(item.get("distance_km"), (int, float)):
                return float(item["distance_km"])
        return None

    def _format_route(self, result: Dict[str, Any], train_number: str) -> str:
        route = result.get("route", [])
        total_stops = result.get("total_stops", len(route))
        if not route:
            return result.get("message", "Route not found.")

        lines = [f"🗺️ **Route for Train {train_number}** ({total_stops} stops):\n"]
        for stop in route[:25]:
            seq = stop.get("seq", stop.get("stop_no", "-"))
            station = stop.get("station_code", stop.get("station", "-"))
            arrival = stop.get("arrival", "--:--")
            departure = stop.get("departure", "--:--")
            dist = stop.get("distance_km")
            extra = f" ({dist} km)" if dist is not None else ""
            lines.append(f"{seq:>2}. **{station}**  arr:{arrival}  dep:{departure}{extra}")
        if total_stops > 25:
            lines.append(f"...and {total_stops - 25} more stops.")
        return "\n".join(lines)

    def _format_train_info(self, info: Dict[str, Any], route: Dict[str, Any]) -> str:
        if info.get("status") != "ok" and route.get("status") != "ok":
            return "I could not retrieve train information from the available data."

        train_number = info.get("train_number") or route.get("train_number") or "-"
        train_name = info.get("train_name") or "-"
        src = info.get("source") or info.get("from") or "-"
        dst = info.get("destination") or info.get("to") or "-"
        days = info.get("days") or info.get("running_days") or info.get("day_pattern")
        duration = info.get("duration") or info.get("journey_time")
        lines = [
            f"🚆 **Train {train_number}** — {train_name}",
            f"📍 Route: {src} → {dst}",
        ]
        if duration:
            lines.append(f"⏱ Journey time: {duration}")
        if days:
            lines.append(f"🗓 Running pattern: {days}")
        if route.get("status") == "ok" and route.get("total_stops") is not None:
            lines.append(f"🛑 Total stops: {route.get('total_stops')}")
        if route.get("status") == "ok" and route.get("route"):
            first = route["route"][0].get("station_code", "-")
            last = route["route"][-1].get("station_code", "-")
            lines.append(f"📌 First/last stop: {first} → {last}")
        return "\n".join(lines)

    def _format_station_info(self, result: Dict[str, Any]) -> str:
        return (
            f"🏠 **{result.get('station_name', '-') }** ({result.get('station_code', '-')})\n"
            f"📍 City: {result.get('city', 'Unknown')}\n"
            f"ℹ️ Type: {result.get('station_type', 'Railway station')}"
        )

    def _format_booking_history(self, result: Dict[str, Any]) -> str:
        bookings = result.get("bookings", [])
        lines = [f"📋 **Your bookings ({result.get('count', len(bookings))} total)**:\n"]
        for booking in bookings:
            icon = "✅" if str(booking.get("status", "")).upper() == "CONFIRMED" else "❌"
            lines.append(
                f"{icon} **#{booking.get('booking_id', '-') }** — Train {booking.get('train_number', '-') } | "
                f"{booking.get('class', '-') } × {booking.get('passengers', '-') } | "
                f"{str(booking.get('travel_date', ''))[:10]} | {booking.get('status', '-') }"
            )
        return "\n".join(lines)

    def _format_booking_confirmation(self, result: Dict[str, Any]) -> str:
        return (
            f"✅ **Booking confirmed**\n\n"
            f"🆔 Booking ID: **{result.get('booking_id', '-') }**\n"
            f"🚆 Train: **{result.get('train_number', '-') }**\n"
            f"📍 {result.get('source', '-') } → {result.get('destination', '-') }\n"
            f"🎫 Class: **{result.get('class', '-') }** | 👥 {result.get('passengers', '-') } pax\n"
            f"📅 Date: **{result.get('travel_date', '-') }**\n"
            f"💰 Est. fare: ₹{float(result.get('estimated_total_fare', 0)):,.0f}"
        )

    def _booking_clarification_message(self, missing: List[str], raw: str) -> str:
        labels = {
            "source": "source station",
            "destination": "destination station",
            "class": "travel class",
            "passengers": "number of passengers",
            "date": "travel date",
        }
        readable = ", ".join(labels.get(x, x) for x in missing)
        return f"I need the {readable} before I can continue with booking."

    def _clarify_missing_route(self, raw_text: str) -> str:
        return (
            "Please tell me the source and destination stations.\n"
            "Example: **Show trains from Bangalore to Mangalore**"
        )

    def _fallback_help_message(self) -> str:
        return (
            "I can help with train search, fares, route details, station info, and demo bookings.\n\n"
            "Examples:\n"
            "• Show trains from Bangalore to Mangalore\n"
            "• What is the sleeper fare for train 16585?\n"
            "• Tell me the route of train 12627"
        )

    def _greeting_message(self) -> str:
        return (
            "Hello! I’m **RailMitra**.\n\n"
            "Try asking:\n"
            "• Show trains from Bangalore to Mangalore\n"
            "• What is the sleeper fare for train 16585?\n"
            "• Tell me the route of train 12627"
        )

    def _friendly_tool_error(self, action: str, result: Dict[str, Any]) -> str:
        msg = result.get("message") or result.get("error") or "Unknown error"
        return f"I could not complete the {action}. {msg}"

    # ------------------------------------------------------------------
    # Safety / utilities
    # ------------------------------------------------------------------

    def _normalize_class_code(self, value: str) -> str:
        v = value.strip().lower()
        return CLASS_ALIASES.get(v, value.strip().upper())

    def _wants_all_classes(self, text: str) -> bool:
        return any(x in text.lower() for x in ["all classes", "show fare for all classes", "fare for all", "compare fare", "compare fares"])

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

    def _sanitize_output(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"hf_[A-Za-z0-9]{20,}", "[redacted]", cleaned)
        return cleaned

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _default_travel_date(self) -> str:
        return datetime.now().date().isoformat()

    def estimate_demo_fare(
        self,
        distance_km: float,
        class_code: str,
        train_category_multiplier: float = 1.0,
    ) -> int:
        """Optional public helper for other modules."""
        class_code = self._normalize_class_code(class_code)
        base = max(1.0, float(distance_km)) * float(os.environ.get("FARE_BASE_PER_KM", "0.82"))
        class_multiplier = {
            "GN": 0.55, "2S": 0.7, "SL": 1.0, "CC": 1.25, "3A": 1.9, "2A": 2.8, "1A": 4.4, "EC": 4.8
        }.get(class_code, 1.0)
        fare = base * class_multiplier * max(0.75, float(train_category_multiplier))
        service_floor = 25 if class_code in {"GN", "2S"} else 60
        return int(round(max(fare + service_floor, service_floor)))


def build_agent_service() -> AgentService:
    return AgentService()
