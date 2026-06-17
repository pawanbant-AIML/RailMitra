"""
agent/agent_service.py – Core LLM agent using Hugging Face Llama 3.1 8B Instruct.

Architecture:
  - Uses LangChain's tool-calling agent via HuggingFace Inference API (serverless).
  - Falls back to a structured rule-based handler if the LLM is unavailable.
  - Maintains conversation history from the full messages array passed by the frontend.
  - All tools are defined in agent/tools.py and injected per-request with the DB session.

Design decisions:
  - We use the HuggingFace text-generation-inference (TGI) API directly via requests
    for reliability, with a LangChain wrapper for tool routing.
  - System prompt is carefully crafted to prevent hallucination and guide tool use.
  - Max retries = 2, timeout = 25s to stay within Render's 30s request limit.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from app.agent.tools import AgentTools
from app.core.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions"
)

SYSTEM_PROMPT = """You are RailMitra, a friendly and knowledgeable Indian Railways AI assistant. \
You help users search for trains, check fares, view routes, book tickets, cancel bookings, and answer questions about Indian Railways.

## RULES
1. ALWAYS use the available tools to answer questions - never make up train numbers, fares, or station codes.
2. When the user asks about trains between two cities, call `search_trains` first.
3. When asked about fares, call `get_fare` with the train number from the search results.
4. For bookings, always confirm the class and passenger count before calling `book_ticket`.
5. If the user's query is ambiguous (no source/destination), ask a clarifying question instead of guessing.
6. Respond in friendly, conversational English. Use emojis sparingly (1-2 per response max).
7. Format train lists as clean bullet points. Format fares as a table when showing multiple classes.
8. If a tool returns an error, apologise briefly and suggest alternatives.
9. Remember context from the conversation - if the user says "that one" or "the first train", refer to what was shown.
10. NEVER invent station codes, train numbers, or fares. Only use data returned by tools.

## AVAILABLE TOOLS
- `search_trains(source, destination, date)` - Find trains between two stations
- `get_fare(train_number, source, destination, travel_class, passengers)` - Get fare estimates
- `get_train_route(train_number)` - Get the full route/schedule of a train
- `book_ticket(source, destination, travel_class, passengers, travel_date, train_number)` - Book a ticket
- `cancel_booking(booking_id)` - Cancel a booking
- `get_booking_history(user_id)` - Show booking history
- `get_station_info(station)` - Get station details

## RESPONSE STYLE
- Be concise but friendly. Don't pad responses unnecessarily.
- When listing trains, show train number, name, and the from/to codes.
- When showing fares, clearly indicate if they are estimates.
- Always end booking confirmations with the booking ID so the user can reference it."""


# ---------------------------------------------------------------------------
# AgentService
# ---------------------------------------------------------------------------

class AgentService:
    """
    Manages the full request-response cycle for one chat turn.

    Usage:
        svc = AgentService()
        reply = svc.run(user_message, conversation_history, db)
    """

    def __init__(self):
        self.hf_token = (
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or ""
        )
        self.max_tool_iterations = 5
        self.timeout = 25  # seconds

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        db: Session,
    ) -> str:
        """
        Process one user turn.

        Args:
            user_message:        The latest user message text.
            conversation_history: List of {"role": ..., "content": ...} dicts
                                  NOT including the current user message.
            db:                  SQLAlchemy session for DB access.

        Returns:
            The assistant's reply as a plain string (may include markdown).
        """
        logger.info(f"[agent] Processing: {user_message[:100]!r}")

        if not self.hf_token:
            logger.warning("[agent] No HF token - falling back to structured handler")
            return self._fallback_handler(user_message, conversation_history, db)

        agent_tools = AgentTools(db)
        tools = agent_tools.build()
        tool_map = {t.name: t for t in tools}

        # Build the full message list for the LLM
        messages = self._build_messages(user_message, conversation_history)

        # Build tool schemas for the API call
        tool_schemas = self._build_tool_schemas(tools)

        # Agentic loop: up to max_tool_iterations rounds of tool calling
        for iteration in range(self.max_tool_iterations):
            logger.info(f"[agent] LLM call iteration {iteration + 1}")
            response = self._call_llm(messages, tool_schemas)

            if response is None:
                logger.warning("[agent] LLM call failed - using fallback")
                return self._fallback_handler(user_message, conversation_history, db)

            # Check if the model wants to call tools
            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""

            if not tool_calls:
                # Final answer from the model
                logger.info(f"[agent] Final answer: {content[:100]!r}")
                return content.strip() if content.strip() else self._fallback_handler(
                    user_message, conversation_history, db
                )

            # Execute each requested tool call
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", f"call_{tool_name}")

                logger.info(f"[agent] Tool call: {tool_name}({raw_args[:120]})")

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}

                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(args)
                    except Exception as e:
                        tool_result = json.dumps({"status": "error", "message": str(e)})
                else:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": f"Unknown tool: {tool_name}"
                    })

                logger.info(f"[agent] Tool result: {str(tool_result)[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": str(tool_result),
                })

        # Exhausted iterations - return last content or fallback
        logger.warning("[agent] Max iterations reached")
        last_content = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            ""
        )
        return last_content.strip() or self._fallback_handler(
            user_message, conversation_history, db
        )

    # ------------------------------------------------------------------ #
    # LLM call via HuggingFace Inference API                              #
    # ------------------------------------------------------------------ #

    def _call_llm(
        self,
        messages: List[Dict],
        tools: List[Dict],
        retries: int = 2,
    ) -> Optional[Dict]:
        """
        Call the Llama 3.1 8B Instruct model via HF Inference API.
        Returns the assistant message dict or None on failure.
        """
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    return choice.get("message", {})

                elif resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"[agent] Rate limited, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)

                elif resp.status_code in (503, 500):
                    logger.warning(f"[agent] HF server error {resp.status_code}, attempt {attempt+1}")
                    time.sleep(2)

                else:
                    logger.error(f"[agent] HF API error {resp.status_code}: {resp.text[:300]}")
                    return None

            except requests.Timeout:
                logger.warning(f"[agent] Timeout on attempt {attempt+1}")
                if attempt == retries:
                    return None

            except Exception as exc:
                logger.error(f"[agent] Request exception: {exc}")
                return None

        return None

    # ------------------------------------------------------------------ #
    # Message construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        user_message: str,
        history: List[Dict[str, str]],
    ) -> List[Dict]:
        """Build the full messages array for the LLM, respecting token budget."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include last N turns of history to stay within token budget
        max_history_turns = 8
        recent = history[-(max_history_turns * 2):]
        for h in recent:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------ #
    # Tool schema builder (OpenAI-compatible format)                      #
    # ------------------------------------------------------------------ #

    def _build_tool_schemas(self, tools: list) -> List[Dict]:
        """
        Convert LangChain tools to OpenAI-compatible function schemas
        for the HuggingFace API.
        """
        schemas = []
        for t in tools:
            schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.model_json_schema()
                    if hasattr(t, "args_schema") and t.args_schema
                    else {"type": "object", "properties": {}},
                },
            }
            schemas.append(schema)
        return schemas

    # ------------------------------------------------------------------ #
    # Structured fallback (no LLM)                                        #
    # ------------------------------------------------------------------ #

    def _fallback_handler(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        db: Session,
    ) -> str:
        """
        Rule-based fallback when the LLM is unavailable.
        Extracts intent and entities from the message using simple heuristics
        and calls the appropriate tool directly.
        """
        agent_tools = AgentTools(db)
        msg = user_message.lower().strip()

        # ── Resolve context from history ──────────────────────────────
        ctx_src, ctx_dst, ctx_train = self._extract_context(history)

        # ── Entity extraction ──────────────────────────────────────────
        src, dst = self._extract_stations(msg)
        src = src or ctx_src
        dst = dst or ctx_dst
        train_num = self._extract_train_number(msg) or ctx_train
        cls = self._extract_class(msg)
        pax = self._extract_passengers(msg)

        # ── Route: book  ──────────────────────────────────────────────
        if any(w in msg for w in ["book", "reserve", "ticket", "buy"]):
            if src and dst:
                result = json.loads(agent_tools.book_ticket(
                    src, dst, cls or "SL", pax or 1
                ))
                if result.get("status") == "confirmed":
                    return (
                        f"✅ **Booking Confirmed!**\n\n"
                        f"🆔 Booking ID: **{result['booking_id']}**\n"
                        f"🚆 Train: **{result['train_number']}**\n"
                        f"📍 {result['source']} → {result['destination']}\n"
                        f"🎫 Class: **{result['class']}** | 👥 {result['passengers']} pax\n"
                        f"📅 Date: **{result['travel_date']}**\n"
                        f"💰 Est. fare: ₹{result['estimated_total_fare']:,.0f}"
                    )
                return f"❌ Booking failed: {result.get('message', 'Unknown error')}"
            return "To book a ticket, please tell me the source and destination stations."

        # ── Cancel booking ────────────────────────────────────────────
        if "cancel" in msg:
            bid = re.search(r'\b(\d{1,6})\b', msg)
            if bid:
                result = json.loads(agent_tools.cancel_booking(int(bid.group(1))))
                return result.get("message", "Cancellation processed.")
            return "Please provide your Booking ID to cancel (e.g. 'Cancel booking 42')."

        # ── Booking history ───────────────────────────────────────────
        if any(w in msg for w in ["my booking", "history", "my ticket", "reservations"]):
            result = json.loads(agent_tools.get_booking_history())
            if result.get("status") == "empty":
                return "📭 You have no bookings yet. Search for trains to get started!"
            lines = [f"📋 **Your Bookings ({result['count']} total):**\n"]
            for b in result.get("bookings", []):
                icon = "✅" if b["status"] == "CONFIRMED" else "❌"
                lines.append(f"{icon} **#{b['booking_id']}** – Train {b['train_number']} | "
                             f"{b['class']} × {b['passengers']} | {b['travel_date'][:10]} | {b['status']}")
            return "\n".join(lines)

        # ── Train route ───────────────────────────────────────────────
        if any(w in msg for w in ["route", "stop", "schedule", "station"]) and train_num:
            result = json.loads(agent_tools.get_train_route(train_num))
            if result.get("status") == "ok":
                stops = result["route"][:15]
                lines = [f"🗺️ **Route for Train {train_num}** ({result['total_stops']} stops):\n"]
                for s in stops:
                    lines.append(f"  {s['seq']:>2}. **{s['station_code']}**  "
                                 f"arr:{s['arrival']}  dep:{s['departure']}"
                                 + (f"  ({s['distance_km']} km)" if s.get('distance_km') else ""))
                if result['total_stops'] > 15:
                    lines.append(f"  ...and {result['total_stops'] - 15} more stops.")
                return "\n".join(lines)
            return result.get("message", "Route not found.")

        # ── Fare query ────────────────────────────────────────────────
        if any(w in msg for w in ["fare", "cost", "price", "how much", "charge"]):
            if train_num and src and dst:
                result = json.loads(agent_tools.get_fare(
                    train_num, src, dst, cls or "ALL", pax or 1
                ))
                return self._format_fare_result(result, src, dst, pax or 1)
            elif src and dst:
                # Search for a train first
                trains_raw = json.loads(agent_tools.search_trains(src, dst))
                if trains_raw.get("status") == "ok" and trains_raw["trains"]:
                    first = trains_raw["trains"][0]
                    tn = first["train_number"]
                    result = json.loads(agent_tools.get_fare(
                        tn, src, dst, cls or "ALL", pax or 1
                    ))
                    return self._format_fare_result(result, src, dst, pax or 1)
            return "Please tell me the source, destination, and (optionally) travel class to check fares."

        # ── Train search (default) ────────────────────────────────────
        if src and dst:
            result = json.loads(agent_tools.search_trains(src, dst))
            if result.get("status") == "no_results":
                return (f"😔 No trains found from **{src}** to **{dst}**. "
                        "Please check the station names or try nearby stations.")
            trains = result.get("trains", [])
            if not trains:
                return f"I couldn't find any trains from {src} to {dst}."
            lines = [
                f"🚂 Found **{result['count']} train(s)** from "
                f"**{result['source_resolved']}** → **{result['destination_resolved']}**:\n"
            ]
            for t in trains:
                lines.append(f"• **{t['train_number']}** – {t['train_name']}  ({t['from']} → {t['to']})")
            lines.append("\n💡 Ask for fares, route, or to book any of these trains.")
            return "\n".join(lines)

        # ── Station info ──────────────────────────────────────────────
        if any(w in msg for w in ["station", "about", "tell me"]):
            words = user_message.split()
            for w in words:
                if len(w) >= 3:
                    result = json.loads(agent_tools.get_station_info(w))
                    if result.get("status") == "ok":
                        s = result
                        return (f"🏠 **{s['station_name']}** ({s['station_code']})\n"
                                f"📍 City: {s['city']}")

        # ── Greeting ──────────────────────────────────────────────────
        if any(w in msg for w in ["hi", "hello", "hey", "namaste", "help"]):
            return ("👋 Hello! I'm **RailMitra**, your Indian Railways assistant.\n\n"
                    "I can help you:\n"
                    "• 🔍 Search trains between any two stations\n"
                    "• 💰 Check realistic fare estimates for all classes\n"
                    "• 🗺️ View train routes and schedules\n"
                    "• 🎫 Book or cancel tickets\n"
                    "• 📋 View your booking history\n\n"
                    "Just ask me something like:\n"
                    "*\"Show trains from Bangalore to Mangalore\"*")

        # ── Default ───────────────────────────────────────────────────
        return ("I'm not sure I understood that. Try asking me:\n"
                "• *\"Find trains from Bangalore to Mangalore\"*\n"
                "• *\"What is the sleeper fare for train 16585?\"*\n"
                "• *\"Book 2 tickets from SBC to MAQ\"*")

    # ------------------------------------------------------------------ #
    # Helper: entity extraction for fallback                              #
    # ------------------------------------------------------------------ #

    _STATION_ALIASES = {
        "bangalore": "SBC", "bengaluru": "SBC", "blr": "SBC", "sbc": "SBC",
        "yesvantpur": "YPR", "ypr": "YPR",
        "mysore": "MYS", "mysuru": "MYS", "mys": "MYS",
        "hubli": "UBL", "hubballi": "UBL", "ubl": "UBL",
        "mangalore": "MAQ", "mangaluru": "MAQ", "maq": "MAQ",
        "mumbai": "CSMT", "bombay": "CSMT", "csmt": "CSMT",
        "delhi": "NDLS", "new delhi": "NDLS", "ndls": "NDLS",
        "chennai": "MAS", "madras": "MAS", "mas": "MAS",
        "kolkata": "HWH", "howrah": "HWH", "hwh": "HWH",
        "hyderabad": "HYB", "hyd": "HYB", "secunderabad": "SC",
        "pune": "PUNE", "goa": "MAO", "udupi": "UD",
        "hassan": "HAS", "shimoga": "SMET", "davangere": "DVG",
        "kochi": "ERS", "ernakulam": "ERS", "ers": "ERS",
        "trivandrum": "TVC", "thiruvananthapuram": "TVC", "tvc": "TVC",
        "coimbatore": "CBE", "madurai": "MDU",
        "ahmedabad": "ADI", "surat": "ST", "vadodara": "BRC",
        "jaipur": "JP", "jodhpur": "JU", "udaipur": "UDZ",
        "lucknow": "LKO", "varanasi": "BSB", "patna": "PNBE",
        "bhopal": "BPL", "nagpur": "NGP", "indore": "INDB",
        "visakhapatnam": "VSKP", "vizag": "VSKP",
        "amritsar": "ASR", "chandigarh": "CDG", "ludhiana": "LDH",
        "guwahati": "GHY", "bhubaneswar": "BBS",
    }

    _PREPOSITIONS = ["from", "between", "to", "towards", "via", "and", "going to", "arriving at"]

    def _extract_stations(self, msg: str) -> tuple:
        """Very lightweight station extractor for the fallback path."""
        src, dst = None, None
        # Longest-match on alias keys
        sorted_aliases = sorted(self._STATION_ALIASES.keys(), key=len, reverse=True)
        found = []
        for alias in sorted_aliases:
            if alias in msg and self._STATION_ALIASES[alias] not in found:
                found.append(self._STATION_ALIASES[alias])
                if len(found) == 2:
                    break
        if len(found) >= 2:
            return found[0], found[1]
        if len(found) == 1:
            return found[0], None
        return None, None

    def _extract_train_number(self, msg: str) -> Optional[str]:
        m = re.search(r'\b(1\d{4}|[2-9]\d{4})\b', msg)
        return m.group(1) if m else None

    def _extract_class(self, msg: str) -> Optional[str]:
        mapping = {
            "sleeper": "SL", "sl": "SL",
            "general": "GN", "gn": "GN",
            "3ac": "3A", "3a": "3A", "third ac": "3A",
            "2ac": "2A", "2a": "2A", "second ac": "2A",
            "1ac": "1A", "1a": "1A", "first ac": "1A",
            "chair car": "CC", "cc": "CC",
            "executive": "EC", "ec": "EC",
        }
        for k, v in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if k in msg:
                return v
        return None

    def _extract_passengers(self, msg: str) -> Optional[int]:
        m = re.search(r'\b(\d+)\s*(?:passenger|pax|ticket|seat|person|people)\b', msg)
        if m:
            return int(m.group(1))
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4,
                    "five": 5, "six": 6, "seven": 7, "eight": 8}
        for w, n in word_map.items():
            if w in msg:
                return n
        return None

    def _extract_context(self, history: List[Dict]) -> tuple:
        """Extract the most recent source, destination, train from conversation history."""
        src, dst, train = None, None, None
        for msg in reversed(history[-6:]):
            text = msg.get("content", "").lower()
            if not src or not dst:
                s, d = self._extract_stations(text)
                src = src or s
                dst = dst or d
            if not train:
                train = self._extract_train_number(text)
        return src, dst, train

    # ------------------------------------------------------------------ #
    # Helper: format fare result                                          #
    # ------------------------------------------------------------------ #

    def _format_fare_result(self, result: dict, src: str, dst: str, pax: int) -> str:
        if result.get("status") == "error":
            return f"❌ Could not get fare: {result.get('message')}"

        if "fares" in result:
            # All-class result
            lines = [
                f"💰 **Fare Estimates** – {src} → {dst}",
                f"🚆 Train: **{result['train_number']}** ({result.get('train_name', '')})",
                f"👥 Passengers: **{pax}**\n",
                "| Class | Per Ticket | Total |",
                "|-------|-----------|-------|",
            ]
            order = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]
            fares = result["fares"]
            for cls in order:
                if cls in fares:
                    f = fares[cls]
                    lines.append(
                        f"| {f['class_name']} | ₹{f['per_passenger']:,.0f} | ₹{f['total']:,.0f} |"
                    )
            lines.append(f"\n_Fares are estimates based on ~{fares.get('SL', {}).get('distance_km', 'N/A')} km._")
            return "\n".join(lines)

        # Single class result
        return (
            f"💰 **Fare for {result.get('class_name', result.get('class_code'))}** "
            f"({src} → {dst})\n"
            f"🚆 Train: **{result['train_number']}** | {result.get('distance_km', 'N/A')} km\n"
            f"Per ticket: **₹{result['per_passenger']:,.0f}**\n"
            f"Total ({pax} pax): **₹{result['total_fare']:,.0f}**\n"
            f"_{'~Estimated fare' if result.get('is_estimated') else 'From route data'}_"
        )
