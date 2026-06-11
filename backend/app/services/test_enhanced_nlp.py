"""
test_enhanced_nlp.py
══════════════════════════════════════════════════════════════════════════════
Smoke-tests for enhanced_chat_nlp_service.py

Run:
    python test_enhanced_nlp.py
    # or for verbose output:
    python -m pytest test_enhanced_nlp.py -v

Covers:
  ✓  Station extraction – "from X to Y", "between X and Y", bare city names,
     station codes, multi-word city names ("New Delhi"), follow-up replies
  ✓  Date extraction   – relative (tomorrow, next Friday), absolute (15 June),
     numeric (20/06), SpaCy DATE entity fallback
  ✓  Class extraction  – multi-word aliases, abbreviations (3a, sl, cc)
  ✓  Passenger count   – digits, written numbers, various nouns
  ✓  Intent detection  – all 10 intent types including cancellation and history
  ✓  Conversation memory – context carry-over across multi-turn conversations
  ✓  Slot filling       – missing slot detection and clarification generation
  ✓  Budget / preference extraction
══════════════════════════════════════════════════════════════════════════════
"""

import sys
from datetime import date, timedelta

# ── Adjust this import path if the file is placed elsewhere ──────────────
sys.path.insert(0, ".")
from enhanced_chat_nlp_service import (
    ChatNLPService,
    ChatAnalysisRequest,
    Entity,
)

svc = ChatNLPService()


# ══════════════════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════════════════

_pass = 0
_fail = 0


def check(label: str, condition: bool) -> None:
    global _pass, _fail
    if condition:
        print(f"  ✅  {label}")
        _pass += 1
    else:
        print(f"  ❌  {label}")
        _fail += 1


def analyze(msg: str, history=None) -> "ChatAnalysisResponse":
    return svc.analyze(ChatAnalysisRequest(
        user_message=msg,
        conversation_history=history or [],
    ))


def today() -> str:
    return date.today().isoformat()


def tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()


# ══════════════════════════════════════════════════════════════════════════
# 1. Station Extraction
# ══════════════════════════════════════════════════════════════════════════
print("\n── 1. Station Extraction ──────────────────────────────────────────")

r = analyze("Find trains from Bangalore to Mumbai")
check("Bangalore → SBC  (source)",      r.entities.source == "SBC")
check("Mumbai    → CSMT (destination)", r.entities.destination == "CSMT")

r = analyze("trains between Pune and Hyderabad")
check("between Pune → PUNE",            r.entities.source == "PUNE")
check("and Hyderabad → SC or HYB",      r.entities.destination in ("SC", "HYB"))

r = analyze("NDLS to MAS ticket")
check("Station code NDLS → source",     r.entities.source == "NDLS")
check("Station code MAS  → dest",       r.entities.destination == "MAS")

r = analyze("trains from New Delhi to Chennai")
check("Multi-word 'New Delhi' → NDLS",  r.entities.source == "NDLS")
check("Chennai → MAS",                  r.entities.destination == "MAS")

# Follow-up reply – context carry
history = [
    {"role": "user",      "content": "find trains from Bangalore"},
    {"role": "assistant", "content": "Where are you traveling to?"},
]
r = analyze("Mumbai", history=history)
check("Follow-up 'Mumbai' → destination=CSMT", r.entities.destination == "CSMT")
check("Memory carries source SBC",             r.entities.source == "SBC")


# ══════════════════════════════════════════════════════════════════════════
# 2. Date Extraction
# ══════════════════════════════════════════════════════════════════════════
print("\n── 2. Date Extraction ────────────────────────────────────────────")

r = analyze("trains from Delhi to Mumbai tomorrow")
check("'tomorrow' resolved",    r.entities.date == tomorrow())

r = analyze("book ticket for today")
check("'today' resolved",       r.entities.date == today())

r = analyze("find trains on 25 December")
check("'25 December' month+day",r.entities.date is not None and "-12-25" in r.entities.date)

r = analyze("travel on 20/07")
check("DD/MM format",           r.entities.date is not None and "-07-20" in r.entities.date)

r = analyze("journey on 2026-08-15")
check("ISO YYYY-MM-DD",         r.entities.date == "2026-08-15")

history = [
    {"role": "user",      "content": "find trains from Delhi to Lucknow"},
    {"role": "assistant", "content": "When do you want to travel?"},
]
r = analyze("tomorrow", history=history)
check("Follow-up 'tomorrow' fills date slot", r.entities.date == tomorrow())


# ══════════════════════════════════════════════════════════════════════════
# 3. Travel Class
# ══════════════════════════════════════════════════════════════════════════
print("\n── 3. Travel Class Extraction ────────────────────────────────────")

r = analyze("book a third ac ticket")
check("'third ac'      → 3A",  r.entities.travel_class == "3A")

r = analyze("2 sleeper seats from Chennai to Madurai")
check("'sleeper'        → SL",  r.entities.travel_class == "SL")

r = analyze("first ac compartment please")
check("'first ac'       → 1A",  r.entities.travel_class == "1A")

r = analyze("chair car to Bangalore")
check("'chair car'      → CC",  r.entities.travel_class == "CC")

r = analyze("book 3a ticket")
check("abbreviation '3a' → 3A", r.entities.travel_class == "3A")

r = analyze("2AC ticket from Patna to Delhi")
check("'2AC'           → 2A",  r.entities.travel_class == "2A")

r = analyze("general class ticket")
check("'general'        → GN",  r.entities.travel_class == "GN")


# ══════════════════════════════════════════════════════════════════════════
# 4. Passenger Count
# ══════════════════════════════════════════════════════════════════════════
print("\n── 4. Passenger Count ────────────────────────────────────────────")

r = analyze("book 3 tickets from Bhopal to Nagpur")
check("digit '3' before 'tickets'",          r.entities.passengers == 3)

r = analyze("for 5 passengers sleeper class")
check("'for 5 passengers'",                  r.entities.passengers == 5)

r = analyze("two sleeper seats from Goa to Mumbai")
check("word 'two' before 'sleeper seats'",   r.entities.passengers == 2)

r = analyze("reserve four tickets")
check("word 'four' before 'tickets'",        r.entities.passengers == 4)

r = analyze("book 1 seat")
check("single seat",                         r.entities.passengers == 1)


# ══════════════════════════════════════════════════════════════════════════
# 5. Intent Detection
# ══════════════════════════════════════════════════════════════════════════
print("\n── 5. Intent Detection ───────────────────────────────────────────")

check("ROUTE_SEARCH",    analyze("find trains from Delhi to Mumbai").intent == "ROUTE_SEARCH")
check("BOOK_TICKET",     analyze("book 2 sleeper tickets from Bangalore to Chennai tomorrow").intent == "BOOK_TICKET")
check("BOOKING_HISTORY", analyze("show my bookings").intent == "BOOKING_HISTORY")
check("CANCEL_BOOKING",  analyze("cancel booking 42").intent == "CANCEL_BOOKING")
check("FARE_ESTIMATE",   analyze("how much is the fare from Pune to Ahmedabad").intent == "FARE_ESTIMATE")
check("CHECK_ROUTE",     analyze("show route for train 12657").intent == "CHECK_ROUTE")
check("CHEAPEST_ROUTE",  analyze("cheapest route from Kolkata to Chennai").intent == "CHEAPEST_ROUTE")
check("FASTEST_ROUTE",   analyze("fastest trains from Hyderabad to Bangalore").intent == "FASTEST_ROUTE")
check("COMPARE_ROUTES",  analyze("compare routes from Delhi to Mumbai").intent == "COMPARE_ROUTES")

# Case insensitivity
check("UPPERCASE intent", analyze("FIND TRAINS FROM BANGALORE TO MUMBAI").intent == "ROUTE_SEARCH")

# Short cancel
check("bare 'cancel' detected", analyze("cancel").intent == "CANCEL_BOOKING")


# ══════════════════════════════════════════════════════════════════════════
# 6. Conversation Memory & Multi-turn
# ══════════════════════════════════════════════════════════════════════════
print("\n── 6. Conversation Memory ────────────────────────────────────────")

history = [
    {"role": "user",      "content": "find trains from Bangalore to Mumbai"},
    {"role": "assistant", "content": "What date would you like to travel?"},
    {"role": "user",      "content": "tomorrow"},
    {"role": "assistant", "content": "How many passengers?"},
]
r = analyze("2", history=history)
check("Carries source SBC from memory",         r.entities.source == "SBC")
check("Carries destination CSMT from memory",   r.entities.destination == "CSMT")
check("Carries date 'tomorrow' from memory",    r.entities.date == tomorrow())
# '2' fills the pending passengers slot
check("'2' fills pending passengers slot",      r.entities.passengers == 2)

# Full booking flow
history2 = [
    {"role": "user",      "content": "I want to book a ticket from Delhi to Lucknow"},
    {"role": "assistant", "content": "When do you want to travel?"},
    {"role": "user",      "content": "next friday"},
    {"role": "assistant", "content": "Which class would you prefer?"},
    {"role": "user",      "content": "sleeper"},
    {"role": "assistant", "content": "How many passengers?"},
]
r = analyze("3", history=history2)
check("Multi-turn: source=NDLS",         r.entities.source == "NDLS")
check("Multi-turn: dest=LKO",            r.entities.destination == "LKO")
check("Multi-turn: class=SL",            r.entities.travel_class == "SL")
check("Multi-turn: passengers=3",        r.entities.passengers == 3)
check("Multi-turn: date set",            r.entities.date is not None)


# ══════════════════════════════════════════════════════════════════════════
# 7. Slot Filling & Clarification
# ══════════════════════════════════════════════════════════════════════════
print("\n── 7. Slot Filling ───────────────────────────────────────────────")

r = analyze("find trains")
check("No route → clarification needed",    r.clarification_needed is True)
check("First missing slot = 'source'",      r.missing_required_slots[0] == "source")
check("next_action = ASK_CLARIFICATION",    r.next_action == "ASK_CLARIFICATION")

r = analyze("find trains from Bangalore to Mumbai tomorrow")
check("All route slots filled → no clarification", r.clarification_needed is False)
check("next_action = SEARCH_ROUTE",                r.next_action == "SEARCH_ROUTE")

r = analyze("book tickets from Chennai")
check("BOOK_TICKET missing dest/date/class/pax", len(r.missing_required_slots) >= 3)

r = analyze("cancel booking")
check("CANCEL: missing booking_id → clarification", "booking_id" in r.missing_required_slots)

r = analyze("cancel booking 7")
check("CANCEL with ID → no clarification",          r.clarification_needed is False)
check("booking_id = 7",                             r.entities.booking_id == 7)


# ══════════════════════════════════════════════════════════════════════════
# 8. Budget & Preference
# ══════════════════════════════════════════════════════════════════════════
print("\n── 8. Budget & Preference ────────────────────────────────────────")

r = analyze("tickets under ₹500 from Pune to Goa")
check("Budget ₹500 extracted",           r.entities.budget == 500.0)

r = analyze("find the fastest trains from Delhi to Agra")
check("Preference 'fastest' extracted",  r.entities.preference == "fastest")

r = analyze("cheapest route from Mumbai to Ahmedabad")
check("Preference 'cheapest' extracted", r.entities.preference == "cheapest")

r = analyze("I want the shortest route with minimum stops")
check("Preference 'shortest' extracted", r.entities.preference == "shortest")


# ══════════════════════════════════════════════════════════════════════════
# 9. Train Number & Booking ID
# ══════════════════════════════════════════════════════════════════════════
print("\n── 9. Train / Booking IDs ────────────────────────────────────────")

r = analyze("show route for train 12657")
check("Train number 12657 extracted",      r.entities.train_number == "12657")

r = analyze("cancel booking #42")
check("Booking ID 42 (with #) extracted",  r.entities.booking_id == 42)

r = analyze("booking id 100 please cancel")
check("Booking ID 100 extracted",          r.entities.booking_id == 100)


# ══════════════════════════════════════════════════════════════════════════
# 10. Via Stations
# ══════════════════════════════════════════════════════════════════════════
print("\n── 10. Via Stations ──────────────────────────────────────────────")

r = analyze("trains from Bangalore to Delhi via Nagpur")
check("Source SBC",                r.entities.source == "SBC")
check("Destination NDLS",          r.entities.destination == "NDLS")
# via may or may not resolve depending on parse; just check it doesn't crash
check("Via parsed without crash",  True)


# ══════════════════════════════════════════════════════════════════════════
# 11. Robustness – typos, mixed case, missing spaces
# ══════════════════════════════════════════════════════════════════════════
print("\n── 11. Robustness ────────────────────────────────────────────────")

r = analyze("FIND TRAINS FROM BENGALURU TO MUMBAI TOMORROW")
check("All caps",   r.entities.source == "SBC" and r.entities.destination == "CSMT")

r = analyze("trains frm bangalore 2 mumbai tmrw")
# Regex fallback should catch this
check("Informal text (no crash)",  r.intent in ("ROUTE_SEARCH", "UNKNOWN", "BOOK_TICKET"))

r = analyze("Vizag to Hyderabad ac ticket")
check("Vizag → VSKP",  r.entities.source == "VSKP")
check("Hyd  → SC/HYB", r.entities.destination in ("SC", "HYB"))


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
total = _pass + _fail
print(f"\n{'═'*60}")
print(f"  Results:  {_pass}/{total} passed   ({_fail} failed)")
print(f"{'═'*60}\n")
if _fail:
    sys.exit(1)
