from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.models import schemas
from app.services.chat_nlp_service import ChatNLPService, ChatAnalysisRequest
from app.services.nlp_service import NLPService
from app.services.booking_service import BookingService
from app.api.v1.dependencies import get_db

router = APIRouter()
nlp_service = NLPService()
chat_nlp = ChatNLPService()
booking_svc = BookingService()


@router.post("/chat/analyze")
def analyze_chat(request: ChatAnalysisRequest):
    """
    Analyze user message and return structured intent, entities, and next action.
    Used for smart chat understanding with NLP.
    """
    return chat_nlp.analyze(request)


@router.post("/chat", response_model=List[schemas.ChatMessage])
def chat_endpoint(messages: List[schemas.ChatMessage], db: Session = Depends(get_db)):
    if not messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty")

    # Find last user message
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build request for the new NLP service
    analysis_request = ChatAnalysisRequest(
        user_message=last_user.content,
        conversation_history=[
            {
                "role": m.role,
                "content": m.content,
            }
            for m in messages[:-1]  # Exclude current message
        ],
    )

    analysis = chat_nlp.analyze(analysis_request)

    # Fallback to old NLP (kept for logging / backward compat)
    intent, entities = nlp_service.predict(last_user.content)
    print(f"[chat] intent={analysis.intent!r}  entities={analysis.entities}")

    # ── ASK CLARIFICATION ─────────────────────────────────────────────────
    if analysis.next_action == "ASK_CLARIFICATION":
        reply = f"❓ {analysis.clarification_question or 'Please provide more details.'}"
        messages.append(schemas.ChatMessage(role="assistant", content=reply))
        return messages

    # ── SEARCH ROUTE ───────────────────────────────────────────────────────
    if analysis.next_action in ["SEARCH_ROUTE", "ROUTE_ANALYSIS"]:
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = (
                "🔍 Please tell me the source and destination.\n"
                'Example: *"Find trains from Bangalore to Mumbai"*'
            )
        else:
            search_entities = {
                "source_station": src,
                "destination_station": dst,
                "date": analysis.entities.date,
                "travel_class": analysis.entities.travel_class,
                "passengers": analysis.entities.passengers,
            }
            results = booking_svc.search_trains(search_entities, db)

            if not results:
                reply = (
                    f"❌ No trains found from **{src}** → **{dst}**.\n"
                    "Try using city names like Bangalore, Mumbai, Delhi, Chennai, etc."
                )
            else:
                lines = [f"🚂 Found **{len(results)}** train(s) from {src} → {dst}:\n"]
                for r in results[:10]:
                    lines.append(
                        f"  • **{r.train_number}** — {r.train_name}\n"
                        f"    ({r.source_station_code} → {r.destination_station_code})"
                    )
                if len(results) > 10:
                    lines.append(f"\n  _...and {len(results) - 10} more_")

                if analysis.entities.preference:
                    lines.append(f"\n💡 Sorted by: **{analysis.entities.preference}**")

                lines.append('\n💡 Say *"Book 2 sleeper tickets"* to book one!')
                reply = "\n".join(lines)

    # ── BOOK TICKET ─────────────────────────────────────────────────────
    elif analysis.next_action == "BOOK":
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = (
                "🎫 To book a ticket, please mention source and destination.\n"
                'Example: *"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow"*'
            )
        else:
            try:
                booking_entities = {
                    "source_station": src,
                    "destination_station": dst,
                    "date": analysis.entities.date,
                    "travel_class": analysis.entities.travel_class or "SL",
                    "passengers": analysis.entities.passengers or 1,
                }
                booking = booking_svc.create_mock_booking(booking_entities, db)
                date_str = (
                    booking.travel_date.strftime("%d %b %Y")
                    if hasattr(booking.travel_date, "strftime")
                    else str(booking.travel_date)
                )
                reply = (
                    f"✅ **Booking Confirmed!**\n\n"
                    f"  🆔 Booking ID  : **{booking.id}**\n"
                    f"  🚆 Train       : **{booking.train_number}**\n"
                    f"  🎫 Class       : **{booking.travel_class}**\n"
                    f"  👥 Passengers  : **{booking.passenger_count}**\n"
                    f"  📅 Travel Date : **{date_str}**\n"
                    f"  📌 Status      : **{booking.status}**\n\n"
                    'Say *"Show my bookings"* to see all your bookings.'
                )
            except Exception as e:
                reply = (
                    f"❌ Booking failed: {str(e)}\n"
                    'Please try: *"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow"*'
                )

    # ── ESTIMATE FARE ──────────────────────────────────────────────────
    elif analysis.next_action == "ESTIMATE_FARE":
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = (
                "💰 To check fares, please mention the route.\n"
                'Example: *"Fare from Bangalore to Mumbai"*'
            )
        else:
            search_entities = {
                "source_station": src,
                "destination_station": dst,
            }
            results = booking_svc.search_trains(search_entities, db)

            if results:
                from app.repository.fare_repo import FareRepository

                fares = FareRepository().get_by_train(results[0].train_number, db)
                if fares:
                    lines = [f"💰 **Fares for {src} → {dst}** ({results[0].train_number}):\n"]
                    for f in fares:
                        lines.append(f"  • **{f.class_type}**: ₹{f.amount:.0f}")
                    reply = "\n".join(lines)
                else:
                    reply = "💰 No fare data available for this route."
            else:
                reply = (
                    f"💰 No trains found for **{src}** → **{dst}**.\n"
                    "Try different city names."
                )

    # ── COMPARE ROUTES ────────────────────────────────────────────────
    elif analysis.next_action == "COMPARE_ROUTES":
        reply = "📊 Route comparison feature coming soon! For now, try searching individual routes."

    # ── CANCEL BOOKING ───────────────────────────────────────────────────────
    # Fix #6: use analysis.intent instead of old intent variable
    elif analysis.intent == "CANCEL_BOOKING":
        bid = analysis.entities.booking_id
        if not bid:
            reply = '❌ Please provide a Booking ID.\nExample: *"Cancel booking 5"*'
        else:
            success = booking_svc.cancel_booking(int(bid), db)
            if success:
                reply = f"✅ Booking **#{bid}** has been cancelled successfully."
            else:
                reply = f'❌ Booking **#{bid}** not found. Say *"Show my bookings"* to see valid IDs.'

    # ── BOOKING HISTORY ────────────────────────────────────────────────
    elif analysis.intent == "BOOKING_HISTORY":
        history = booking_svc.list_user_bookings(1, db)
        if not history:
            reply = '📭 You have no bookings yet.\nTry: *"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow"*'
        else:
            lines = [f"📋 **Your Bookings** ({len(history)} total):\n"]
            for b in history:
                icon = "✅" if b.status == "CONFIRMED" else "❌"
                date_s = (
                    b.travel_date.strftime("%d %b %Y")
                    if hasattr(b.travel_date, "strftime")
                    else str(b.travel_date)
                )
                lines.append(
                    f"  {icon} **#{b.id}** — Train {b.train_number}  |  "
                    f"{b.travel_class} × {b.passenger_count} pax  |  "
                    f"{date_s}  |  {b.status}"
                )
            reply = "\n".join(lines)

    # ── CHECK ROUTE ────────────────────────────────────────────────────────
    elif analysis.intent == "CHECK_ROUTE":
        tn = analysis.entities.train_number
        if not tn:
            # try to get a train number from search if source/destination are known
            search_entities = {
                "source_station": analysis.entities.source,
                "destination_station": analysis.entities.destination,
            }
            results = booking_svc.search_trains(search_entities, db)
            if results:
                tn = results[0].train_number

        if tn:
            from app.repository.route_repo import RouteRepository

            stops = RouteRepository().get_by_train(tn, db)
            if stops:
                lines = [f"🗺️ **Route for Train {tn}** ({len(stops)} stops):\n"]
                for s in stops[:20]:
                    arr = s.arrival_time or "--:--"
                    dep = s.departure_time or "--:--"
                    lines.append(f"  {s.sequence:>2}. {s.station_code:<8}  arr:{arr}  dep:{dep}")
                if len(stops) > 20:
                    lines.append(f"  ... and {len(stops) - 20} more stops")
                reply = "\n".join(lines)
            else:
                reply = f"🗺️ No route data found for train {tn}."
        else:
            reply = (
                "🗺️ Please provide a train number.\n"
                'Example: *"Show route for 12657"*'
            )

    # ── UNKNOWN ────────────────────────────────────────────────────────
    else:
        reply = (
            "👋 Hi! I'm your **AI Train Ticket Assistant** for Indian Railways.\n\n"
            "Here's what I can do:\n"
            '  🔍 *"Find trains from Bangalore to Mumbai"*\n'
            '  🎫 *"Book 2 sleeper tickets from Delhi to Chennai tomorrow"*\n'
            '  📋 *"Show my bookings"*\n'
            '  ❌ *"Cancel booking 5"*\n'
            '  💰 *"Fare from Pune to Hyderabad"*\n'
            '  🗺️ *"Route for train 12657"*\n'
            '  🚂 *"Cheapest route from Bangalore to Mumbai"*\n'
            '  ⚡ *"Fastest trains to Chennai"*'
        )

    messages.append(schemas.ChatMessage(role="assistant", content=reply))
    return messages
