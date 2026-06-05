from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.services.nlp_service import NLPService
from app.services.booking_service import BookingService
from app.api.v1.dependencies import get_db

router = APIRouter()
nlp         = NLPService()
booking_svc = BookingService()


@router.post("/chat", response_model=List[schemas.ChatMessage])
def chat_endpoint(messages: List[schemas.ChatMessage], db: Session = Depends(get_db)):
    if not messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty")

    # Find last user message
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(status_code=400, detail="No user message found")

    intent, entities = nlp.predict(last_user.content)
    print(f"[chat] intent={intent!r}  entities={entities}")

    # ── SEARCH TRAIN ───────────────────────────────────────────────────
    if intent == "search_train":
        src = entities.get("source_station", "")
        dst = entities.get("destination_station", "")
        if not src or not dst:
            reply = (
                "🔍 Please tell me the source and destination.\n"
                "Example: *\"Find trains from Bangalore to Mumbai\"*"
            )
        else:
            results = booking_svc.search_trains(entities, db)
            if not results:
                reply = (
                    f"❌ No trains found from **{src}** → **{dst}**.\n"
                    "Try using city names like Bangalore, Mumbai, Delhi, Chennai, etc."
                )
            else:
                lines = [f"🚂 Found **{len(results)}** train(s) from {src} → {dst}:\n"]
                for r in results[:10]:
                    lines.append(
                        f"  • **{r.train_number}** — {r.train_name}"
                        f"  ({r.source_station_code} → {r.destination_station_code})"
                    )
                if len(results) > 10:
                    lines.append(f"\n  _...and {len(results)-10} more_")
                lines.append("\n💡 Say *\"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow\"* to book one!")
                reply = "\n".join(lines)

    # ── BOOK TICKET ────────────────────────────────────────────────────
    elif intent == "book_ticket":
        src = entities.get("source_station", "")
        dst = entities.get("destination_station", "")
        if not src or not dst:
            reply = (
                "🎫 To book a ticket, please mention source and destination.\n"
                "Example: *\"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow\"*"
            )
        else:
            try:
                booking = booking_svc.create_mock_booking(entities, db)
                date_str = booking.travel_date.strftime("%d %b %Y") if hasattr(booking.travel_date, "strftime") else str(booking.travel_date)
                reply = (
                    f"✅ **Booking Confirmed!**\n\n"
                    f"  🆔 Booking ID  : **{booking.id}**\n"
                    f"  🚆 Train       : **{booking.train_number}**\n"
                    f"  🎫 Class       : **{booking.travel_class}**\n"
                    f"  👥 Passengers  : **{booking.passenger_count}**\n"
                    f"  📅 Travel Date : **{date_str}**\n"
                    f"  📌 Status      : **{booking.status}**\n\n"
                    f"Say *\"Show my bookings\"* to see all your bookings."
                )
            except Exception as e:
                reply = (
                    f"❌ Booking failed: {str(e)}\n"
                    "Please try: *\"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow\"*"
                )

    # ── CANCEL TICKET ──────────────────────────────────────────────────
    elif intent == "cancel_ticket":
        bid = entities.get("booking_id")
        if not bid:
            reply = "❌ Please provide a Booking ID.\nExample: *\"Cancel booking 5\"*"
        else:
            success = booking_svc.cancel_booking(int(bid), db)
            if success:
                reply = f"✅ Booking **#{bid}** has been cancelled successfully."
            else:
                reply = f"❌ Booking **#{bid}** not found. Say *\"Show my bookings\"* to see valid IDs."

    # ── BOOKING HISTORY ────────────────────────────────────────────────
    elif intent == "booking_history":
        history = booking_svc.list_user_bookings(1, db)
        if not history:
            reply = "📭 You have no bookings yet.\nTry: *\"Book 2 sleeper tickets from Bangalore to Mumbai tomorrow\"*"
        else:
            lines = [f"📋 **Your Bookings** ({len(history)} total):\n"]
            for b in history:
                icon    = "✅" if b.status == "CONFIRMED" else "❌"
                date_s  = b.travel_date.strftime("%d %b %Y") if hasattr(b.travel_date, "strftime") else str(b.travel_date)
                lines.append(
                    f"  {icon} **#{b.id}** — Train {b.train_number}  |  "
                    f"{b.travel_class} × {b.passenger_count} pax  |  "
                    f"{date_s}  |  {b.status}"
                )
            reply = "\n".join(lines)

    # ── CHECK FARE ─────────────────────────────────────────────────────
    elif intent == "check_fare":
        tn = entities.get("train_number")
        if not tn:
            # Try searching for trains and show fare for first result
            results = booking_svc.search_trains(entities, db)
            if results:
                tn = results[0].train_number
        if tn:
            from app.repository.fare_repo import FareRepository
            fares = FareRepository().get_by_train(tn, db)
            if fares:
                lines = [f"💰 **Fares for Train {tn}:**\n"]
                for f in fares:
                    lines.append(f"  • **{f.class_type}**: ₹{f.amount:.0f}")
                reply = "\n".join(lines)
            else:
                reply = f"💰 No fare data available for train {tn}."
        else:
            reply = (
                "💰 To check fares, please mention the train number or route.\n"
                "Example: *\"What is the fare for 12657\"* or "
                "*\"Fare from Bangalore to Mumbai\"*"
            )

    # ── CHECK ROUTE ────────────────────────────────────────────────────
    elif intent == "check_route":
        tn = entities.get("train_number")
        if not tn:
            results = booking_svc.search_trains(entities, db)
            if results:
                tn = results[0].train_number
        if tn:
            from app.repository.route_repo import RouteRepository
            stops = RouteRepository().get_by_train(tn, db)
            if stops:
                lines = [f"🗺️ **Route for Train {tn}** ({len(stops)} stops):\n"]
                for s in stops[:20]:
                    arr  = s.arrival_time   or "--:--"
                    dep  = s.departure_time or "--:--"
                    lines.append(f"  {s.sequence:>2}. {s.station_code:<8}  arr:{arr}  dep:{dep}")
                if len(stops) > 20:
                    lines.append(f"  ... and {len(stops)-20} more stops")
                reply = "\n".join(lines)
            else:
                reply = f"🗺️ No route data found for train {tn}."
        else:
            reply = (
                "🗺️ Please provide a train number.\n"
                "Example: *\"Show route for 12657\"*"
            )

    # ── UNKNOWN ────────────────────────────────────────────────────────
    else:
        reply = (
            "👋 Hi! I'm your **AI Train Ticket Assistant** for Indian Railways.\n\n"
            "Here's what I can do:\n"
            "  🔍 *\"Find trains from Bangalore to Mumbai\"*\n"
            "  🎫 *\"Book 2 sleeper tickets from Delhi to Chennai tomorrow\"*\n"
            "  📋 *\"Show my bookings\"*\n"
            "  ❌ *\"Cancel booking 5\"*\n"
            "  💰 *\"Fare from Pune to Hyderabad\"*\n"
            "  🗺️ *\"Route for train 12657\"*\n"
            "  🚂 *\"Is there a train between Kolkata and Varanasi?\"*"
        )

    messages.append(schemas.ChatMessage(role="assistant", content=reply))
    return messages