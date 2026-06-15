from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.models import schemas
from app.services.chat_nlp_service import ChatNLPService, ChatAnalysisRequest
from app.services.nlp_service import NLPService
from app.services.booking_service import BookingService
from app.api.v1.dependencies import get_db
from app.services.response_generator import ResponseGenerator

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

    generator = ResponseGenerator()

    # ── SMALL TALK ────────────────────────────────────────────────────────
    if analysis.next_action == "SMALL_TALK":
        reply = generator.small_talk_response(analysis.intent)
        messages.append(schemas.ChatMessage(role="assistant", content=reply))
        return messages

    # ── ASK CLARIFICATION ─────────────────────────────────────────────────
    if analysis.next_action == "ASK_CLARIFICATION":
        missing = analysis.missing_required_slots[0] if analysis.missing_required_slots else None
        reply = generator.ask_clarification(missing, analysis.clarification_question)
        messages.append(schemas.ChatMessage(role="assistant", content=reply))
        return messages

    # ── SEARCH ROUTE ───────────────────────────────────────────────────────
    if analysis.next_action in ["SEARCH_ROUTE", "ROUTE_ANALYSIS"]:
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = generator.ask_clarification("source" if not src else "destination")
        else:
            search_entities = {
                "source_station": src,
                "destination_station": dst,
                "date": analysis.entities.date,
                "travel_class": analysis.entities.travel_class,
                "passengers": analysis.entities.passengers,
            }
            results = booking_svc.search_trains(search_entities, db)
            reply = generator.format_search_results(src, dst, results, analysis.entities.preference)

    # ── BOOK TICKET ─────────────────────────────────────────────────────
    elif analysis.next_action == "BOOK":
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = generator.ask_clarification("source" if not src else "destination")
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
                reply = generator.format_booking_confirmation(booking)
            except Exception as e:
                reply = generator.format_booking_failure(str(e))

    # ── ESTIMATE FARE ──────────────────────────────────────────────────
    elif analysis.next_action == "ESTIMATE_FARE":
        src = analysis.entities.source
        dst = analysis.entities.destination

        if not src or not dst:
            reply = generator.ask_clarification("source" if not src else "destination")
        else:
            search_entities = {
                "source_station": src,
                "destination_station": dst,
            }
            results = booking_svc.search_trains(search_entities, db)

            if results:
                from app.repository.fare_repo import FareRepository
                fares = FareRepository().get_by_train(results[0].train_number, db)
                reply = generator.format_fares(src, dst, results[0].train_number, fares)
            else:
                reply = generator.format_search_results(src, dst, [])

    # ── COMPARE ROUTES ────────────────────────────────────────────────
    elif analysis.next_action == "COMPARE_ROUTES":
        reply = "📊 Route comparison feature coming soon! For now, try searching individual routes."

    # ── CANCEL BOOKING ──────────────────────────────────────────────────────
    elif analysis.next_action == "CANCEL_BOOKING":
        bid = analysis.entities.booking_id
        if not bid:
            reply = generator.ask_clarification("booking_id")
        else:
            success = booking_svc.cancel_booking(int(bid), db)
            reply = generator.format_cancellation(str(bid), success)

    # ── BOOKING HISTORY ─────────────────────────────────────────────────────
    elif analysis.next_action == "BOOKING_HISTORY":
        history = booking_svc.list_user_bookings(1, db)
        reply = generator.format_history(history)

    # ── CHECK ROUTE ─────────────────────────────────────────────────────────
    elif analysis.next_action == "CHECK_ROUTE":
        tn = analysis.entities.train_number
        if not tn:
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
            reply = generator.format_route(tn, stops)
        else:
            reply = generator.ask_clarification("train_number")

    # ── UNKNOWN ────────────────────────────────────────────────────────
    else:
        reply = generator.get_unknown()

    messages.append(schemas.ChatMessage(role="assistant", content=reply))
    return messages
