"""
api/v1/endpoints/chat.py

Chat endpoint for RailAI.

This version accepts the new frontend request shape:

{
  "message": "Which is cheapest?",
  "session_id": "abc123",
  "history": [...]
}

It converts the provided history into the structure expected by the agent,
passes the current message + history + session_id to AgentService.run(),
and returns the full chat list with the assistant reply appended.

Legacy / debugging NLP route is kept at /chat/analyze.
"""

from __future__ import annotations

from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.agent_service import AgentRunResult, AgentService
from app.api.v1.dependencies import get_db
from app.core.logger import logger
from app.models import schemas
from app.services.chat_nlp_service import ChatAnalysisRequest, ChatNLPService

router = APIRouter()

# Singletons — safe for stateless use
_chat_nlp = ChatNLPService()
_agent_svc = AgentService()


class ChatRequest(BaseModel):
    """
    New request format expected from the frontend.
    """
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None)
    history: List[schemas.ChatMessage] = Field(default_factory=list)


def _prepare_chat_payload(
    request: Union[ChatRequest, List[schemas.ChatMessage]],
) -> tuple[str, str, List[schemas.ChatMessage], List[dict]]:
    if isinstance(request, list):
        raw_history = request
        if not raw_history:
            raise HTTPException(status_code=400, detail="History cannot be empty.")
        last_item = raw_history[-1]
        if (last_item.role or "").strip() != "user" or not (last_item.content or "").strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Legacy chat body must end with the latest user message. "
                    "Send the new request shape {message, session_id, history} instead."
                ),
            )
        message = last_item.content.strip()
        session_id = "default"
        history_items = raw_history[:-1]
        response_messages: List[schemas.ChatMessage] = list(raw_history)
    else:
        message = (request.message or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        session_id = (request.session_id or "default").strip() or "default"
        history_items = request.history or []
        response_messages = list(history_items)
        response_messages.append(schemas.ChatMessage(role="user", content=message))

    conversation_history = []
    for item in history_items:
        if not item:
            continue
        role = (item.role or "").strip()
        content = (item.content or "").strip()
        if role in {"user", "assistant"} and content:
            conversation_history.append({"role": role, "content": content})

    return message, session_id, response_messages, conversation_history


@router.post("/chat/analyze")
def analyze_chat(request: ChatAnalysisRequest):
    """
    Legacy rule-based analyser, kept for debugging/backward compatibility.
    """
    return _chat_nlp.analyze(request)


@router.post("/chat", response_model=List[schemas.ChatMessage])
def chat_endpoint(
    request: Union[ChatRequest, List[schemas.ChatMessage]],
    db: Session = Depends(get_db),
):
    """
    Main chat endpoint.

    Accepts either the new frontend payload:
      {message, session_id, history}
    or the legacy raw history array for backward compatibility.

    Returns:
      - full message list (previous history + current user message + assistant reply)
    """
    message, session_id, response_messages, conversation_history = _prepare_chat_payload(request)

    logger.info(
        "[chat] session=%s user_msg=%r history_turns=%d",
        session_id,
        message[:120],
        len(conversation_history),
    )

    try:
        reply = _agent_svc.run(
            user_message=message,
            conversation_history=conversation_history,
            db=db,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("[chat] Agent error: %s", exc, exc_info=True)
        reply = (
            "⚠️ Sorry, I ran into an unexpected error. "
            "Please try again or rephrase your question."
        )

    response_messages.append(schemas.ChatMessage(role="assistant", content=reply))
    return response_messages


@router.post("/chat/structured", response_model=schemas.StructuredChatResponse)
def structured_chat_endpoint(
    request: Union[ChatRequest, List[schemas.ChatMessage]],
    db: Session = Depends(get_db),
):
    """
    Structured chat endpoint for UI actions.

    Keeps /chat backward compatible while giving the frontend enough metadata
    to open a booking form instead of parsing assistant text.
    """
    message, session_id, response_messages, conversation_history = _prepare_chat_payload(request)

    logger.info(
        "[chat:structured] session=%s user_msg=%r history_turns=%d",
        session_id,
        message[:120],
        len(conversation_history),
    )

    try:
        result = _agent_svc.run_structured(
            user_message=message,
            conversation_history=conversation_history,
            db=db,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("[chat:structured] Agent error: %s", exc, exc_info=True)
        result = AgentRunResult(
            answer=(
                "Sorry, I ran into an unexpected error. "
                "Please try again or rephrase your question."
            ),
            diagnostics={
                "route": "endpoint_exception",
                "fallback_used": True,
                "local_error": f"{type(exc).__name__}: {exc}",
            },
        )

    response_messages.append(schemas.ChatMessage(role="assistant", content=result.answer))
    return schemas.StructuredChatResponse(
        messages=response_messages,
        action=result.action,
        booking_draft=result.booking_draft,
        missing_required_fields=result.missing_required_fields,
        diagnostics=result.diagnostics,
    )
