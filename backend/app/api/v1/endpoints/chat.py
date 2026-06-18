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

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.agent_service import AgentService
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
    history: Optional[List[schemas.ChatMessage]] = Field(default_factory=list)


@router.post("/chat/analyze")
def analyze_chat(request: ChatAnalysisRequest):
    """
    Legacy rule-based analyser, kept for debugging/backward compatibility.
    """
    return _chat_nlp.analyze(request)


@router.post("/chat", response_model=List[schemas.ChatMessage])
def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Main chat endpoint.

    Accepts:
      - message: current user message
      - session_id: conversation/session identifier
      - history: previous messages from the frontend

    Returns:
      - full message list (previous history + current user message + assistant reply)
    """
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = (request.session_id or "default").strip() or "default"

    # Build conversation history for the agent
    conversation_history = []
    for item in request.history or []:
        if not item:
            continue
        role = (item.role or "").strip()
        content = (item.content or "").strip()
        if role in {"user", "assistant"} and content:
            conversation_history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

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

    # Return the updated chat list
    response_messages: List[schemas.ChatMessage] = list(request.history or [])
    response_messages.append(schemas.ChatMessage(role="user", content=message))
    response_messages.append(schemas.ChatMessage(role="assistant", content=reply))

    return response_messages
