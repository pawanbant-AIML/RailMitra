"""
api/v1/endpoints/chat.py

Refactored chat endpoint that routes every user message through the
AI Agent (agent_service.AgentService).  The old rule-based ChatNLPService
is removed from the hot path; it is still available at /chat/analyze for
debugging / backward compatibility.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from pydantic import BaseModel
from app.models import schemas
from app.services.chat_nlp_service import ChatNLPService, ChatAnalysisRequest
from app.api.v1.dependencies import get_db
from app.agent.agent_service import AgentService
from app.core.logger import logger

router = APIRouter()

# Singletons – constructed once at startup, thread-safe for read-only state
_chat_nlp = ChatNLPService()
_agent_svc = AgentService()


# ---------------------------------------------------------------------------
# New request model matching frontend
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[schemas.ChatMessage]] = None


# ---------------------------------------------------------------------------
# Debug endpoint: run the legacy NLP analyser (kept for dev / testing)
# ---------------------------------------------------------------------------
@router.post("/chat/analyze")
def analyze_chat(request: ChatAnalysisRequest):
    """
    Legacy rule-based intent analyser.
    Returns structured intent, entities, and next_action.
    Useful for debugging what the old NLP layer detected.
    """
    return _chat_nlp.analyze(request)


# ---------------------------------------------------------------------------
# Primary chat endpoint
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=List[schemas.ChatMessage])
def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Main conversational endpoint.

    Accepts a ChatRequest with `message`, `session_id`, and optional
    `history` (list of previous messages).  Routes the user's message through
    the LLM agent, which may call multiple tools before producing a final reply.

    Returns the full messages list with the assistant reply appended.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Build conversation history from the frontend-provided list (if any)
    history = [
        {"role": m.role, "content": m.content}
        for m in (request.history or [])
        if m.role in ("user", "assistant") and m.content
    ]

    logger.info(
        f"[chat] session={request.session_id} "
        f"user_msg={request.message[:100]!r}  "
        f"history_turns={len(history)}"
    )

    try:
        reply = _agent_svc.run(
            user_message=request.message,
            conversation_history=history,
            db=db,
            session_id=request.session_id or "default",
        )
    except Exception as exc:
        logger.error(f"[chat] Agent error: {exc}", exc_info=True)
        reply = (
            "⚠️ Sorry, I ran into an unexpected error. "
            "Please try again or rephrase your question."
        )

    # Build response: all previous messages (if provided) + the new reply
    response_messages = list(request.history) if request.history else []
    response_messages.append(schemas.ChatMessage(role="user", content=request.message))
    response_messages.append(schemas.ChatMessage(role="assistant", content=reply))

    return response_messages
