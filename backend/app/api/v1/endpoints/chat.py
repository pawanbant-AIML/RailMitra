"""
api/v1/endpoints/chat.py

Refactored chat endpoint that routes every user message through the
AI Agent (agent_service.AgentService).  The old rule-based ChatNLPService
is removed from the hot path; it is still available at /chat/analyze for
debugging / backward compatibility.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

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
    messages: List[schemas.ChatMessage],
    db: Session = Depends(get_db),
):
    """
    Main conversational endpoint.

    Accepts the full conversation history (all past messages + the current
    user message as the last item).  Routes the user's message through the
    LLM agent, which may call multiple tools before producing a final reply.

    Returns the full messages list with the assistant reply appended.
    """
    if not messages:
        raise HTTPException(status_code=400, detail="Message list cannot be empty.")

    # The last user message drives this turn
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(status_code=400, detail="No user message found in the list.")

    # Build conversation history (everything before the current message)
    history = [
        {"role": m.role, "content": m.content}
        for m in messages[:-1]
        if m.role in ("user", "assistant") and m.content
    ]

    logger.info(f"[chat] user_msg={last_user.content[:100]!r}  history_turns={len(history)}")

    try:
        reply = _agent_svc.run(
            user_message=last_user.content,
            conversation_history=history,
            db=db,
        )
    except Exception as exc:
        logger.error(f"[chat] Agent error: {exc}", exc_info=True)
        reply = (
            "⚠️ Sorry, I ran into an unexpected error. "
            "Please try again or rephrase your question."
        )

    messages.append(schemas.ChatMessage(role="assistant", content=reply))
    return messages
