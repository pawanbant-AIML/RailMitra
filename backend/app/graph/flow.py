"""
graph/flow.py – Agent graph entry point.

The RailMitra agent runs as a simple tool-calling loop (see agent/agent_service.py).
This module is retained as a thin wrapper for future LangGraph migration.
"""

from app.agent.agent_service import AgentService

_agent = None


def get_agent() -> AgentService:
    """Return a singleton AgentService instance."""
    global _agent
    if _agent is None:
        _agent = AgentService()
    return _agent