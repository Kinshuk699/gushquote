"""In-memory session state.

A plain dict keyed by session UUID is plenty for a single-tenant demo. Each
session tracks the running conversation history and the variables extracted so
far. Restarting the server clears all sessions, which is fine for a demo.
"""
from __future__ import annotations

from threading import Lock

from models import QuoteVariables

_sessions: dict[str, dict] = {}
_lock = Lock()


def get_or_create(session_id: str) -> dict:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "history": [],          # list[{"role", "content"}]
                "variables": QuoteVariables(),
                "quote_sent": False,
            }
        return _sessions[session_id]


def reset(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def all_sessions() -> dict[str, dict]:
    return _sessions
