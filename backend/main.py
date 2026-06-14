"""GushQuote FastAPI app.

Wires together extraction -> RAG -> quote computation behind two endpoints:
  POST /chat           — conversational lead intake
  POST /voice-webhook  — simulated voicemail transcript, same pipeline

Run:  uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import session_manager
from extraction_agent import _MODEL, _PROVIDER, _USE_LLM, extract
from models import (
    ChatRequest,
    ChatResponse,
    QuoteResult,
    VoiceWebhookRequest,
)
from quote_computer import compute_quote
from rag_pipeline import build_index, find_best_match

app = FastAPI(title="GushQuote API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.on_event("startup")
def _startup() -> None:
    # Ensure the vector index exists before serving traffic.
    try:
        from rag_pipeline import _collection

        if _collection().count() == 0:
            build_index()
    except Exception as exc:  # pragma: no cover
        print(f"[startup] index build deferred: {exc}")
    mode = f"LLM ({_PROVIDER}: {_MODEL})" if _USE_LLM else "deterministic fallback (no API key)"
    print(f"[GushQuote] extraction backend: {mode}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_enabled": _USE_LLM, "provider": _PROVIDER, "model": _MODEL}


def _var_hash(v) -> str:
    """Stable fingerprint of the quote variables so we can skip re-quoting."""
    return "|".join(str(getattr(v, f, "")) for f in
                    ["equipment_type", "size_class", "quantity", "duration_months", "zip_code"])


_CHIT_CHAT = {
    "hi", "hello", "hey", "yo", "sup", "good morning", "good afternoon",
    "whats your name", "what is your name", "who are you", "how are you",
    "im good", "i'm good", "na im good", "ok", "okay", "thanks", "thank you",
    "cool", "nice", "great", "bye", "see ya", "no", "nope", "not really",
    "never mind", "nevermind", "nothing", "just looking", "just browsing",
}


def _is_chit_chat(message: str) -> bool:
    t = message.lower().strip().rstrip(".!?")
    if t in _CHIT_CHAT:
        return True
    # Broad match: any message starting with a casual opener or containing a question
    casual_starts = ("hi ", "hey ", "hello", "yo ", "sup", "ok ", "okay", "thanks", "thank",
                     "bye", "cool", "nice", "great", "no ", "nope", "not really", "never",
                     "im ", "i'm ", "good morning", "good afternoon", "na ")
    if t.startswith(casual_starts):
        return True
    # Contains a question word — likely asking about the company, not requesting equipment
    question_words = ("what ", "who ", "where ", "when ", "why ", "how ", "which ",
                      "tell me", "explain", "describe", "can you", "do you", "is there",
                      "are you", "could you", "would you", "about this", "about the")
    if any(q in t for q in question_words):
        return True
    return False


def _run_pipeline(session_id: str, message: str) -> ChatResponse:
    session = session_manager.get_or_create(session_id)
    variables = session["variables"]
    history = session["history"]

    result = extract(variables, history, message)
    session["variables"] = result.variables

    history.append({"role": "user", "content": message})

    quote_card: QuoteResult | None = None

    # If all variables are present but this is clearly NOT a quote request
    # (greeting, chit-chat, "what's your name") AND the variables haven't
    # changed — just reply conversationally. Don't re-fire the quote engine.
    already_quoted = session.get("last_quote_hash") == _var_hash(result.variables)
    is_casual = _is_chit_chat(message)

    if result.is_complete and not is_casual:
        pricing_row = find_best_match(
            result.variables.equipment_type,
            result.variables.size_class,
            result.variables.additional_requirements,
        )
        if pricing_row is None:
            agent_reply = (
                "I couldn't match that to anything in our fleet. We carry excavators, "
                "bulldozers, skid steers, boom lifts and generators — which of those "
                "fits your job?"
            )
            result.variables.equipment_type = None
            session["variables"] = result.variables
        elif not already_quoted:
            quote_card = compute_quote(result.variables, pricing_row)
            agent_reply = (
                f"Here's your estimate for {quote_card.line_items[0].description.split(' — ')[0]} "
                f"delivered to {result.variables.zip_code}. "
                f"Total comes to ${quote_card.total:,.2f}. "
                "You can download the PDF below, and I'm here if you have other questions."
            )
            session["quote_sent"] = True
            session["last_quote_hash"] = _var_hash(result.variables)
        else:
            # Variables unchanged — let the LLM/fallback handle conversation naturally.
            preamble = result.reply_preamble.strip()
            if preamble:
                agent_reply = preamble
            else:
                agent_reply = (
                    "That estimate is still right above. Need a different machine, "
                    "quantity or ZIP? Or ask me anything about our fleet, delivery, "
                    "pricing or service area."
                )
    elif result.is_complete and is_casual:
        # Casual message after a complete quote — just be friendly.
        name = "GushQuote" if not hasattr(result.variables, "_name") else ""
        agent_reply = result.reply_preamble.strip() or (
            "I'm GushQuote, Midwest Power Rentals' quoting assistant. "
            "Need a different piece of equipment, or have a question about your estimate?"
        )
    else:
        preamble = result.reply_preamble.strip()
        question = result.follow_up_question.strip()
        agent_reply = f"{preamble} {question}".strip() if preamble else question

    history.append({"role": "assistant", "content": agent_reply})

    return ChatResponse(
        agent_reply=agent_reply,
        quote_card=quote_card,
        session_id=session_id,
        extracted=result.variables,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    try:
        return _run_pipeline(session_id, req.message)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        # Include the error in the reply so we can debug on Render
        return ChatResponse(
            agent_reply=f"Sorry, hit an error: {type(exc).__name__}: {exc}",
            session_id=session_id,
        )


@app.post("/voice-webhook", response_model=ChatResponse)
def voice_webhook(req: VoiceWebhookRequest) -> ChatResponse:
    """Simulated voicemail transcript -> same quoting pipeline.

    A fresh session is created per call so a voicemail is processed as a single
    self-contained intake (Twilio would send the full transcribed message at once).
    """
    session_id = f"voice-{uuid.uuid4()}"
    return _run_pipeline(session_id, req.transcript)


@app.post("/reset")
def reset(req: ChatRequest) -> dict:
    session_manager.reset(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


# --- Serve the frontend (so the whole demo runs from one origin) -----------
if FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            str(FRONTEND_DIR / "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
