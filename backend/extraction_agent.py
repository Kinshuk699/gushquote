"""Variable extraction from unstructured lead messages.

Two backends:
  1. LLM (Instructor + OpenAI) — used when OPENAI_API_KEY is set. Best quality,
     handles messy natural language and writes friendly conversational replies.
  2. Deterministic fallback — pure regex/keyword extraction. No API key needed,
     so the demo always runs. Slightly more rigid phrasing.

The public entry point `extract` accepts the running conversation state plus the
latest user message and returns an updated ExtractionResult.
"""
from __future__ import annotations

import os
import re
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from company_kb import COMPANY_FACTS, looks_like_question, match_faq
from models import ExtractionResult, QuoteVariables

# --- Optional LLM backend ---------------------------------------------------
# Supports either DeepSeek (OpenAI-compatible) or OpenAI. DeepSeek is preferred
# when its key is present: cheaper, text-only, and a drop-in via base_url.
_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if _DEEPSEEK_KEY:
    _PROVIDER = "deepseek"
    _API_KEY = _DEEPSEEK_KEY
    _BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    _MODEL = os.getenv("GUSHQUOTE_MODEL", "deepseek-chat")
elif _OPENAI_KEY:
    _PROVIDER = "openai"
    _API_KEY = _OPENAI_KEY
    _BASE_URL = None
    _MODEL = os.getenv("GUSHQUOTE_MODEL", "gpt-4o-mini")
else:
    _PROVIDER = None
    _API_KEY = None
    _BASE_URL = None
    _MODEL = None

_USE_LLM = bool(_API_KEY)

_client = None
if _USE_LLM:
    try:
        import instructor
        from openai import OpenAI

        _oai = OpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _BASE_URL else OpenAI(api_key=_API_KEY)
        # DeepSeek needs JSON mode rather than tool-calling for structured output.
        if _PROVIDER == "deepseek":
            _client = instructor.from_openai(_oai, mode=instructor.Mode.JSON)
        else:
            _client = instructor.from_openai(_oai)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[extraction_agent] LLM backend unavailable ({exc}); using fallback.")
        _USE_LLM = False


SYSTEM_PROMPT = """You are GushQuote, a friendly, sharp sales assistant for Midwest \
Power Rentals — a heavy equipment rental company.

You do TWO things:
  A) Answer general questions and handle casual conversation naturally.
  B) Build rental quotes by collecting FIVE variables from the customer.

COMPANY FACTS (use these to answer questions accurately; never invent details):
{facts}

The five quote variables:
  1. equipment_type (excavator, bulldozer, skid steer, boom lift, generator)
  2. size_class (mini, mid, or large)
  3. quantity (how many units)
  4. duration_months (how long, in months; convert weeks/days)
  5. zip_code (5-digit job-site ZIP)

Rules:
- If the user is just chatting (greeting, "what's your name", "how are you", \
"thanks", "ok", "no", "I'm good", etc.), respond naturally in reply_preamble. \
Set is_complete to the CURRENT state of known variables — do NOT fabricate new \
values. If all five variables happen to already be filled from a prior turn, \
set is_complete True but make reply_preamble your friendly chat response. \
Leave follow_up_question empty.
- If the user asks a general question (hours, delivery, deposit, payment, fleet, \
what you can do, etc.), ANSWER IT helpfully in reply_preamble using the facts, \
then gently nudge toward a quote if appropriate. Leave follow_up_question empty \
unless you still need a quote variable.
- When the user clearly wants a NEW or DIFFERENT quote ("actually make that a \
large dozer", "change to 4 units", "what about generators"), update only the \
fields that changed, fill fresh gaps, and set is_complete accordingly.
- When the user is requesting equipment, MERGE new info with what's already \
known. Never discard a value you already have.
- Ask for only ONE missing quote item at a time, leading with the most important \
gap (equipment, then size, quantity, duration, ZIP).
- Normalise size words ("mid-size", "compact") to mini/mid/large.
- Convert durations: "a couple weeks" -> 0.5, "3 months" -> 3, "90 days" -> 3.
- reply_preamble: your conversational reply (answer + acknowledgement).
- Keep it human. No corporate filler, no emojis.""".format(facts=COMPANY_FACTS)


def extract(
    current: QuoteVariables,
    history: list[dict],
    user_message: str,
) -> ExtractionResult:
    if _USE_LLM and _client is not None:
        try:
            return _extract_llm(current, history, user_message)
        except Exception as exc:  # pragma: no cover
            print(f"[extraction_agent] LLM extraction failed ({exc}); using fallback.")
    return _extract_fallback(current, user_message)


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
def _extract_llm(
    current: QuoteVariables, history: list[dict], user_message: str
) -> ExtractionResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append(
        {
            "role": "system",
            "content": f"Known so far (merge, do not discard): {current.model_dump_json()}",
        }
    )
    for turn in history[-8:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    result = _client.chat.completions.create(
        model=_MODEL,
        response_model=ExtractionResult,
        messages=messages,
        max_retries=2,
        temperature=0.3,
    )
    result.variables = _merge(current, result.variables)
    result.is_complete = _is_complete(result.variables)
    return result


# ---------------------------------------------------------------------------
# Deterministic fallback backend
# ---------------------------------------------------------------------------
_EQUIPMENT_KEYWORDS = {
    "excavator": "excavator",
    "digger": "excavator",
    "bulldozer": "bulldozer",
    "dozer": "bulldozer",
    "skid steer": "skid steer",
    "skid-steer": "skid steer",
    "bobcat": "skid steer",
    "boom lift": "boom lift",
    "boom": "boom lift",
    "man lift": "boom lift",
    "generator": "generator",
    "genset": "generator",
}

_SIZE_KEYWORDS = {
    "mini": "mini",
    "small": "mini",
    "compact": "mini",
    "mid": "mid",
    "medium": "mid",
    "mid-size": "mid",
    "midsize": "mid",
    "standard": "mid",
    "large": "large",
    "big": "large",
    "heavy": "large",
}

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2,
    "pair": 2, "few": 3, "dozen": 12,
}


def _extract_fallback(current: QuoteVariables, msg: str) -> ExtractionResult:
    v = current.model_copy(deep=True)
    before = v.model_dump()
    text = msg.lower()

    # Equipment type
    for kw, val in _EQUIPMENT_KEYWORDS.items():
        if kw in text:
            v.equipment_type = val
            break

    # Size class
    for kw, val in _SIZE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}\b", text):
            v.size_class = val
            break

    # Quantity — digits near a unit word, or number words
    qty = _parse_quantity(text, v.equipment_type)
    if qty is not None:
        v.quantity = qty

    # Duration in months
    dur = _parse_duration(text)
    if dur is not None:
        v.duration_months = dur

    # ZIP code
    zip_match = re.search(r"\b(\d{5})\b", msg)
    if zip_match:
        v.zip_code = zip_match.group(1)

    extracted_something = v.model_dump() != before
    complete = _is_complete(v)

    # If the user is clearly just chatting (greeting, "what's your name", etc.)
    # and the variables are already complete from a prior turn, don't push for
    # another quote — just be friendly.
    chit = msg.lower().strip().rstrip(".!?")
    _CHIT = {"hi","hello","hey","whats your name","what is your name","who are you",
             "how are you","im good","i'm good","na im good","ok","okay","thanks",
             "thank you","cool","nice","great","bye","no","nope","not really",
             "never mind","nevermind","nothing","just browsing","just looking",
             "good morning","good afternoon","sup","yo"}
    if complete and not extracted_something and chit in _CHIT:
        preamble = _chit_chat_reply(msg)
        return ExtractionResult(
            variables=v, is_complete=True, follow_up_question="",
            reply_preamble=preamble,
        )

    # If the user asked a general question and we didn't pull any new quote data,
    # answer the question from the knowledge base instead of pushing for a variable.
    if not complete and not extracted_something:
        faq = match_faq(msg)
        if faq or looks_like_question(msg):
            answer = faq or (
                "Good question — I can help with quotes, our fleet, pricing, delivery, "
                "deposits, hours and service area. Could you say a bit more, or tell me "
                "what equipment you need?"
            )
            return ExtractionResult(
                variables=v,
                is_complete=False,
                follow_up_question="",
                reply_preamble=answer,
            )

    question = "" if complete else _next_question(v)
    preamble = _fallback_preamble(v, complete)

    return ExtractionResult(
        variables=v,
        is_complete=complete,
        follow_up_question=question,
        reply_preamble=preamble,
    )


def _parse_quantity(text: str, equipment_type: Optional[str]) -> Optional[int]:
    # Allow up to a few adjective words between the number and the unit noun,
    # e.g. "two mid-size bulldozers", "3 large electric boom lifts".
    units = (r"(?:units?|of them|excavators?|diggers?|bulldozers?|dozers?|"
             r"skid[ -]?steers?|bobcats?|boom lifts?|lifts?|man lifts?|"
             r"generators?|gensets?|machines?)")
    gap = r"(?:[ -]+\w+){0,3}[ -]+"  # 0-3 filler words then a separator

    m = re.search(rf"\b(\d{{1,3}})\b{gap}{units}", text)
    if m:
        return int(m.group(1))
    for word, num in _NUM_WORDS.items():
        if re.search(rf"\b{word}\b{gap}{units}", text):
            return num
    # bare "I need 2" only if an equipment type is already established this turn
    m2 = re.search(r"\bneed\s+(\d{1,3})\b", text)
    if m2:
        return int(m2.group(1))
    return None


def _parse_duration(text: str) -> Optional[float]:
    # months
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?)\b", text)
    if m:
        return float(m.group(1))
    for word, num in _NUM_WORDS.items():
        if re.search(rf"\b{word}\b\s+months?\b", text):
            return float(num)
    # weeks -> months
    w = re.search(r"(\d+(?:\.\d+)?)\s*weeks?\b", text)
    if w:
        return round(float(w.group(1)) / 4.0, 2)
    for word, num in _NUM_WORDS.items():
        if re.search(rf"\b{word}\b\s+weeks?\b", text):
            return round(num / 4.0, 2)
    # days -> months
    d = re.search(r"(\d+)\s*days?\b", text)
    if d:
        return round(int(d.group(1)) / 30.0, 2)
    return None


def _is_complete(v: QuoteVariables) -> bool:
    return all(
        [
            v.equipment_type,
            v.size_class,
            v.quantity,
            v.duration_months,
            v.zip_code,
        ]
    )


def _merge(old: QuoteVariables, new: QuoteVariables) -> QuoteVariables:
    merged = old.model_copy(deep=True)
    for field in ["equipment_type", "size_class", "quantity", "duration_months", "zip_code"]:
        val = getattr(new, field)
        if val not in (None, "", 0):
            setattr(merged, field, val)
    if new.additional_requirements:
        merged.additional_requirements = new.additional_requirements
    return merged


def _next_question(v: QuoteVariables) -> str:
    if not v.equipment_type:
        return "What kind of equipment are you after — excavator, bulldozer, skid steer, boom lift, or generator?"
    if not v.size_class:
        return f"What size {v.equipment_type} do you need — mini, mid, or large?"
    if not v.quantity:
        return f"How many {v.equipment_type}s do you need?"
    if not v.duration_months:
        return "How long will you need them? You can give me weeks or months."
    if not v.zip_code:
        return "Last thing — what's the job-site ZIP code so I can price delivery?"
    return ""


def _fallback_preamble(v: QuoteVariables, complete: bool) -> str:
    if complete:
        return "Perfect, I've got everything I need."
    known = []
    if v.quantity and v.equipment_type:
        size = f"{v.size_class} " if v.size_class else ""
        known.append(f"{v.quantity} {size}{v.equipment_type}{'s' if v.quantity != 1 else ''}")
    elif v.equipment_type:
        known.append(f"a {v.equipment_type}")
    if known:
        return f"Got it — {known[0]}."
    return "Happy to help with that."


def _chit_chat_reply(msg: str) -> str:
    t = msg.lower().strip().rstrip(".!?")
    if t in ("hi", "hello", "hey", "sup", "yo", "good morning", "good afternoon"):
        return "Hey! Need a different machine or have a question about your estimate?"
    if "name" in t or "who are you" in t:
        return "I'm GushQuote — the quoting assistant for Midwest Power Rentals. I build instant equipment estimates. Got questions, or want a different quote?"
    if "how are you" in t:
        return "Doing great — ready to build or adjust any rental quote you need. What can I do for you?"
    if t in ("im good", "i'm good", "na im good", "ok", "okay", "cool", "nice", "great"):
        return "Awesome. Need a different machine, or anything else I can help with?"
    if t in ("thanks", "thank you"):
        return "Anytime. Want me to price something else, or answer a question?"
    if t in ("bye", "no", "nope", "not really", "never mind", "nevermind", "nothing"):
        return "No problem — I'm here when you need a quote. Just pick a machine or tell me the job."
    if t in ("just looking", "just browsing"):
        return "Take your time. When you see a machine you like, tell me the ZIP, how many and how long, and I'll price it."
    return "Need a different machine, or can I help with a question about pricing, delivery or our fleet?"
