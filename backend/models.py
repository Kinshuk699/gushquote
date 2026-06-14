"""Pydantic models for GushQuote.

These models define the structured contract between the LLM extraction layer,
the pricing engine, and the HTTP API. Keeping them in one place makes the data
flow easy to reason about.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class QuoteVariables(BaseModel):
    """The variables we need to extract from a lead before we can build a quote.

    The extraction agent fills these in across multiple turns of conversation.
    `is_complete` flips to True only when every required field is present.
    """

    equipment_type: Optional[str] = Field(
        default=None,
        description="Type of equipment, e.g. 'excavator', 'bulldozer', 'skid steer', "
        "'boom lift', 'generator'. Leave null if not yet mentioned.",
    )
    size_class: Optional[str] = Field(
        default=None,
        description="Size of the equipment: 'mini', 'mid', or 'large'. Leave null if unknown.",
    )
    quantity: Optional[int] = Field(
        default=None, description="Number of units requested. Leave null if not mentioned."
    )
    duration_months: Optional[float] = Field(
        default=None,
        description="Rental duration in months (fractions allowed, e.g. 0.5 for two weeks). "
        "Leave null if not mentioned.",
    )
    zip_code: Optional[str] = Field(
        default=None,
        description="Five-digit job-site ZIP code. Leave null if not mentioned.",
    )
    additional_requirements: str = Field(
        default="", description="Any special needs, attachments, or notes."
    )


class ExtractionResult(BaseModel):
    """What the extraction agent returns each turn."""

    variables: QuoteVariables
    is_complete: bool = Field(
        description="True only when equipment_type, size_class, quantity, "
        "duration_months and zip_code are ALL known."
    )
    follow_up_question: str = Field(
        default="",
        description="A short, friendly question asking for the single most important "
        "missing piece of information. Empty when is_complete is True.",
    )
    reply_preamble: str = Field(
        default="",
        description="A short conversational acknowledgement of what the lead said.",
    )


# ---------------------------------------------------------------------------
# Quote computation
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    description: str
    amount: float


class QuoteResult(BaseModel):
    equipment_label: str
    line_items: list[LineItem]
    subtotal: float
    discount: float = 0.0
    tax: float
    total: float
    deposit_total: float
    valid_until: str
    quote_id: str
    delivery_depot: str
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    agent_reply: str
    quote_card: Optional[QuoteResult] = None
    session_id: str
    extracted: Optional[QuoteVariables] = None


class VoiceWebhookRequest(BaseModel):
    transcript: str
    caller_phone: str = ""
