"""Pure-Python pricing engine.

The LLM is never trusted to do arithmetic. It extracts variables and the RAG
layer finds the relevant pricing row; everything below is deterministic Python.

Pricing logic:
  base       = monthly_rate * quantity * duration_months
  delivery   = base_delivery_fee(depot) + per_mile * estimated_miles
  surcharge  = flat environmental + fuel surcharge
  discount   = long-term discount tiers on the base
  tax        = (base - discount + delivery + surcharge) * TAX_RATE
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Optional

from models import LineItem, QuoteResult, QuoteVariables
from rag_pipeline import load_delivery_rows

TAX_RATE = 0.085
ENVIRONMENTAL_SURCHARGE = 150.0
QUOTE_VALID_DAYS = 7

# Long-term discount tiers applied to the equipment base subtotal.
DISCOUNT_TIERS = [
    (12, 0.15),  # 12+ months -> 15% off
    (6, 0.10),   # 6-11 months -> 10% off
    (3, 0.05),   # 3-5 months  -> 5% off
]


def _money(x: float) -> float:
    return round(x + 1e-9, 2)


def _depot_for_zip(zip_code: Optional[str]) -> Optional[dict]:
    if not zip_code or len(zip_code) < 3:
        return None
    prefix = zip_code[:3]
    for row in load_delivery_rows():
        served = row["zip_prefixes_served"].split("|")
        if prefix in served:
            return row
    return None


def _estimate_miles(zip_code: str, depot: dict) -> int:
    """Deterministic pseudo-distance from the depot, based on the ZIP.

    Real systems geocode; for a demo we derive a stable, plausible mileage from
    the digits of the ZIP so the same ZIP always yields the same number.
    """
    seed = int(hashlib.sha256(zip_code.encode()).hexdigest(), 16)
    max_radius = int(depot["max_delivery_radius_miles"])
    return 12 + (seed % max(1, max_radius - 12))


def compute_quote(variables: QuoteVariables, pricing_row: dict) -> QuoteResult:
    """Compute a full itemised quote from extracted variables + a pricing row."""
    qty = int(variables.quantity or 1)
    duration = float(variables.duration_months or 1)
    monthly = float(pricing_row["monthly_rate"])
    equipment_label = pricing_row["equipment_type"]
    size_class = pricing_row["size_class"]

    line_items: list[LineItem] = []
    notes: list[str] = []

    # --- Base equipment cost ------------------------------------------------
    base = monthly * qty * duration
    dur_label = _duration_label(duration)
    line_items.append(
        LineItem(
            description=(
                f"{equipment_label} ({size_class}) — {qty} unit"
                f"{'s' if qty != 1 else ''} x {dur_label} @ ${monthly:,.0f}/mo"
            ),
            amount=_money(base),
        )
    )

    # --- Long-term discount -------------------------------------------------
    discount = 0.0
    for months, rate in DISCOUNT_TIERS:
        if duration >= months:
            discount = base * rate
            line_items.append(
                LineItem(
                    description=f"Long-term discount ({int(rate * 100)}% for {months}+ month rental)",
                    amount=-_money(discount),
                )
            )
            break

    # --- Delivery -----------------------------------------------------------
    depot = _depot_for_zip(variables.zip_code)
    delivery = 0.0
    depot_name = "TBD"
    if depot:
        depot_name = depot["depot_city"]
        miles = _estimate_miles(variables.zip_code, depot)
        delivery = float(depot["base_delivery_fee"]) + float(depot["per_mile_rate"]) * miles
        # Round-trip delivery + pickup is one combined fee for the demo.
        line_items.append(
            LineItem(
                description=(
                    f"Delivery & pickup — {variables.zip_code} "
                    f"(~{miles} mi from {depot_name} depot)"
                ),
                amount=_money(delivery),
            )
        )
    else:
        notes.append(
            f"ZIP {variables.zip_code or '(none)'} is outside our standard service "
            "area — a sales rep will confirm delivery feasibility and cost."
        )

    # --- Surcharge ----------------------------------------------------------
    line_items.append(
        LineItem(description="Environmental & fuel surcharge", amount=ENVIRONMENTAL_SURCHARGE)
    )

    # --- Totals -------------------------------------------------------------
    subtotal = base - discount + delivery + ENVIRONMENTAL_SURCHARGE
    tax = subtotal * TAX_RATE
    total = subtotal + tax
    deposit_total = float(pricing_row["deposit_per_unit"]) * qty

    # --- Minimum rental warning --------------------------------------------
    min_days = int(pricing_row["min_rental_days"])
    if duration * 30 < min_days:
        notes.append(
            f"Note: this unit has a {min_days}-day minimum rental; pricing reflects the minimum term."
        )

    valid_until = (_dt.date.today() + _dt.timedelta(days=QUOTE_VALID_DAYS)).isoformat()
    quote_id = _quote_id(variables, equipment_label)

    return QuoteResult(
        equipment_label=f"{equipment_label} ({size_class})",
        line_items=line_items,
        subtotal=_money(subtotal),
        discount=_money(discount),
        tax=_money(tax),
        total=_money(total),
        deposit_total=_money(deposit_total),
        valid_until=valid_until,
        quote_id=quote_id,
        delivery_depot=depot_name,
        notes=notes,
    )


def _duration_label(duration: float) -> str:
    if duration < 1:
        weeks = round(duration * 4)
        return f"{weeks} week{'s' if weeks != 1 else ''}"
    if duration == int(duration):
        d = int(duration)
        return f"{d} month{'s' if d != 1 else ''}"
    return f"{duration:g} months"


def _quote_id(variables: QuoteVariables, label: str) -> str:
    raw = f"{label}-{variables.zip_code}-{variables.quantity}-{_dt.datetime.now().isoformat()}"
    digest = int(hashlib.sha256(raw.encode()).hexdigest(), 16) % 10000
    return f"GQ-{_dt.date.today().year}-{digest:04d}"
