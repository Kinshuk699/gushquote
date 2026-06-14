"""Company knowledge base for Midwest Power Rentals.

Powers the chatbot's ability to answer general questions (hours, service area,
delivery, deposits, payment, capabilities) in addition to building quotes.

- The LLM path injects COMPANY_FACTS into its system prompt for natural answers.
- The offline fallback path uses FAQS (keyword-matched) so it still answers
  common questions with no API key.
"""
from __future__ import annotations

import re

COMPANY_FACTS = """
COMPANY: Midwest Power Rentals — heavy equipment rental & leasing. Family-run since 1998.
TAGLINE: Move Earth. Rent Smart.
SERVICE AREA: Illinois, Indiana, Ohio, Michigan and Wisconsin.
DEPOTS (5): Chicago IL (HQ, ZIP 600-608, 120 mi radius), Indianapolis IN (ZIP 460-474, 100 mi),
  Detroit MI (ZIP 480-488, 100 mi), Columbus OH (ZIP 430-438, 100 mi), Milwaukee WI (ZIP 530-535, 110 mi).
HOURS: Monday-Saturday, 6:00am-7:00pm Central. Closed Sundays. 24/7 emergency breakdown line for active rentals.
PHONE: (312) 555-0147.  EMAIL: rent@midwestpower.example.
FLEET: Excavators (mini 1-3t, mid 5-10t, large 15-30t), Bulldozers (mini <100HP, mid 100-200HP, large >200HP),
  Skid steers (mini/mid/large), Boom lifts (towable/electric/diesel, 30-80ft), Generators (20kW-100kW+).
  340+ machines total, all late-model and dealer-serviced.
PRICING: Daily, weekly and monthly rates. Monthly is the best value. Long-term discounts:
  5% off at 3+ months, 10% off at 6+ months, 15% off at 12+ months.
DELIVERY: Priced from the nearest depot — a base fee plus per-mile rate. Same-day or next-day on
  stocked units when you're inside the radius. Delivery AND pickup are included in the one delivery line.
DEPOSIT: A refundable deposit is required per unit (varies by machine, e.g. $1,500 for a mini excavator
  up to $8,000 for a large bulldozer). Returned after the equipment passes return inspection.
SURCHARGE: A flat environmental & fuel surcharge applies. Tax is 8.5%.
MINIMUM RENTAL: Each machine has a minimum rental term (1-7 days depending on size).
PAYMENT: Major credit cards, ACH, and approved net-30 terms for established commercial accounts.
OPERATORS: We rent equipment only (no operators), but we offer free delivery-day walkarounds and
  phone support. Certified operator referrals available on request.
INSURANCE: Renters must carry general liability and equipment coverage; we can recommend a same-day
  short-term policy partner.
ATTACHMENTS: Buckets, augers, breakers, grapples, forks, mulchers and more — ask and we'll add them to a quote.
WHAT THE ASSISTANT CAN DO: build instant line-item rental quotes, answer questions about the fleet,
  pricing, delivery, deposits, hours and service area, take requests by chat or by voice, and email/text quotes.
"""

# (keyword pattern, answer). First match wins. Used by the offline fallback.
FAQS: list[tuple[str, str]] = [
    (r"\b(what can you (do|help)|who are you|what are you|how (can|do) you help|capabilities|help me with)\b",
     "I'm the GushQuote assistant for Midwest Power Rentals. I can build you an instant line-item "
     "rental quote, answer questions about our fleet, pricing, delivery, deposits, hours and service "
     "area, and take your request by chat or voice. Want a quote, or have a question?"),
    (r"\b(hours|open|closed|timing|what time|when are you open)\b",
     "We're open Monday to Saturday, 6am to 7pm Central, closed Sundays. Active rentals also get a "
     "24/7 emergency breakdown line."),
    (r"\b(where|service area|areas|states|region|cover|locations?|depots?|near me|deliver to)\b",
     "We serve Illinois, Indiana, Ohio, Michigan and Wisconsin from 5 depots — Chicago, Indianapolis, "
     "Detroit, Columbus and Milwaukee. Tell me your ZIP and I'll price delivery from the closest one."),
    (r"\b(phone|call|number|contact|reach you|email)\b",
     "You can reach us at (312) 555-0147 or rent@midwestpower.example — or just tell me what you need "
     "right here and I'll quote it."),
    (r"\b(deliver|delivery|drop ?off|pick ?up|transport|how (do|does) (i|delivery) get)\b",
     "We deliver and pick up from your nearest depot — a base fee plus a per-mile rate, billed as one "
     "line on your quote. Stocked units are often same-day or next-day inside our radius. What's your ZIP?"),
    (r"\b(deposit|down ?payment|security)\b",
     "Yes, there's a refundable deposit per unit — from about $1,500 for a mini excavator up to $8,000 "
     "for a large bulldozer. It's returned after the equipment passes return inspection."),
    (r"\b(discount|deal|cheaper|long ?term|best (price|rate|value))\b",
     "Monthly rates are the best value, and long-term rentals get automatic discounts: 5% off at 3+ "
     "months, 10% at 6+, and 15% at 12+. I apply those for you in the quote."),
    (r"\b(pay|payment|invoice|credit card|net ?30|financing|terms)\b",
     "We take major credit cards, ACH, and net-30 terms for established commercial accounts."),
    (r"\b(operator|driver|run it|certified|who operates)\b",
     "We rent equipment only, but you get a free delivery-day walkaround, phone support, and certified "
     "operator referrals on request."),
    (r"\b(insurance|insured|liability|coverage)\b",
     "Renters carry general liability and equipment coverage. We can point you to a same-day short-term "
     "policy partner if you need one."),
    (r"\b(attachment|bucket|auger|breaker|grapple|fork|mulcher|hammer)\b",
     "We stock buckets, augers, breakers, grapples, forks, mulchers and more. Tell me what you're doing "
     "and I'll add the right attachment to your quote."),
    (r"\b(what (do you have|equipment|machines)|fleet|inventory|catalog|types? of|list)\b",
     "Our fleet covers excavators (mini/mid/large), bulldozers, skid steers, boom lifts and generators — "
     "340+ machines, all dealer-serviced. Which type fits your job?"),
    (r"\b(minimum|min rental|shortest|how short|one day)\b",
     "Each machine has a short minimum term, from 1 day on compact units up to 7 days on the largest "
     "iron. I'll factor it into your quote automatically."),
    (r"\b(tax|surcharge|fees?|hidden)\b",
     "Quotes are fully itemized: equipment, any discount, delivery & pickup, a flat environmental & fuel "
     "surcharge, and 8.5% tax. No hidden fees."),
]


def match_faq(message: str) -> str:
    """Return a canned FAQ answer for `message`, or '' if nothing matches."""
    text = message.lower()
    for pattern, answer in FAQS:
        if re.search(pattern, text):
            return answer
    return ""


def looks_like_question(message: str) -> bool:
    text = message.lower().strip()
    if text.endswith("?"):
        return True
    starters = ("what", "where", "when", "how", "do you", "can you", "are you",
                "is there", "who", "why", "which", "tell me", "could you", "would you")
    return text.startswith(starters)
