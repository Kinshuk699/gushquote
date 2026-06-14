# GushQuote — Weekend Build Brief

## TL;DR

Build **GushQuote**: an AI agent that lives as a chat widget on a B2B company's website, takes unstructured lead inquiries, extracts variables via LLM, runs RAG over pricing data, computes a line-item quote, and responds conversationally — all under 60 seconds.

This is a demo product for a **Founder's Office (New Products)** role application at **Gushwork** (gushwork.ai — AI marketing employees for traditional industries, $11M raised from Lightspeed, B Capital, etc.).

---

## What We're Building

### Demo Flow
1. A fake heavy equipment rental company landing page (e.g., "Midwest Power Rentals")
2. A chat bubble widget in the bottom-right corner
3. Lead types: *"Hey, I need some excavators for a project"*
4. GushQuote agent asks follow-ups: *"How many? What size? What zip code? How long?"*
5. Once all variables are extracted, the agent:
   - Queries a RAG pipeline over the company's pricing CSV
   - Computes line items (base rate × quantity × duration + delivery fees by zip + surcharges)
   - Renders a formatted quote card in-chat
   - Offers to email/SMS the quote
6. Quote is "sent" (logged/displayed for demo)

### What We're NOT Building (v2)
- Real Twilio SMS/voice integration (just a simulated transcript endpoint)
- Multi-tenant (one fake company only)
- Auth, payments, production deployment
- Streaming SSE responses (simple request-response is enough)
- Actual email delivery

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Fake Company Website (static HTML/CSS/JS)               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Chat Widget (vanilla JS)                        │    │
│  │  - Sends messages to POST /chat                  │    │
│  │  - Renders quote cards                           │    │
│  │  - Session ID in localStorage                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend (Python 3.11+)                         │
│                                                         │
│  POST /chat                                             │
│    ├── SessionManager (in-memory dict, UUID keys)       │
│    ├── ExtractionAgent (Instructor + GPT-4o)            │
│    │     └── Extracts: equipment_type, quantity,        │
│    │         duration, zip_code, additional_requirements │
│    ├── RAG Pipeline (ChromaDB + sentence-transformers)  │
│    │     └── Indexes pricing CSV rows                   │
│    │     └── Retrieves relevant pricing tiers           │
│    ├── QuoteComputer (pure Python)                      │
│    │     └── base_rate × qty × duration                 │
│    │     └── + delivery_fee(zip_code)                   │
│    │     └── + surcharges                               │
│    └── ResponseFormatter                                │
│          └── Returns: agent_reply + quote_card JSON     │
│                                                         │
│  POST /voice-webhook (stretch goal)                     │
│    └── Simulated transcript → same pipeline             │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack & Recommended Libraries

### 1. Backend Framework
| Library | Why |
|---------|-----|
| **FastAPI** | Async Python, auto OpenAPI docs, Pydantic-native, trivial to set up |
| **Uvicorn** | ASGI server, comes with FastAPI |

```bash
pip install "fastapi[standard]" uvicorn
```

### 2. LLM & Structured Extraction
| Library | Why |
|---------|-----|
| **Instructor** (v1.15+) | Best library for structured LLM output. 13.2k stars. Pydantic models → validated extraction. Supports OpenAI, Anthropic, Google, 15+ providers. Auto-retries on validation failure. |
| **OpenAI GPT-4o** | Best balance of speed, structured output reliability, and cost for a demo. Use `gpt-4o-mini` during dev to save costs. |

```bash
pip install instructor openai
```

**Pattern**: Define a Pydantic model for the extracted variables:
```python
from pydantic import BaseModel, Field

class QuoteVariables(BaseModel):
    equipment_type: str = Field(description="Type of equipment needed")
    quantity: int = Field(description="Number of units")
    duration_months: int = Field(description="Rental duration in months")
    zip_code: str = Field(description="Job site zip code")
    additional_requirements: str = Field(default="", description="Any special needs")
    is_complete: bool = Field(description="Whether all required variables are collected")
    follow_up_question: str = Field(default="", description="Question to ask if incomplete")
```

Use Instructor's `create_with_completion` to get both the structured output and the raw completion for the conversational reply.

### 3. RAG Pipeline — Document Parsing
| Library | Why |
|---------|-----|
| **Docling** (IBM) | 61.5k stars, MIT license. Best-in-class PDF table extraction. Handles merged cells, complex layouts. Pure Python. Also parses XLSX, CSV, HTML. Integrates with LangChain/LlamaIndex. |
| **pandas** | For CSV reading and manipulation |

```bash
pip install docling pandas
```

**For the weekend**: Start with a handcrafted CSV (not PDF). CSV is trivial to parse and index. Add Docling PDF parsing as a live demo moment — drag in a messy PDF and watch it work.

### 4. RAG Pipeline — Vector Store & Embeddings
| Library | Why |
|---------|-----|
| **ChromaDB** | 28.4k stars, Apache 2.0. `pip install chromadb`. In-memory mode for prototyping. 4-function API (create_collection, add, query, get). Zero infrastructure. |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Free, local, 384-dim embeddings. No API keys needed. Good enough for pricing table retrieval. 80MB download. |

```bash
pip install chromadb sentence-transformers
```

**Alternative**: Use OpenAI `text-embedding-3-small` if you already have an API key and want to skip local model download. Slightly better quality, costs ~$0.02 per 1M tokens.

### 5. Session Management
**Simple Python dict** — keyed by UUID session IDs. No Redis, no SQLite needed for a demo with <5 concurrent sessions.

```python
sessions: dict[str, dict] = {}
# Each session stores: conversation_history, extracted_vars, quote_state
```

### 6. Pricing Computation
**Pure Python functions** — do NOT use the LLM for arithmetic. The LLM extracts variables and finds relevant pricing rows; Python computes the actual quote. This guarantees accuracy.

```python
def compute_quote(variables: QuoteVariables, pricing_rows: list[dict]) -> QuoteResult:
    base = pricing_rows["monthly_rate"] * variables.quantity * variables.duration_months
    delivery = get_delivery_fee(variables.zip_code)
    subtotal = base + delivery
    tax = subtotal * 0.08  # or zip-based tax rate
    total = subtotal + tax
    return QuoteResult(line_items=[...], total=total)
```

### 7. Chat Widget & Landing Page (Frontend)

**IMPORTANT: Use the `design-taste-frontend` agent skill when building the frontend.** This skill is already installed at `~/.agents/skills/design-taste-frontend/SKILL.md`. It enforces anti-slop design rules — no AI-purple gradients, no Inter font defaults, no three-equal-cards layouts, no em-dashes, proper hero discipline, WCAG contrast checks, and a strict pre-flight checklist. The agent should declare a "Design Read" before writing any HTML/CSS. For Midwest Power Rentals (industrial B2B equipment rental), the skill will infer: serious B2B, trust-first, high-contrast, industrial sans-serif typography, restrained motion.

**Tech approach: Custom vanilla HTML/CSS/JS** — a single `index.html` with embedded `<style>` and `<script>`. No frameworks, no build step. 

The page has two parts:
1. **The company landing page** — hero, services grid, equipment catalog, about section, contact footer. Built with the taste-skill design rules.
2. **The chat widget** — chat bubble toggle (bottom-right), message list with typing indicator, quote card rendering (styled div with line items table), session ID in `localStorage`, fetches to `POST /chat` with `{session_id, message}`.

### 8. Voice Entry Point (Concrete Plan)

**The idea: "Call for a Quote" phone number on the landing page.** The fake company site has a prominent phone number in the header and a "Call for a Quote" section. When clicked (demo mode), it triggers a modal showing a simulated voicemail transcript being processed in real-time through the same GushQuote pipeline. This proves voice → quote without requiring actual Twilio setup.

**Implementation:**
- Add a `POST /voice-webhook` endpoint to FastAPI
- Accepts `{transcript: str, caller_phone: str}` — exactly what Twilio would send after transcribing a voicemail
- Runs the SAME extraction → RAG → compute pipeline as the chat widget
- Returns the same `{agent_reply, quote_card}` response
- On the frontend: a "📞 Call for a Quote" button opens a modal showing a hardcoded sample transcript flowing in, then the quote appearing. Or, a simple `<textarea>` where the user can paste a transcript and hit "Process"

**Sample demo transcript** (pre-loaded for the demo):
> "Hi yeah this is Mike from Tri-State Construction. I'm looking at renting two mid-size bulldozers for a land clearing job we've got coming up. Probably need them for 3 months, maybe 4. We're out in zip 46818, Fort Wayne area. Can you give me a number?"

**Production path (for the slide deck, not built this weekend):** Twilio phone number → `<Record>` verb → recording URL → OpenAI Whisper transcription → same `/voice-webhook` pipeline. ngrok for local dev tunnel.

### 9. Running Locally & Demo
- **ngrok** — expose local FastAPI to the internet for demo (free tier works)
- Or just run everything on `localhost:8000` and open the HTML file in a browser

```bash
# Terminal 1
uvicorn main:app --reload

# Terminal 2 (if needed for ngrok)
ngrok http 8000
```

---

## Project Structure

```
/Users/Kinshuk/Developer/Gush/
├── backend/
│   ├── main.py              # FastAPI app, /chat endpoint
│   ├── session_manager.py   # In-memory session state
│   ├── extraction_agent.py  # Instructor + GPT-4o variable extraction
│   ├── rag_pipeline.py      # ChromaDB indexing + query
│   ├── quote_computer.py    # Pure Python pricing math
│   ├── models.py            # Pydantic models (QuoteVariables, QuoteResult, etc.)
│   ├── data/
│   │   ├── pricing.csv        # Equipment pricing matrix (15-20 rows)
│   │   └── delivery_fees.csv  # Zip-based delivery fees
│   └── requirements.txt
├── frontend/
│   └── index.html           # Fake company landing page + chat widget
├── scripts/
│   └── seed_data.py         # Generate pricing CSV + index into ChromaDB
└── README.md
```

---

## Fake Company Profile: "Midwest Power Rentals"

- **Industry**: Heavy equipment rental & leasing
- **Tagline**: "Move Earth. Rent Smart."
- **Location**: Serving IL, IN, OH, MI, WI. HQ in Chicago, depots in Indianapolis, Detroit, Columbus, Milwaukee.
- **Equipment**: Excavators (mini, mid, large), bulldozers, skid steers, boom lifts, generators
- **Pricing structure**: Daily/Weekly/Monthly rates, delivery fees by distance from nearest depot, weekend surcharge, long-term discounts

---

## Dummy Data Generation for RAG (Concrete Approach)

### Step 1: Hand-craft `backend/data/pricing.csv`
Create 15-20 rows of realistic equipment. Here's the exact schema and sample rows:

```csv
equipment_type,size_class,weight_class,daily_rate,weekly_rate,monthly_rate,min_rental_days,deposit_per_unit,description
Mini Excavator,1-3 ton,Compact,350,1050,2800,2,1500,Ideal for tight-access digging. Residential trenching. backyard pools. utility work. Zero tail-swing for confined jobsites.
Mid Excavator,5-10 ton,Standard,650,1950,5200,3,3500,Foundation digging. medium demolition. trenching. grading. Fits on a standard lowboy trailer.
Large Excavator,15-30 ton,Heavy,1200,3600,9500,5,6000,Mass excavation. deep foundations. quarry work. rock breaking with hydraulic hammer attachment.
Mini Skid Steer,<1750 lbs,Compact,280,840,2200,1,1200,Landscaping. interior demo. tight spaces. Available with auger/trencher/bucket attachments.
Mid Skid Steer,1750-2200 lbs,Standard,400,1200,3200,2,2000,General construction. grading. material handling. Compatible with 50+ attachments.
Large Skid Steer,>2200 lbs,Heavy,550,1650,4400,3,3000,Heavy-duty grading. asphalt milling. forestry mulching. High-flow hydraulics.
Mini Bulldozer,<100 HP,Compact,500,1500,4000,3,2500,Finish grading. light clearing. residential site prep. 6-way blade.
Mid Bulldozer,100-200 HP,Standard,850,2550,6800,4,4500,Land clearing. road building. medium earthmoving. LGP (low ground pressure) available.
Large Bulldozer,>200 HP,Heavy,1400,4200,11000,7,8000,Heavy earthmoving. mining overburden. large-scale grading and ripping.
Towable Boom Lift,30-50 ft,Compact,320,960,2600,2,1500,Towable. fits through standard gates. Indoor/outdoor rated. 500lb platform capacity.
Electric Boom Lift,40-60 ft,Standard,480,1440,3900,3,2800,Zero-emission indoor work. warehouse maintenance. narrow chassis. Quiet operation.
Diesel Boom Lift,60-80 ft,Heavy,650,1950,5200,4,4000,Rough-terrain capable. outdoor construction. 4WD. Articulating boom for up-and-over reach.
Towable Generator,20kW,Compact,180,540,1400,1,800,Backup power for small sites. single-phase. quiet-run enclosure. 24hr runtime per tank.
Mid Generator,50-75kW,Standard,350,1050,2800,2,1800,Jobsite primary power. three-phase. trailer-mounted. 200A breaker panel included.
Large Generator,>100kW,Heavy,600,1800,4800,3,3500,Full-site power for multiple trades. 480V three-phase. auto-transfer-switch compatible.
```

### Step 2: Hand-craft `backend/data/delivery_fees.csv`
```csv
depot_zip,depot_city,zip_prefixes_served,base_delivery_fee,per_mile_rate,max_delivery_radius_miles
60601,Chicago,600-602,604,606-608,150,4.50,120
60601,Chicago,463-464,466,150,4.50,120
46204,Indianapolis,460-462,468,472-474,125,4.00,100
48226,Detroit,480-485,488,140,4.25,100
43215,Columbus,430-432,437-438,130,4.00,100
53202,Milwaukee,530-532,534-535,140,4.25,110
```

### Step 3: Build `scripts/seed_data.py`
This script does the following (exact flow):

1. **Load CSVs** with pandas — `pricing.csv` and `delivery_fees.csv`
2. **Create text chunks for embedding**: For each pricing row, generate a human-readable description string that combines ALL the information an LLM needs to match a lead inquiry to the right row. Example chunk:
   > "Mid Excavator: 5-10 ton standard excavator from Midwest Power Rentals. Daily rate $650. Weekly rate $1,950. Monthly rate $5,200. Minimum rental 3 days. Deposit $3,500 per unit. Used for foundation digging, medium demolition, trenching, and grading. Fits on a standard lowboy trailer for transport."
   Also create chunks for delivery fee rows:
   > "Delivery from Chicago depot (zip 60601) to zip codes starting with 600-602, 604, 606-608. Base delivery fee $150 plus $4.50 per mile. Maximum delivery radius 120 miles."
3. **Embed with sentence-transformers** (`all-MiniLM-L6-v2`) — each text chunk gets a 384-dim vector
4. **Store in ChromaDB** — create a collection called `"pricing"`, add all chunks with metadata (equipment_type, size_class, monthly_rate, etc.) so the retrieval layer can return actual pricing numbers alongside the chunks
5. **Run once** before starting the FastAPI server

### Step 4: How RAG queries work at runtime
When a lead says "I need 3 excavators for 2 months near Chicago", the system:
1. Embeds the lead message
2. Queries ChromaDB for top-3 similar pricing chunks
3. Returns the matching rows with `monthly_rate`, `delivery_fee`, etc.
4. `quote_computer.py` does the math: `3 × $5,200 × 2 + delivery fee`

---

## Implementation Plan (8-10 hours)

### Phase 1: Foundation (2h)
1. Set up project structure, virtual env, `requirements.txt`
2. Create pricing CSV with 15-20 realistic equipment rows
3. Build `seed_data.py` — load CSV → embed with sentence-transformers → store in ChromaDB
4. Create Pydantic models (`models.py`)

### Phase 2: Core Pipeline (3h)
5. Build `extraction_agent.py` — Instructor + GPT-4o, system prompt for multi-turn variable extraction
6. Build `rag_pipeline.py` — query ChromaDB by equipment type + size, return top-k pricing rows
7. Build `quote_computer.py` — compute line items from extracted vars + retrieved pricing
8. Build `session_manager.py` — create/read/update conversation state

### Phase 3: API (1h)
9. Build `main.py` — `POST /chat` endpoint wiring everything together
10. Test manually with curl / FastAPI docs

### Phase 4: Frontend (2h)
11. Build fake company landing page — hero, services grid, about section, contact. **Use `design-taste-frontend` skill for anti-slop design.**
12. Build chat widget — toggle, message list, input, loading state, quote card rendering
13. Wire widget to `POST /chat`

### Phase 5: Polish & Stretch (1-2h)
14. Add voice webhook endpoint (simulated transcript)
15. Polish quote card CSS
16. Add error handling for missing pricing data
17. Write README with demo instructions

---

## Key Design Decisions

1. **Instructor over LangChain**: LangChain is overkill for a weekend. Instructor does one thing (structured extraction) extremely well with Pydantic. Less code, fewer abstractions.

2. **GPT-4o over local models**: For a demo that needs to impress, GPT-4o's conversational ability and structured output reliability are worth the API cost (~$0.01-0.03 per conversation). Use `gpt-4o-mini` during development.

3. **Python math over LLM math**: The LLM is unreliable for arithmetic. Do all pricing computation in deterministic Python. The LLM only extracts variables and finds relevant pricing rows.

4. **CSV over PDF for pricing data**: Start with CSV (zero parsing friction). Add Docling PDF parsing as a live demo moment — drag in a messy PDF and watch it work.

5. **Single-turn quote rendering over streaming**: Stream the final quote card JSON. The conversational back-and-forth is the "streaming" — each message is a discrete HTTP request. No need for SSE/WebSocket complexity.

6. **In-memory sessions over Redis**: For a demo, a Python dict works perfectly. Restarting the server loses sessions — that's fine for a demo.

7. **`design-taste-frontend` skill for the landing page**: The taste-skill (43k stars, installed at `~/.agents/skills/design-taste-frontend/`) enforces premium, non-generic UI. It reads the brief, infers the right design language, and applies anti-slop rules (no AI-purple, no Inter font, no three-equal-cards). For Midwest Power Rentals, it'll produce an industrial B2B aesthetic instead of generic SaaS-landing-page slop.

---

## What to Tell the Next Chat Session

When starting the next chat, paste this brief and say:

> "I'm building GushQuote — an AI agent that estimates B2B quotes from unstructured leads. Here's the full brief. Let's start with Phase 1: project setup and pricing data. Create the project structure, install dependencies, and build the fake pricing CSV + ChromaDB indexing."
>
> **Important:** When building the landing page and chat widget frontend (Phase 4), use the `design-taste-frontend` skill. It's already installed and will automatically apply anti-slop design rules so the site looks premium, not AI-generated.

---

## Appendix: API Contract

### POST /chat
**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "I need 3 excavators for a 2-month project near Chicago, zip 60007"
}
```

**Response:**
```json
{
  "agent_reply": "Got it! Let me put together a quote for 3 excavators. What size class do you need — mini (1-3 ton), mid (5-10 ton), or large (15-30 ton)?",
  "quote_card": null,
  "session_id": "uuid-string"
}
```

**When quote is ready:**
```json
{
  "agent_reply": "Here's your estimate for 3 Mid-Class Excavators delivered to 60007:",
  "quote_card": {
    "line_items": [
      {"description": "Mid-Class Excavator (5-10 ton) — 3 units × 2 months @ $4,200/mo", "amount": 25200.00},
      {"description": "Delivery to 60007 (45 miles from Chicago depot)", "amount": 350.00},
      {"description": "Environmental surcharge", "amount": 150.00},
      {"description": "Tax (8.5%)", "amount": 2180.25}
    ],
    "subtotal": 25700.00,
    "tax": 2180.25,
    "total": 27880.25,
    "valid_until": "2026-06-21",
    "quote_id": "GQ-2026-0042"
  },
  "session_id": "uuid-string"
}
```

### POST /voice-webhook (stretch)
**Request:**
```json
{
  "transcript": "Hi yeah I'm calling about renting some bulldozers. I need two of them for about 3 months. We're out in Fort Wayne, Indiana area.",
  "caller_phone": "+12605551234"
}
```
**Response:** Same as `/chat` — feeds into the identical pipeline.
