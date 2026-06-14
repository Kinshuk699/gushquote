# GushQuote

An AI quoting agent that turns unstructured B2B rental inquiries into itemized quotes in under a minute. Lives as a chat widget on a fake heavy-equipment rental site, **Midwest Power Rentals**.

Lead types *"I need 2 mid excavators for 3 months, ZIP 60007"* → the agent extracts the variables, runs RAG over the pricing data, computes a deterministic line-item quote, and renders a quote card in chat.

Built as a demo for a Founder's Office (New Products) application at [Gushwork](https://gushwork.ai).

---

## What's inside

```
Gush/
├── backend/
│   ├── main.py              FastAPI app — /chat, /voice-webhook, serves frontend
│   ├── extraction_agent.py  Instructor + GPT-4o extraction, with a no-API-key fallback
│   ├── rag_pipeline.py      ChromaDB + sentence-transformers over the pricing rows
│   ├── quote_computer.py    Pure-Python pricing math (the LLM never does arithmetic)
│   ├── session_manager.py   In-memory multi-turn session state
│   ├── models.py            Pydantic models / API contract
│   ├── data/                pricing.csv + delivery_fees.csv
│   └── requirements.txt
├── frontend/
│   └── index.html           Landing page + chat widget + voice modal (no build step)
├── scripts/
│   └── seed_data.py         Embed the CSV rows into ChromaDB
└── README.md
```

---

## APIs / keys you need

GushQuote is designed to **run with zero API keys** so the demo never breaks. There is exactly one optional key, and **DeepSeek is the recommended choice** (cheaper, text-only).

| Service | Required? | What it's for | Cost |
|---|---|---|---|
| **DeepSeek API key** | **Optional (recommended)** | Natural conversational extraction via Instructor + `deepseek-chat`. Set `DEEPSEEK_API_KEY` and it's used automatically. Text-only, no vision needed. | ~$0.0001–0.001 per conversation |
| OpenAI API key | Optional alternative | Same role, used only if `DEEPSEEK_API_KEY` is empty. `gpt-4o-mini`. | ~$0.01–0.03 per conversation |
| ChromaDB built-in embedder (ONNX MiniLM-L6-v2) | No key | Local embeddings for RAG. Downloads ~80 MB once. **No PyTorch.** | Free / local |
| ChromaDB | No key | Local vector store. | Free / local |

**Disk footprint:** the whole virtualenv is ~420 MB (we deliberately avoid PyTorch by using ChromaDB's bundled ONNX embedder instead of `sentence-transformers`).

**Without any key:** the agent falls back to a deterministic regex/keyword extractor. It still asks follow-ups, fills all five variables, and produces identical quotes — just with slightly more rigid phrasing. Great for an offline-safe demo.

**With DeepSeek (or OpenAI):** the conversation feels noticeably more natural and handles messy phrasing better. This is the version to demo live.

> Bottom line for you: grab **one DeepSeek API key** (platform.deepseek.com) for the polished demo. Everything else is local and free.

---

## Setup & run

```bash
cd /Users/Kinshuk/Developer/Gush

# 1. Install backend deps (a venv is already created at backend/.venv)
./backend/.venv/bin/python -m pip install -r backend/requirements.txt

# 2. (Optional) add your OpenAI key
cp backend/.env.example backend/.env
#   then edit backend/.env and paste OPENAI_API_KEY=sk-...

# 3. Seed the vector index (also self-heals on first request if skipped)
./backend/.venv/bin/python scripts/seed_data.py

# 4. Run the server (serves both API and the landing page)
cd backend && ../backend/.venv/bin/python -m uvicorn main:app --reload
```

Then open **http://localhost:8000** — the landing page, chat widget and voice modal are all served from there.

API docs: **http://localhost:8000/docs**

---

## Demo script

1. Open http://localhost:8000 and click **Get a quote** (bottom-right).
2. Type: `I need 2 mid excavators for 3 months, ZIP 60007` → watch the quote card render.
3. Try an incomplete one: `I need a bulldozer` → the agent asks for size, quantity, duration, ZIP one at a time.
4. Click **📞 Call for a quote** in the hero → a sample voicemail transcribes, then runs through the *same* pipeline via `/voice-webhook`.

---

## How it works

```
user message
   │
   ▼
extraction_agent.extract()   ── merges with prior turns, fills 5 variables
   │  (Instructor+GPT-4o  OR  regex fallback)
   ▼
is_complete? ── no ──▶ ask one follow-up question
   │ yes
   ▼
rag_pipeline.find_best_match()  ── semantic search over pricing rows (ChromaDB)
   │
   ▼
quote_computer.compute_quote()  ── deterministic line items, delivery, discount, tax
   │
   ▼
quote_card JSON ──▶ rendered in the chat widget
```

**Key decision:** the LLM only extracts variables and the RAG layer only retrieves rows. All money math is plain Python, so quotes are always correct.

---

## API contract

### `POST /chat`
```json
{ "session_id": "sess-abc", "message": "I need 3 excavators for 2 months near 60007" }
```
Returns `{ agent_reply, quote_card | null, session_id, extracted }`.

### `POST /voice-webhook`
```json
{ "transcript": "Hi, I'm calling about renting two bulldozers...", "caller_phone": "+12605551234" }
```
Same response shape — feeds the identical pipeline. This is the seam where a real Twilio + Whisper integration would plug in.
