"""RAG pipeline over the pricing data.

Indexes each pricing row as an embedded text chunk in ChromaDB and exposes a
semantic `query` that returns the matching pricing rows (with their raw numeric
metadata) for a given natural-language lead inquiry.

Embeddings use ChromaDB's built-in default embedder (ONNX MiniLM, ~80 MB, no
PyTorch), so retrieval needs no API key and stays light.

Uses an in-memory ephemeral ChromaDB client — the index is rebuilt from CSV at
startup, so no persistent disk is needed. Perfect for free Render/Railway hosting.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).parent / "data"
PRICING_CSV = DATA_DIR / "pricing.csv"
DELIVERY_CSV = DATA_DIR / "delivery_fees.csv"
COLLECTION_NAME = "pricing"


@lru_cache(maxsize=1)
def _embedding_fn():
    # ONNX MiniLM-L6-v2 bundled with ChromaDB — same model as sentence-transformers
    # but via onnxruntime, so no torch dependency and a much smaller footprint.
    return embedding_functions.ONNXMiniLM_L6_V2()


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    """Ephemeral in-memory client — free on Render/Railway (no disk needed)."""
    return chromadb.Client()


def _row_to_chunk(row: dict) -> str:
    """Build a human-readable chunk that captures everything the retriever needs."""
    return (
        f"{row['equipment_type']}: {row['size_class']} {row['weight_class'].lower()} "
        f"equipment from Midwest Power Rentals. "
        f"Daily rate ${row['daily_rate']}. Weekly rate ${row['weekly_rate']}. "
        f"Monthly rate ${row['monthly_rate']}. Minimum rental {row['min_rental_days']} days. "
        f"Deposit ${row['deposit_per_unit']} per unit. {row['description']}"
    )


def load_pricing_rows() -> list[dict]:
    with open(PRICING_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_delivery_rows() -> list[dict]:
    with open(DELIVERY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def build_index(reset: bool = True) -> int:
    """(Re)build the ChromaDB pricing collection. Returns number of rows indexed."""
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_fn()
    )

    rows = load_pricing_rows()
    documents, metadatas, ids = [], [], []
    for i, row in enumerate(rows):
        documents.append(_row_to_chunk(row))
        metadatas.append(
            {
                "equipment_type": row["equipment_type"],
                "size_class": row["size_class"],
                "weight_class": row["weight_class"],
                "daily_rate": float(row["daily_rate"]),
                "weekly_rate": float(row["weekly_rate"]),
                "monthly_rate": float(row["monthly_rate"]),
                "min_rental_days": int(row["min_rental_days"]),
                "deposit_per_unit": float(row["deposit_per_unit"]),
                "description": row["description"],
            }
        )
        ids.append(f"pricing-{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(rows)


@lru_cache(maxsize=1)
def _collection():
    return _client().get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedding_fn()
    )


def query(text: str, n_results: int = 3) -> list[dict]:
    """Return top-k pricing rows (metadata dicts) most relevant to `text`."""
    collection = _collection()
    if collection.count() == 0:
        # Index not built yet — build it lazily so the demo never hard-fails.
        build_index()
        _collection.cache_clear()
        collection = _collection()

    res = collection.query(query_texts=[text], n_results=n_results)
    metas = res.get("metadatas") or [[]]
    return metas[0] if metas else []


def find_best_match(
    equipment_type: Optional[str],
    size_class: Optional[str],
    free_text: str = "",
) -> Optional[dict]:
    """Find the single best pricing row for the requested equipment + size.

    Combines structured hints with semantic search. The size_class keyword
    ('mini' / 'mid' / 'large') is used to disambiguate among the retrieved rows.
    """
    query_text = " ".join(
        part for part in [size_class, equipment_type, free_text] if part
    ).strip()
    if not query_text:
        return None

    candidates = query(query_text, n_results=5)
    if not candidates:
        return None

    size = (size_class or "").lower().strip()
    # Normalise common synonyms.
    size_synonyms = {
        "small": "mini",
        "compact": "mini",
        "medium": "mid",
        "standard": "mid",
        "big": "large",
        "heavy": "large",
    }
    size = size_synonyms.get(size, size)

    if size:
        for cand in candidates:
            if cand["equipment_type"].lower().startswith(size):
                # also make sure equipment family matches if we know it
                if not equipment_type or _family(equipment_type) in cand["equipment_type"].lower():
                    return cand

    # Fall back to family match on the top candidates.
    if equipment_type:
        fam = _family(equipment_type)
        for cand in candidates:
            if fam in cand["equipment_type"].lower():
                return cand

    return candidates[0]


def _family(equipment_type: str) -> str:
    """Reduce an equipment phrase to its core noun for matching."""
    t = equipment_type.lower()
    families = ["excavator", "skid steer", "skid-steer", "bulldozer", "dozer", "boom lift", "boom", "generator"]
    for fam in families:
        if fam in t:
            return {"dozer": "bulldozer", "skid-steer": "skid steer", "boom": "boom"}.get(fam, fam)
    return t.split()[-1] if t.split() else t


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} pricing rows into ChromaDB at {CHROMA_DIR}")
