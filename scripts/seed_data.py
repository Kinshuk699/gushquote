"""Seed the ChromaDB pricing index from the CSV data.

Run once before starting the server (the server also self-heals if the index is
missing, but running this explicitly gives clear feedback):

    python scripts/seed_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from rag_pipeline import build_index, load_delivery_rows, load_pricing_rows  # noqa: E402


def main() -> None:
    pricing = load_pricing_rows()
    delivery = load_delivery_rows()
    print(f"Loaded {len(pricing)} pricing rows, {len(delivery)} delivery depots.")
    print("Embedding with all-MiniLM-L6-v2 and indexing into ChromaDB...")
    count = build_index(reset=True)
    print(f"Done. Indexed {count} pricing chunks.")
    print("Sample retrieval check:")

    from rag_pipeline import find_best_match

    match = find_best_match("excavator", "mid", "foundation digging")
    if match:
        print(f"  query 'mid excavator' -> {match['equipment_type']} @ ${match['monthly_rate']}/mo")


if __name__ == "__main__":
    main()
