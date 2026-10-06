"""Load LitQA2 through AstaBench as a small Hugging Face smoke test.

Run from the repository root:

    uv run python sk-experiments/load_litqa2.py --split dev
"""

from __future__ import annotations

import argparse
import json
import os

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from astabench.evals.labbench.litqa2.task import load_litqa2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="LitQA2 split to load (default: dev).",
    )
    parser.add_argument(
        "--show-first",
        action="store_true",
        help="Print the first loaded example as formatted JSON.",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host (default: localhost).",
    )
    parser.add_argument(
        "--qdrant-collection", 
        default="litqa2", 
        help="Qdrant collection name (default: litqa2)."
    )
    return parser.parse_args()


def search_by_doi(client: QdrantClient, collection_name: str, doi: str) -> list[dict]:
    response, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doi",  # The exact key in your payload
                    match=models.MatchValue(value=doi)  # Your exact DOI string
                )
            ]
        ),
        limit=1,               # Only fetch the single matching document
        with_payload=True,     # Retrieve metadata fields
        with_vectors=False     # Change to True if you also need the raw embedding vector
    )
    return response

def main() -> None:
    args = parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set in this process.")

    examples = load_litqa2(split=args.split)
    print(f"Loaded {len(examples)} LitQA2 examples from split={args.split!r}.")

    if examples:
        first = examples[0]
        print(f"First example ID: {first.get('id', '<missing>')}")
        print(f"Available fields: {', '.join(sorted(first))}")
        if args.show_first:
            print(json.dumps(first, indent=2, ensure_ascii=False, default=str))
    
    # parse the examples object and check if the doi exists in my
    # Qdrant deployment
    # Connects directly to your self-hosted instance port
    client = QdrantClient(host=args.qdrant_host, port=6333)
    # how_many_in_qdrant = client.count(collection_name=args.qdrant_collection).count
    how_many_from_litqa2 = len(examples)
    # print(f"Qdrant collection '{args.qdrant_collection}' has {how_many_in_qdrant} records.")
    print(f"LitQA2 split '{args.split}' has {how_many_from_litqa2} records.")
    print("Checking if DOIs from LitQA2 exist in Qdrant collection...")
    how_many_exist = 0
    for record in tqdm(examples, desc="Checking DOIs in Qdrant"):
        doi = record.get("sources")[0].split("https://doi.org/")[-1]
        if doi:
            results = search_by_doi(client, args.qdrant_collection, doi)
            if results:
                how_many_exist += 1
        else:
            print("No DOI found for this record.")
    print(f"Found {how_many_exist} DOIs from LitQA2 in Qdrant collection.")

if __name__ == "__main__":
    main()
