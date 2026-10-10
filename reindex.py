"""
SmartSearch AI - CLI Reindexer
Rebuilds the document inverted index and updates semantic vector embeddings.
Usage:
    python reindex.py
    python reindex.py --force
"""

import sys
import argparse
from search_engine import reindex

def main():
    parser = argparse.ArgumentParser(description="SmartSearch AI Indexer")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of embeddings even if document files have not changed"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("SMARTSEARCH AI — REINDEXING KNOWLEDGE BASE")
    print("=" * 60)

    result = reindex(force=args.force)

    print("\n[RESULT]")
    print(f"Status: {result['status']}")
    print(f"Indexed Documents: {result['documents_count']}")
    print(f"Categories: {', '.join(result['categories'])}")
    print(f"Embedding Model: {result['model']}")
    print(f"Embeddings Generated: {result['has_embeddings']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
