"""
Seed the data/corpus directory with sample documents.

Run this script before starting the engine for the first time:
    make seed-corpus
    # or: uv run python scripts/seed_corpus.py

The corpus documents are already committed to the repository under data/corpus/.
This script just validates they exist and reports a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve paths relative to the repo root
REPO_ROOT = Path(__file__).parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
ADVERSARIAL_DIR = REPO_ROOT / "data" / "adversarial_queries"


def main() -> None:
    print("=" * 60)
    print("Agentic Circuit Breaker — Corpus Seed Check")
    print("=" * 60)

    # Check corpus
    corpus_files = sorted(CORPUS_DIR.glob("*.md")) + sorted(CORPUS_DIR.glob("*.txt"))
    if not corpus_files:
        print(f"❌ No corpus documents found in {CORPUS_DIR}")
        print("   The data/corpus/ directory should contain .md or .txt files.")
        sys.exit(1)

    print(f"\n✅ Corpus ({len(corpus_files)} documents):")
    total_chars = 0
    for f in corpus_files:
        content = f.read_text(encoding="utf-8")
        chars = len(content)
        total_chars += chars
        words = len(content.split())
        print(f"   {f.name:<40} {words:>5} words  ({chars:>6} chars)")
    print(f"   {'TOTAL':<40} {'':>5}        ({total_chars:>6} chars)")

    # Check adversarial queries
    adversarial_files = sorted(ADVERSARIAL_DIR.glob("*.jsonl"))
    if not adversarial_files:
        print(f"\n⚠️  No adversarial query files found in {ADVERSARIAL_DIR}")
    else:
        print(f"\n✅ Adversarial queries ({len(adversarial_files)} files):")
        for f in adversarial_files:
            lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            print(f"   {f.name:<40} {len(lines):>3} queries")

    print("\n✅ Seed check complete. You can now start the engine:")
    print("   make dev")
    print()


if __name__ == "__main__":
    main()
