"""Command-line entry point."""

import argparse
import json
from pathlib import Path
from src.rag import RegulatoryQA, list_pdf_files


def main():
    parser = argparse.ArgumentParser(description="Search a local library of regulatory PDFs.")
    parser.add_argument("question")
    parser.add_argument("--docs", default="data/regulations")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = list_pdf_files(args.docs)
    if not paths:
        parser.error(f"No PDFs found in {args.docs!r}; add PDFs and try again.")
    result = RegulatoryQA(paths).answer(args.question, args.top_k, args.min_score)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
