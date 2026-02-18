from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .indexer import build_index
from .search import find_similar, find_similar_in_index_with_context, load_runtime_index
from .text_search import TextSearchCache


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-finder",
        description="Fast local image similarity search focused on black/white CNC patterns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Scan folders and build a local index.")
    index_parser.add_argument("--folders", nargs="+", required=True, help="One or more folders to scan.")
    index_parser.add_argument("--output", required=True, help="Directory to save index files.")
    index_parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads for indexing (0 = auto).",
    )

    search_parser = subparsers.add_parser("search", help="Search similar images from the index.")
    search_parser.add_argument("--query", required=True, help="Path to query image.")
    search_parser.add_argument("--index", required=True, help="Index directory created by index command.")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of matches to return.")
    search_parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable OCR text matching and use visual similarity only.",
    )

    gui_parser = subparsers.add_parser("gui", help="Launch drag-and-drop GUI search window.")
    gui_parser.add_argument(
        "--index",
        required=False,
        default=None,
        help="Index directory. If omitted, app uses Local AppData default path.",
    )
    gui_parser.add_argument("--top-k", type=int, default=12, help="Number of matches to return.")

    return parser


def main() -> int:
    _configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "index":
        folders = [Path(p).expanduser().resolve() for p in args.folders]
        output = Path(args.output).expanduser().resolve()
        workers = None if args.workers <= 0 else args.workers
        stats = build_index(folders=folders, output_dir=output, workers=workers)
        print(f"Indexed images: {stats['indexed']}")
        print(f"Reused from last index: {stats.get('reused', 0)}")
        print(f"Skipped unreadable: {stats.get('failed', 0)}")
        print(f"Index saved to: {output}")
        return 0

    if args.command == "search":
        query = Path(args.query).expanduser().resolve()
        index_dir = Path(args.index).expanduser().resolve()
        if args.no_text:
            results = find_similar(index_dir=index_dir, query_image=query, top_k=args.top_k)
            query_text = ""
            text_used = False
        else:
            runtime_index = load_runtime_index(index_dir=index_dir)
            text_cache = TextSearchCache(index_dir=index_dir)
            results, context = find_similar_in_index_with_context(
                index=runtime_index,
                query_image=query,
                top_k=args.top_k,
                text_cache=text_cache,
            )
            query_text = context.query_text
            text_used = context.used_text_rerank

        print(f"Query: {query}")
        if query_text:
            if text_used:
                print(f"OCR text: {query_text}")
            else:
                print(f"OCR text detected (no text matches found): {query_text}")
        print(f"Results (top {len(results)}):")
        for i, result in enumerate(results, start=1):
            print(
                f"{i:02d}. score={result.score:.4f} "
                f"vector={result.vector_score:.4f} hash={result.hash_score:.4f} text={result.text_score:.4f} "
                f"path={result.path}"
            )
        return 0

    if args.command == "gui":
        from .gui import run_gui

        index_dir = None if args.index is None else Path(args.index).expanduser().resolve()
        return run_gui(index_dir=index_dir, top_k=args.top_k)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
