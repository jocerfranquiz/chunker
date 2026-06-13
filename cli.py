"""Command-line entry point. Defaults read from ChunkingConfig fields so the CLI
and library can never drift."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import ExitStack

from pydantic import ValidationError

import chunker as ch


def _default(field: str) -> object:
    """Pull the default for a ChunkingConfig field — single source of truth."""
    return ch.ChunkingConfig.model_fields[field].default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunker",
        description=(
            "Chunk extracted PDF pages (JSONL per pdf-extraction.schema.json) "
            "into embedding-ready chunks (JSONL per output-schema.json). "
            "Token counts use add_special_tokens=False — leave headroom in "
            "max-tokens for the embedding model's special tokens (CLS/SEP)."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=list(ch.STRATEGIES.keys()),
        default=_default("strategy"),
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=_default("max_tokens"),
    )
    parser.add_argument(
        "--overlap",
        type=_nonneg_int,
        default=_default("overlap"),
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=_nonneg_int,
        default=_default("min_chunk_tokens"),
    )
    parser.add_argument(
        "--tokenizer-model",
        default=_default("tokenizer_model"),
        help="HF Hub repo id or path to a local tokenizer.json.",
    )
    parser.add_argument(
        "--cross-page",
        action=argparse.BooleanOptionalAction,
        default=_default("cross_page"),
    )
    parser.add_argument(
        "--on-ocr-needed",
        choices=["skip", "include", "error"],
        default=_default("on_ocr_needed"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=_default("strict"),
        help="Abort on malformed input lines (default: count and continue).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL path (default: stdin).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: stdout).",
    )
    return parser


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _nonneg_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    try:
        cfg = ch.ChunkingConfig(
            strategy=ns.strategy,
            max_tokens=ns.max_tokens,
            overlap=ns.overlap,
            min_chunk_tokens=ns.min_chunk_tokens,
            tokenizer_model=ns.tokenizer_model,
            cross_page=ns.cross_page,
            on_ocr_needed=ns.on_ocr_needed,
            strict=ns.strict,
        )
    except ValidationError as e:
        print(f"chunker: invalid configuration:\n{e}", file=sys.stderr)
        return 2

    try:
        with ExitStack() as stack:
            in_f = stack.enter_context(open(ns.input, encoding="utf-8")) if ns.input else sys.stdin
            out_f = (
                stack.enter_context(open(ns.output, "w", encoding="utf-8"))
                if ns.output
                else sys.stdout
            )
            stats, skipped = ch.process_stream(in_f, out_f, cfg)
    except ch.ChunkingError as e:
        print(f"chunker: {e}", file=sys.stderr)
        return 1

    summary = (
        f"chunker: wrote {stats.chunks_written} chunks from "
        f"{stats.documents_seen} document(s), {stats.pages_chunked}/{stats.pages_seen} pages."
    )
    if stats.invalid_lines:
        summary += f" Invalid input lines: {stats.invalid_lines}."
    if skipped:
        summary += f" Skipped pages: {len(skipped)}."
    print(summary, file=sys.stderr)
    for s in skipped:
        print(
            f"  skipped: doc_id={s.doc_id} page_index={s.page_index} reason={s.reason}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
