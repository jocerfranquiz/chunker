"""Chunk extracted PDF pages into embedding-ready records.

Procedural library: pure functions over frozen Pydantic records. Public surface:
ChunkingConfig, process_stream, chunk_document, STRATEGIES, plus the record models
and a handful of utilities (concat_pages, spans_for_range, build_section_index,
tokenizer_funcs, ensure_punkt, make_sentence_splitter, split_sentences_nltk).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Annotated, Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__version__ = "0.1.0"
SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
JOINER = "\n\n"

# ─── types and exceptions ────────────────────────────────────────────────────

Strategy: TypeAlias = Literal["page", "fixed-size", "recursive", "section", "sentence"]
OcrPolicy: TypeAlias = Literal["skip", "include", "error"]

TokenCounter: TypeAlias = Callable[[str], int]
TokenOffsets: TypeAlias = Callable[[str], list[tuple[int, int]]]
SentenceSplitter: TypeAlias = Callable[[str], list[tuple[int, int]]]


class ChunkingError(RuntimeError):
    """The only custom exception. Always includes coordinates (doc_id, page_index,
    or input line number) to make debugging large corpora tractable."""


# ─── data contracts ──────────────────────────────────────────────────────────


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TocEntry(_Record):
    text: str
    level: Annotated[int, Field(ge=0)]
    page_index: int | None = None


class DocMetadata(_Record):
    Title: str = ""
    Author: str = ""
    Subject: str = ""
    Keywords: str = ""
    Creator: str = ""
    Producer: str = ""
    CreationDate: str = ""
    ModDate: str = ""


class DocumentRecord(_Record):
    # Tolerate extra fields from the extractor; we only consume the subset below.
    model_config = ConfigDict(extra="ignore", frozen=True)
    record_type: Literal["document"]
    doc_id: str
    source_path: str
    page_count: Annotated[int, Field(ge=0)]
    metadata: DocMetadata
    toc: list[TocEntry] = Field(default_factory=list)


class PageRecord(_Record):
    model_config = ConfigDict(extra="ignore", frozen=True)
    record_type: Literal["page"]
    doc_id: str
    page_index: Annotated[int, Field(ge=0)]
    page_label: str | None
    text: str
    needs_ocr: bool


class ChunkingConfig(_Record):
    strategy: Strategy = "recursive"
    max_tokens: Annotated[int, Field(ge=1)] = 512
    overlap: Annotated[int, Field(ge=0)] = 64
    min_chunk_tokens: Annotated[int, Field(ge=0)] = 20
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    cross_page: bool = False
    on_ocr_needed: OcrPolicy = "skip"
    strict: bool = False

    @model_validator(mode="after")
    def _check_budgets(self) -> ChunkingConfig:
        if self.overlap >= self.max_tokens:
            raise ValueError(f"overlap ({self.overlap}) must be < max_tokens ({self.max_tokens})")
        if self.min_chunk_tokens >= self.max_tokens:
            raise ValueError(
                f"min_chunk_tokens ({self.min_chunk_tokens}) must be "
                f"< max_tokens ({self.max_tokens})"
            )
        return self


class SourceSpan(_Record):
    page_index: Annotated[int, Field(ge=0)]
    page_label: str | None
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=0)]


class SourceDoc(_Record):
    title: str
    source_path: str


class ChunkRecord(_Record):
    record_type: Literal["chunk"] = "chunk"
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    chunk_id: str
    doc_id: str
    chunk_index: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1)]
    strategy: Strategy
    token_count: Annotated[int, Field(ge=1)]
    tokenizer_model: str
    spans: Annotated[list[SourceSpan], Field(min_length=1)]
    section_path: list[str]
    source_doc: SourceDoc


class SkippedPage(_Record):
    doc_id: str
    page_index: Annotated[int, Field(ge=0)]
    reason: Literal["needs_ocr", "empty_text", "invalid_record"]


class RunStats(BaseModel):
    # Mutable accumulator threaded through the streaming pipeline.
    model_config = ConfigDict(extra="forbid")
    documents_seen: int = 0
    pages_seen: int = 0
    pages_chunked: int = 0
    chunks_written: int = 0
    invalid_lines: int = 0


# ─── tokenizer ───────────────────────────────────────────────────────────────


def load_tokenizer(model: str) -> Any:
    """Load by HF Hub repo id or by local tokenizer.json path.

    Local path is tried first so air-gapped runs work; HF Hub download is the
    fallback (which is also how tests stay offline — they pass a local path)."""
    from tokenizers import Tokenizer  # local import: keeps cold-start light

    if Path(model).is_file():
        return Tokenizer.from_file(model)
    return Tokenizer.from_pretrained(model)


def tokenizer_funcs(tok: Any) -> tuple[TokenCounter, TokenOffsets]:
    """Return (count, offsets) closures over a `tokenizers.Tokenizer`.

    Both use add_special_tokens=False — the budget is content tokens only;
    leave headroom for CLS/SEP at embedding time.
    """

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok.encode(text, add_special_tokens=False).ids)

    def offsets(text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        return list(tok.encode(text, add_special_tokens=False).offsets)

    return count, offsets


# ─── sentence splitting (NLTK punkt_tab) ─────────────────────────────────────


_punkt_ready = False


def ensure_punkt() -> None:
    """Provision NLTK's `punkt_tab` data (the modern replacement for the legacy
    `punkt` pickle, which was deprecated for security reasons). Idempotent."""
    global _punkt_ready
    if _punkt_ready:
        return
    import nltk
    from nltk.tokenize.punkt import PunktTokenizer

    try:
        PunktTokenizer()  # cheap; succeeds iff data is on disk
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
            PunktTokenizer()
        except Exception as e:
            raise ChunkingError(
                "NLTK punkt_tab unavailable. To install offline run: "
                "`python -m nltk.downloader punkt_tab`."
            ) from e
    _punkt_ready = True


def make_sentence_splitter() -> SentenceSplitter:
    """Return a span-emitting NLTK sentence splitter. Requires `ensure_punkt`."""
    ensure_punkt()
    from nltk.tokenize.punkt import PunktTokenizer

    tok = PunktTokenizer()

    def splitter(text: str) -> list[tuple[int, int]]:
        if not text.strip():
            return []
        return [(s, e) for s, e in tok.span_tokenize(text)]

    return splitter


def split_sentences_nltk(text: str) -> list[tuple[int, int]]:
    """Convenience wrapper; used in tests and for one-shot callers."""
    return make_sentence_splitter()(text)


# ─── offset utilities ────────────────────────────────────────────────────────

_PageSegment: TypeAlias = tuple[int, int, PageRecord]


def concat_pages(pages: list[PageRecord]) -> tuple[str, list[_PageSegment]]:
    """Join page texts with JOINER. Return (big_text, segments) where each segment
    is (global_start, global_end, page) covering that page's slice of big_text.

    The JOINER falls into the GAPS between segments — no segment includes joiner
    text. `spans_for_range` relies on that invariant."""
    if not pages:
        return "", []
    parts: list[str] = []
    segments: list[_PageSegment] = []
    cursor = 0
    for i, p in enumerate(pages):
        if i > 0:
            parts.append(JOINER)
            cursor += len(JOINER)
        start = cursor
        parts.append(p.text)
        cursor += len(p.text)
        segments.append((start, cursor, p))
    return "".join(parts), segments


def spans_for_range(segments: list[_PageSegment], start: int, end: int) -> list[SourceSpan]:
    """Map a [start, end) range in the concatenated text back to per-page
    SourceSpans. Joiner gaps are skipped; a span is emitted only where the overlap
    with a page segment is non-empty."""
    out: list[SourceSpan] = []
    for seg_start, seg_end, page in segments:
        lo = max(start, seg_start)
        hi = min(end, seg_end)
        if lo >= hi:
            continue
        out.append(
            SourceSpan(
                page_index=page.page_index,
                page_label=page.page_label,
                char_start=lo - seg_start,
                char_end=hi - seg_start,
            )
        )
    return out


# ─── section index ───────────────────────────────────────────────────────────


def build_section_index(toc: list[TocEntry], page_count: int) -> list[list[str]]:
    """Map each 0..page_count-1 page to a section path (outermost-first list of
    TOC titles). Walks the outline in document order with a level-truncating
    stack. Entries with page_index=None are ignored. Pages before the first TOC
    entry get an empty path."""
    if page_count <= 0:
        return []
    paths: list[list[str]] = [[] for _ in range(page_count)]
    stack: list[tuple[int, str]] = []
    transitions: list[tuple[int, list[str]]] = []
    for entry in toc:
        if entry.page_index is None:
            continue
        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        stack.append((entry.level, entry.text))
        transitions.append((entry.page_index, [t for _, t in stack]))
    if not transitions:
        return paths
    for i, (start, path) in enumerate(transitions):
        end = transitions[i + 1][0] if i + 1 < len(transitions) else page_count
        for p in range(max(start, 0), min(end, page_count)):
            paths[p] = list(path)
    return paths


# ─── internal piece type ─────────────────────────────────────────────────────


@dataclass
class _Piece:
    """Provisional chunk produced by a strategy; mutated by merge_runts and the
    orchestrator (section back-fill). Never crosses the public API."""

    text: str
    spans: list[SourceSpan]
    section_path: list[str] = field(default_factory=list)
    token_count: int = 0


# ─── splitter primitives (for recursive cascade) ─────────────────────────────


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n{2,}", text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _split_words(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _token_windows(
    text: str, base: int, cfg: ChunkingConfig, offsets: TokenOffsets
) -> list[tuple[int, int]]:
    """Sliding token-index windows over `text`. Stops as soon as a window covers
    to the end (avoids emitting a tail-only window that would just become a runt).
    Returned ranges are in absolute coordinates (base + local offset)."""
    offs = offsets(text)
    if not offs:
        return []
    stride = cfg.max_tokens - cfg.overlap
    if stride <= 0:
        stride = cfg.max_tokens
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(offs):
        end_idx = min(i + cfg.max_tokens, len(offs))
        out.append((base + offs[i][0], base + offs[end_idx - 1][1]))
        if end_idx >= len(offs):
            break
        i += stride
    return out


def _units(pages: list[PageRecord], cross_page: bool) -> list[list[PageRecord]]:
    """Group pages into chunking units. cross_page=True → one unit per call;
    cross_page=False → one unit per page (chunks never straddle pages)."""
    if cross_page:
        return [pages]
    return [[p] for p in pages]


def _emit_piece(
    unit_text: str,
    segments: list[_PageSegment],
    s: int,
    e: int,
    count: TokenCounter,
) -> _Piece:
    piece_text = unit_text[s:e]
    return _Piece(
        text=piece_text,
        spans=spans_for_range(segments, s, e),
        token_count=count(piece_text),
    )


# ─── strategy: page ──────────────────────────────────────────────────────────


def chunk_by_page(
    _doc: DocumentRecord,
    pages: list[PageRecord],
    _cfg: ChunkingConfig,
    count: TokenCounter,
    _offsets: TokenOffsets,
    _sentences: SentenceSplitter,
) -> list[_Piece]:
    """One piece per page, never split. The orchestrator already filtered
    empty-text and OCR-skipped pages."""
    pieces: list[_Piece] = []
    for p in pages:
        spans = [
            SourceSpan(
                page_index=p.page_index,
                page_label=p.page_label,
                char_start=0,
                char_end=len(p.text),
            )
        ]
        pieces.append(_Piece(text=p.text, spans=spans, token_count=count(p.text)))
    return pieces


# ─── strategy: fixed-size ────────────────────────────────────────────────────


def chunk_fixed_size(
    _doc: DocumentRecord,
    pages: list[PageRecord],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    _sentences: SentenceSplitter,
) -> list[_Piece]:
    return _do_fixed_size(_units(pages, cfg.cross_page), cfg, count, offsets)


def _do_fixed_size(
    units: list[list[PageRecord]],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    for unit_pages in units:
        unit_text, segments = concat_pages(unit_pages)
        for s, e in _token_windows(unit_text, 0, cfg, offsets):
            pieces.append(_emit_piece(unit_text, segments, s, e, count))
    return pieces


# ─── strategy: sentence ──────────────────────────────────────────────────────


def chunk_by_sentence(
    _doc: DocumentRecord,
    pages: list[PageRecord],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> list[_Piece]:
    return _do_sentence(_units(pages, cfg.cross_page), cfg, count, offsets, sentences)


def _do_sentence(
    units: list[list[PageRecord]],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    for unit_pages in units:
        unit_text, segments = concat_pages(unit_pages)
        sent_spans = sentences(unit_text)
        if not sent_spans:
            continue
        sent_tokens = [count(unit_text[s:e]) for s, e in sent_spans]
        n = len(sent_spans)
        i = 0
        while i < n:
            # Single over-budget sentence: fall back to token-window split.
            if sent_tokens[i] > cfg.max_tokens:
                ss, se = sent_spans[i]
                for s, e in _token_windows(unit_text[ss:se], ss, cfg, offsets):
                    pieces.append(_emit_piece(unit_text, segments, s, e, count))
                i += 1
                continue
            # Greedy pack while budget holds.
            j = i
            total = 0
            while j < n and sent_tokens[j] + total <= cfg.max_tokens:
                total += sent_tokens[j]
                j += 1
            # Guard: progress always made because sent_tokens[i] <= max_tokens.
            ss = sent_spans[i][0]
            se = sent_spans[j - 1][1]
            pieces.append(_emit_piece(unit_text, segments, ss, se, count))
            # Overlap: walk back through trailing sentences fitting in cfg.overlap.
            if cfg.overlap > 0 and j > i:
                k = j
                accum = 0
                while k > i and accum + sent_tokens[k - 1] <= cfg.overlap:
                    accum += sent_tokens[k - 1]
                    k -= 1
                # Always make progress (avoid k == i which would re-emit the pack).
                i = max(k, i + 1)
            else:
                i = j
    return pieces


# ─── strategy: recursive ─────────────────────────────────────────────────────


def chunk_recursive(
    _doc: DocumentRecord,
    pages: list[PageRecord],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> list[_Piece]:
    return _do_recursive(_units(pages, cfg.cross_page), cfg, count, offsets, sentences)


def _do_recursive(
    units: list[list[PageRecord]],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    for unit_pages in units:
        unit_text, segments = concat_pages(unit_pages)
        if not unit_text:
            continue
        for s, e in _recursive_ranges(unit_text, 0, cfg, count, offsets, sentences, level=0):
            pieces.append(_emit_piece(unit_text, segments, s, e, count))
    return pieces


def _recursive_ranges(
    text: str,
    base: int,
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
    *,
    level: int,
) -> list[tuple[int, int]]:
    """The cascade: paragraphs → sentences → words → token-window fallback.

    Greedy-packs parts at the current level. A single over-budget part recurses
    to the next level. Returns absolute (base-shifted) ranges. Order-preserving.
    """
    if count(text) <= cfg.max_tokens:
        return [(base, base + len(text))]

    split_fns: list[Callable[[str], list[tuple[int, int]]]] = [
        _split_paragraphs,
        sentences,
        _split_words,
    ]
    if level >= len(split_fns):
        return _token_windows(text, base, cfg, offsets)

    parts = split_fns[level](text)
    if len(parts) <= 1:
        return _recursive_ranges(text, base, cfg, count, offsets, sentences, level=level + 1)

    out: list[tuple[int, int]] = []
    buf: tuple[int, int] | None = None  # (start, end) in local text coords
    for ps, pe in parts:
        part_text = text[ps:pe]
        if count(part_text) > cfg.max_tokens:
            if buf is not None:
                out.append((base + buf[0], base + buf[1]))
                buf = None
            out.extend(
                _recursive_ranges(
                    part_text,
                    base + ps,
                    cfg,
                    count,
                    offsets,
                    sentences,
                    level=level + 1,
                )
            )
            continue
        candidate_start = buf[0] if buf is not None else ps
        if count(text[candidate_start:pe]) <= cfg.max_tokens:
            buf = (candidate_start, pe)
        else:
            if buf is not None:
                out.append((base + buf[0], base + buf[1]))
            buf = (ps, pe)
    if buf is not None:
        out.append((base + buf[0], base + buf[1]))
    return out


# ─── strategy: section ───────────────────────────────────────────────────────


def chunk_by_section(
    doc: DocumentRecord,
    pages: list[PageRecord],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> list[_Piece]:
    upper = max(doc.page_count, max((p.page_index for p in pages), default=-1) + 1)
    section_index = build_section_index(doc.toc, upper)

    # Group consecutive pages by identical section path.
    groups: list[tuple[list[str], list[PageRecord]]] = []
    current_path: list[str] | None = None
    current_group: list[PageRecord] = []
    for p in pages:
        path = section_index[p.page_index] if p.page_index < len(section_index) else []
        if current_path is None or path != current_path:
            if current_group:
                groups.append((current_path or [], current_group))
            current_path = path
            current_group = [p]
        else:
            current_group.append(p)
    if current_group:
        groups.append((current_path or [], current_group))

    pieces: list[_Piece] = []
    for path, grp in groups:
        units = _units(grp, cfg.cross_page)
        sub = _do_recursive(units, cfg, count, offsets, sentences)
        for piece in sub:
            piece.section_path = list(path)
        pieces.extend(sub)
    return pieces


# ─── strategy registry ───────────────────────────────────────────────────────


StrategyFn: TypeAlias = Callable[
    [
        DocumentRecord,
        list[PageRecord],
        ChunkingConfig,
        TokenCounter,
        TokenOffsets,
        SentenceSplitter,
    ],
    list[_Piece],
]

STRATEGIES: dict[Strategy, StrategyFn] = {
    "page": chunk_by_page,
    "fixed-size": chunk_fixed_size,
    "sentence": chunk_by_sentence,
    "recursive": chunk_recursive,
    "section": chunk_by_section,
}

# Runtime guarantee that registry covers every literal value in `Strategy`.
assert set(STRATEGIES.keys()) == set(get_args(Strategy)), (
    f"STRATEGIES keys {set(STRATEGIES.keys())} != Strategy literals {set(get_args(Strategy))}"
)


# ─── runt merging ────────────────────────────────────────────────────────────


def merge_runts(
    pieces: list[_Piece],
    cfg: ChunkingConfig,
    count: TokenCounter,
    pages_by_idx: dict[int, PageRecord],
) -> list[_Piece]:
    """Merge a sub-min-chunk-tokens piece into its previous neighbor, subject to
    these guard-rails:

      * never cross a section_path boundary,
      * with cross_page=False, never cross a page boundary,
      * merged token_count must stay ≤ max_tokens + min_chunk_tokens
        (we tolerate a controlled overshoot rather than drop content).

    Rebuilds merged text from spans + source pages, so an overlap window collapses
    cleanly (no double-counting at the boundary)."""
    if not pieces:
        return []
    out: list[_Piece] = [pieces[0]]
    for p in pieces[1:]:
        prev = out[-1]
        if p.token_count >= cfg.min_chunk_tokens:
            out.append(p)
            continue
        if prev.section_path != p.section_path:
            out.append(p)
            continue
        prev_pages = {s.page_index for s in prev.spans}
        new_pages = {s.page_index for s in p.spans}
        if not cfg.cross_page and prev_pages != new_pages:
            out.append(p)
            continue

        merged_spans = _union_spans(prev.spans, p.spans)
        merged_text = _text_from_spans(merged_spans, pages_by_idx)
        merged_count = count(merged_text)
        if merged_count > cfg.max_tokens + cfg.min_chunk_tokens:
            out.append(p)
            continue
        out[-1] = _Piece(
            text=merged_text,
            spans=merged_spans,
            section_path=list(prev.section_path),
            token_count=merged_count,
        )
    return out


def _union_spans(a: list[SourceSpan], b: list[SourceSpan]) -> list[SourceSpan]:
    """Merge two contiguous span lists, extending the last range on a page where
    they meet so overlapping windows collapse to one span."""
    out: list[SourceSpan] = list(a)
    for sp in b:
        if out and out[-1].page_index == sp.page_index:
            last = out[-1]
            out[-1] = SourceSpan(
                page_index=last.page_index,
                page_label=last.page_label,
                char_start=min(last.char_start, sp.char_start),
                char_end=max(last.char_end, sp.char_end),
            )
        else:
            out.append(sp)
    return out


def _text_from_spans(spans: list[SourceSpan], pages_by_idx: dict[int, PageRecord]) -> str:
    """Rebuild chunk text from its spans + source pages. JOINER between pages."""
    parts: list[str] = []
    prev_page: int | None = None
    for sp in spans:
        if prev_page is not None and prev_page != sp.page_index:
            parts.append(JOINER)
        page = pages_by_idx[sp.page_index]
        parts.append(page.text[sp.char_start : sp.char_end])
        prev_page = sp.page_index
    return "".join(parts)


# ─── orchestrator ────────────────────────────────────────────────────────────


def chunk_document(
    doc: DocumentRecord,
    pages: list[PageRecord],
    cfg: ChunkingConfig,
    count: TokenCounter,
    offsets: TokenOffsets,
    sentences: SentenceSplitter,
) -> tuple[list[ChunkRecord], list[SkippedPage]]:
    """Apply OCR + empty-text policy, run the configured strategy, back-fill
    section paths, merge runts, materialize ChunkRecords with stable IDs."""
    pages = sorted(pages, key=lambda p: p.page_index)
    skipped: list[SkippedPage] = []
    usable: list[PageRecord] = []
    for p in pages:
        if p.needs_ocr:
            if cfg.on_ocr_needed == "skip":
                skipped.append(
                    SkippedPage(doc_id=doc.doc_id, page_index=p.page_index, reason="needs_ocr")
                )
                continue
            if cfg.on_ocr_needed == "error":
                raise ChunkingError(
                    f"needs_ocr page in doc {doc.doc_id!r} page_index {p.page_index} "
                    f"and on_ocr_needed='error'"
                )
            # "include": fall through to normal processing.
        if not p.text.strip():
            skipped.append(
                SkippedPage(doc_id=doc.doc_id, page_index=p.page_index, reason="empty_text")
            )
            continue
        usable.append(p)

    if not usable:
        return [], skipped

    pieces = STRATEGIES[cfg.strategy](doc, usable, cfg, count, offsets, sentences)

    # Back-fill section_path from the chunk's first span's page (section strategy
    # already set its own; back-fill is idempotent in that case).
    upper = max(doc.page_count, max(p.page_index for p in pages) + 1) if pages else 0
    section_index = build_section_index(doc.toc, upper)
    for piece in pieces:
        if not piece.section_path and piece.spans:
            first_page = piece.spans[0].page_index
            if 0 <= first_page < len(section_index):
                piece.section_path = list(section_index[first_page])

    pages_by_idx = {p.page_index: p for p in pages}
    pieces = merge_runts(pieces, cfg, count, pages_by_idx)

    source_doc = SourceDoc(title=doc.metadata.Title, source_path=doc.source_path)
    chunks: list[ChunkRecord] = []
    idx = 0
    for piece in pieces:
        if not piece.text:
            continue  # belt-and-braces; ChunkRecord requires min_length=1
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc.doc_id}::c{idx:04d}",
                doc_id=doc.doc_id,
                chunk_index=idx,
                text=piece.text,
                strategy=cfg.strategy,
                token_count=piece.token_count,
                tokenizer_model=cfg.tokenizer_model,
                spans=piece.spans,
                section_path=piece.section_path,
                source_doc=source_doc,
            )
        )
        idx += 1
    return chunks, skipped


# ─── streaming IO and manifest ───────────────────────────────────────────────


_InputRecord: TypeAlias = DocumentRecord | PageRecord


def iter_records(lines: Iterable[str], strict: bool, stats: RunStats) -> Iterator[_InputRecord]:
    """Parse and validate JSONL lines into typed records.

    Non-strict mode counts malformed/invalid lines in stats.invalid_lines and
    continues. Strict mode raises ChunkingError naming the 1-based line number.
    Empty/whitespace-only lines are silently skipped."""
    for n, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            if strict:
                raise ChunkingError(f"line {n}: invalid JSON: {e.msg}") from e
            stats.invalid_lines += 1
            continue

        rt = payload.get("record_type") if isinstance(payload, dict) else None
        try:
            if rt == "document":
                yield DocumentRecord.model_validate(payload)
            elif rt == "page":
                yield PageRecord.model_validate(payload)
            else:
                raise ValueError(f"unknown or missing record_type: {rt!r}")
        except (ValidationError, ValueError) as e:
            if strict:
                raise ChunkingError(f"line {n}: {e}") from e
            stats.invalid_lines += 1
            continue


def group_by_document(
    records: Iterable[_InputRecord],
    stats: RunStats,
    skipped: list[SkippedPage],
) -> Iterator[tuple[DocumentRecord, list[PageRecord]]]:
    """Buffer pages per current document; yield when the next document begins or
    the stream ends. Pages with no matching prior document are orphans: counted
    in invalid_lines and added to `skipped` with reason 'invalid_record'."""
    current_doc: DocumentRecord | None = None
    pages: list[PageRecord] = []
    for rec in records:
        if isinstance(rec, DocumentRecord):
            if current_doc is not None:
                yield current_doc, pages
            current_doc = rec
            pages = []
        else:
            if current_doc is None or rec.doc_id != current_doc.doc_id:
                stats.invalid_lines += 1
                skipped.append(
                    SkippedPage(
                        doc_id=rec.doc_id,
                        page_index=rec.page_index,
                        reason="invalid_record",
                    )
                )
                continue
            pages.append(rec)
    if current_doc is not None:
        yield current_doc, pages


def process_stream(
    in_f: IO[str],
    out_f: IO[str],
    cfg: ChunkingConfig,
    *,
    created_at: str | None = None,
) -> tuple[RunStats, list[SkippedPage]]:
    """Top-level pipeline: read JSONL from in_f, emit chunk JSONL + manifest to
    out_f. created_at is injectable so tests can assert byte-identical output;
    production callers leave it None and get UTC now."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    stats = RunStats()
    skipped: list[SkippedPage] = []

    tok = load_tokenizer(cfg.tokenizer_model)
    count, offsets = tokenizer_funcs(tok)

    if cfg.strategy in {"sentence", "recursive", "section"}:
        sentences: SentenceSplitter = make_sentence_splitter()
    else:
        sentences = _unused_sentences

    records = iter_records(in_f, cfg.strict, stats)
    for doc, pages in group_by_document(records, stats, skipped):
        stats.documents_seen += 1
        stats.pages_seen += len(pages)
        chunks, doc_skipped = chunk_document(doc, pages, cfg, count, offsets, sentences)
        skipped.extend(doc_skipped)
        stats.pages_chunked += max(0, len(pages) - len(doc_skipped))
        for c in chunks:
            out_f.write(_dump_line(c.model_dump()))
            stats.chunks_written += 1

    out_f.write(_dump_line(build_manifest(cfg, stats, skipped, created_at)))
    return stats, skipped


def _unused_sentences(_text: str) -> list[tuple[int, int]]:
    """Sentinel splitter for strategies that don't need sentences."""
    return []


def _dump_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False) + "\n"


def build_manifest(
    cfg: ChunkingConfig,
    stats: RunStats,
    skipped: list[SkippedPage],
    created_at: str,
) -> dict[str, Any]:
    return {
        "record_type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "tool": {"name": "chunker", "version": __version__},
        "config": cfg.model_dump(),
        "stats": stats.model_dump(),
        "skipped_pages": [s.model_dump() for s in skipped],
    }


__all__ = [
    "JOINER",
    "SCHEMA_VERSION",
    "STRATEGIES",
    "ChunkRecord",
    "ChunkingConfig",
    "ChunkingError",
    "DocMetadata",
    "DocumentRecord",
    "OcrPolicy",
    "PageRecord",
    "RunStats",
    "SentenceSplitter",
    "SkippedPage",
    "SourceDoc",
    "SourceSpan",
    "Strategy",
    "TocEntry",
    "TokenCounter",
    "TokenOffsets",
    "__version__",
    "build_manifest",
    "build_section_index",
    "chunk_by_page",
    "chunk_by_section",
    "chunk_by_sentence",
    "chunk_document",
    "chunk_fixed_size",
    "chunk_recursive",
    "concat_pages",
    "ensure_punkt",
    "group_by_document",
    "iter_records",
    "load_tokenizer",
    "make_sentence_splitter",
    "merge_runts",
    "process_stream",
    "spans_for_range",
    "split_sentences_nltk",
    "tokenizer_funcs",
]
