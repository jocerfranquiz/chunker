"""Tests for the chunker. Procedural tests; module-level builders, minimal fixtures.

Layout follows guide §5: validation → offsets → strategies (page, fixed-size, sentence,
recursive, section) → pipeline/IO/manifest → CLI/tool → NLTK integration.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Make the project root importable (flat layout: chunker.py, cli.py, tool.py at root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chunker as ch
import cli
import tool as tl

# ─── fixtures and test doubles (§5.1) ────────────────────────────────────────


@pytest.fixture(scope="session")
def offline_tokenizer() -> Tokenizer:
    """WordLevel + WhitespaceSplit: every whitespace-separated word is exactly one
    token, regardless of vocabulary. Real `tokenizers.Tokenizer`, so production code
    paths are exercised; just no network."""
    return Tokenizer.from_file(_tokenizer_path())


_TOKENIZER_PATH_CACHE: dict[str, str] = {}


def _tokenizer_path() -> str:
    """Persist a tiny WordLevel tokenizer to a temp file once; return its path.
    Used by tests that exercise process_stream / CLI / tool (which call
    load_tokenizer); the local-file branch keeps the network out of tests."""
    if "path" not in _TOKENIZER_PATH_CACHE:
        from tempfile import NamedTemporaryFile

        tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]"])  # type: ignore[no-untyped-call]
        tok.train_from_iterator(["placeholder vocabulary text"], trainer)
        with NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tok.save(path)
        _TOKENIZER_PATH_CACHE["path"] = path
    return _TOKENIZER_PATH_CACHE["path"]


@pytest.fixture(scope="session")
def funcs(offline_tokenizer: Tokenizer) -> tuple[ch.TokenCounter, ch.TokenOffsets]:
    return ch.tokenizer_funcs(offline_tokenizer)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def regex_sentences(text: str) -> list[tuple[int, int]]:
    """Trivial sentence splitter for unit tests of strategy logic."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((cursor, m.start()))
        cursor = m.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


def _try_ensure_punkt() -> bool:
    try:
        ch.ensure_punkt()
    except Exception:
        return False
    return True


nltk_ready = pytest.mark.skipif(not _try_ensure_punkt(), reason="punkt_tab unavailable offline")


# ─── record builders ─────────────────────────────────────────────────────────


def make_doc(
    doc_id: str = "d1",
    *,
    source_path: str = "/x.pdf",
    title: str = "",
    page_count: int = 1,
    toc: list[dict[str, Any]] | None = None,
) -> ch.DocumentRecord:
    return ch.DocumentRecord(
        record_type="document",
        doc_id=doc_id,
        source_path=source_path,
        page_count=page_count,
        metadata=ch.DocMetadata(Title=title),
        toc=[ch.TocEntry(**e) for e in (toc or [])],
    )


def make_page(
    doc_id: str = "d1",
    page_index: int = 0,
    text: str = "",
    *,
    page_label: str | None = None,
    needs_ocr: bool = False,
) -> ch.PageRecord:
    return ch.PageRecord(
        record_type="page",
        doc_id=doc_id,
        page_index=page_index,
        page_label=page_label,
        text=text,
        needs_ocr=needs_ocr,
    )


def make_config(**over: Any) -> ch.ChunkingConfig:
    # Test-friendly defaults: overlap=0 and min_chunk_tokens=1 don't collide with
    # the small max_tokens values most strategy tests use. Tests that care about
    # those fields set them explicitly.
    defaults: dict[str, Any] = {"overlap": 0, "min_chunk_tokens": 1}
    return ch.ChunkingConfig(**{**defaults, **over})


def chunk(
    doc: ch.DocumentRecord,
    pages: list[ch.PageRecord],
    cfg: ch.ChunkingConfig,
    funcs_: tuple[ch.TokenCounter, ch.TokenOffsets],
    sentences: ch.SentenceSplitter | None = None,
) -> tuple[list[ch.ChunkRecord], list[ch.SkippedPage]]:
    count, offsets = funcs_
    return ch.chunk_document(doc, pages, cfg, count, offsets, sentences or regex_sentences)


# ─── 5.2 model & config validation ───────────────────────────────────────────


def test_01_records_parse_and_discriminate() -> None:
    doc = make_doc()
    page = make_page(text="hello")
    assert doc.record_type == "document"
    assert page.record_type == "page"
    # round-trip JSON
    assert ch.PageRecord.model_validate_json(page.model_dump_json()).text == "hello"


def test_02_negative_page_index_rejected() -> None:
    with pytest.raises(ValidationError):
        make_page(page_index=-1)


def test_03_config_rejects_bad_budgets() -> None:
    with pytest.raises(ValidationError):
        make_config(max_tokens=10, overlap=10)
    with pytest.raises(ValidationError):
        make_config(max_tokens=10, min_chunk_tokens=10)
    with pytest.raises(ValidationError):
        make_config(max_tokens=10, overlap=15)


def test_04_extra_keys_policy() -> None:
    # output models forbid extra
    with pytest.raises(ValidationError):
        ch.SourceSpan(  # type: ignore[call-arg]
            page_index=0, page_label=None, char_start=0, char_end=1, extra="x"
        )
    # input models tolerate extra (extractor may add fields)
    payload = {
        "record_type": "page",
        "doc_id": "d",
        "page_index": 0,
        "page_label": None,
        "text": "",
        "needs_ocr": False,
        "some_extra_field": "hello",
    }
    ch.PageRecord.model_validate(payload)  # must not raise


# ─── 5.3 offset utilities ────────────────────────────────────────────────────


def test_05_tokenizer_funcs(funcs: tuple[ch.TokenCounter, ch.TokenOffsets]) -> None:
    count, offsets = funcs
    text = "alpha beta gamma"
    assert count(text) == 3
    offs = offsets(text)
    assert len(offs) == 3
    # monotone non-decreasing
    for (s1, e1), (s2, e2) in zip(offs, offs[1:]):
        assert e1 <= s2
        assert s1 < e1 and s2 < e2
    # round-trip each token
    for s, e in offs:
        assert text[s:e].strip() != ""


def test_06_concat_pages_offsets() -> None:
    p0 = make_page(page_index=0, text="hello world")
    p1 = make_page(page_index=1, text="second page")
    big, segs = ch.concat_pages([p0, p1])
    assert big == "hello world\n\nsecond page"
    # one segment per page
    assert len(segs) == 2
    # range fully inside page 0
    spans = ch.spans_for_range(segs, 0, 5)  # "hello"
    assert len(spans) == 1 and spans[0].page_index == 0
    assert spans[0].char_start == 0 and spans[0].char_end == 5
    # range straddling the join → two spans, joiner excluded
    spans = ch.spans_for_range(segs, 6, len(big))  # "world\n\nsecond page"
    assert len(spans) == 2
    s0, s1 = spans
    assert s0.page_index == 0 and s0.char_start == 6 and s0.char_end == 11  # "world"
    assert s1.page_index == 1 and s1.char_start == 0 and s1.char_end == 11  # "second page"


# ─── 5.4 strategy: page ──────────────────────────────────────────────────────


def test_07_page_basic(funcs: tuple[ch.TokenCounter, ch.TokenOffsets]) -> None:
    doc = make_doc(page_count=3)
    pages = [make_page(page_index=i, text=f"page text {i}") for i in range(3)]
    cfg = make_config(strategy="page", min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        assert c.text == pages[i].text
        assert c.chunk_index == i
        assert re.match(r"^d1::c\d{4,}$", c.chunk_id)
        assert len(c.spans) == 1
        sp = c.spans[0]
        assert sp.page_index == i
        assert sp.char_start == 0 and sp.char_end == len(pages[i].text)
    assert chunks[0].chunk_id == "d1::c0000"
    assert chunks[2].chunk_id == "d1::c0002"


def test_08_page_strategy_never_splits(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    long_text = " ".join(f"w{i}" for i in range(50))  # 50 tokens
    doc = make_doc(page_count=1)
    pages = [make_page(text=long_text)]
    cfg = make_config(strategy="page", max_tokens=10, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    assert len(chunks) == 1
    assert chunks[0].text == long_text
    assert chunks[0].token_count == 50  # exceeds max_tokens by design


def test_09_whitespace_only_page_reported_empty(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    doc = make_doc(page_count=2)
    pages = [
        make_page(page_index=0, text="real content here"),
        make_page(page_index=1, text="   \n  \t "),
    ]
    cfg = make_config(strategy="page", min_chunk_tokens=1)
    chunks, skipped = chunk(doc, pages, cfg, funcs)
    assert len(chunks) == 1
    assert chunks[0].spans[0].page_index == 0
    assert any(s.doc_id == "d1" and s.page_index == 1 and s.reason == "empty_text" for s in skipped)


# ─── 5.5 strategy: fixed-size ────────────────────────────────────────────────


def test_10_fixed_size_windows(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = " ".join(f"w{i}" for i in range(10))  # 10 tokens
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="fixed-size", max_tokens=4, overlap=1, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    # stride=3 → token ranges [0:4], [3:7], [6:10]; the last window covers to the
    # end so iteration stops (no runt at [9:10])
    assert len(chunks) == 3
    _, offsets = funcs
    offs = offsets(text)
    expected = [(0, 4), (3, 7), (6, 10)]
    for c, (lo, hi) in zip(chunks, expected):
        want = text[offs[lo][0] : offs[hi - 1][1]]
        assert c.text == want
        assert c.spans[0].char_start == offs[lo][0]
        assert c.spans[0].char_end == offs[hi - 1][1]


def test_11_fixed_size_respects_budget(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = " ".join(f"w{i}" for i in range(20))
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="fixed-size", max_tokens=5, overlap=0, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    assert all(c.token_count <= cfg.max_tokens for c in chunks)


def test_12_fixed_size_overlap_zero_tiles(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = " ".join(f"w{i}" for i in range(8))  # 8 tokens
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="fixed-size", max_tokens=4, overlap=0, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    # token ranges [0:4], [4:8] — disjoint, tile the whole sequence
    assert len(chunks) == 2
    ends = [c.spans[0].char_end for c in chunks]
    starts = [c.spans[0].char_start for c in chunks[1:]]
    # adjacent windows: end of one <= start of next (excluding inter-token whitespace)
    for end, nxt_start in zip(ends, starts):
        assert end <= nxt_start


def test_13_fixed_size_runt_merged(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    # 11 tokens with max=4, overlap=1, stride=3:
    #   windows [0:4], [3:7], [6:10], [9:11] — last has 2 tokens, a runt
    text = " ".join(f"w{i}" for i in range(11))
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="fixed-size", max_tokens=4, overlap=1, min_chunk_tokens=3)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    # the trailing 2-token runt is merged into the previous chunk
    assert len(chunks) == 3
    # the merged chunk may exceed max_tokens, but by at most min_chunk_tokens
    for c in chunks:
        assert c.token_count <= cfg.max_tokens + cfg.min_chunk_tokens


def test_14_cross_page_toggle(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    p0 = make_page(page_index=0, text="alpha beta")  # 2 tokens
    p1 = make_page(page_index=1, text="gamma delta")  # 2 tokens
    doc = make_doc(page_count=2)
    # cross_page=False: each page is its own unit; budget plenty so one chunk per page
    cfg = make_config(
        strategy="fixed-size",
        max_tokens=10,
        overlap=0,
        min_chunk_tokens=1,
        cross_page=False,
    )
    chunks, _ = chunk(doc, [p0, p1], cfg, funcs)
    assert len(chunks) == 2
    assert {c.spans[0].page_index for c in chunks} == {0, 1}
    # cross_page=True: pages concatenate, single chunk spans both
    cfg2 = make_config(
        strategy="fixed-size",
        max_tokens=10,
        overlap=0,
        min_chunk_tokens=1,
        cross_page=True,
    )
    chunks2, _ = chunk(doc, [p0, p1], cfg2, funcs)
    assert len(chunks2) == 1
    spans = chunks2[0].spans
    assert len(spans) == 2
    assert [s.page_index for s in spans] == [0, 1]
    # joiner not included in either span; per-page local offsets are correct
    assert spans[0].char_start == 0 and spans[0].char_end == len(p0.text)
    assert spans[1].char_start == 0 and spans[1].char_end == len(p1.text)


# ─── 5.6 strategy: sentence ──────────────────────────────────────────────────


def test_15_sentence_packing(funcs: tuple[ch.TokenCounter, ch.TokenOffsets]) -> None:
    # 4 sentences, each "AAA BBB CCC." → 3 tokens (period attaches to last word
    # under WhitespaceSplit). max_tokens=7, overlap=0 → packs 2+2.
    text = "AAA BBB CCC. DDD EEE FFF. GGG HHH III. JJJ KKK LLL."
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="sentence", max_tokens=7, overlap=0, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    assert len(chunks) == 2
    # first chunk starts at sentence 1, ends at sentence 2's period
    assert chunks[0].text.startswith("AAA")
    assert chunks[0].text.endswith("FFF.")
    assert chunks[1].text.startswith("GGG")
    assert chunks[1].text.endswith("LLL.")


def test_16_sentence_overlap(funcs: tuple[ch.TokenCounter, ch.TokenOffsets]) -> None:
    text = "AAA BBB CCC. DDD EEE FFF. GGG HHH III. JJJ KKK LLL."
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="sentence", max_tokens=7, overlap=3, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    # last sentence of chunk k re-appears at the head of chunk k+1
    assert "DDD EEE FFF." in chunks[0].text
    assert chunks[1].text.startswith("DDD")
    # overlap budgeted: a chunk doesn't carry forward more than overlap-many tokens
    for c in chunks:
        assert c.token_count <= cfg.max_tokens


def test_17_long_sentence_falls_back(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    long_sent = " ".join(f"w{i}" for i in range(20)) + "."
    short_sent = "Tiny tail end."
    text = f"{long_sent} {short_sent}"
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="sentence", max_tokens=6, overlap=0, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    # no infinite loop, no over-budget chunk
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= cfg.max_tokens


# ─── 5.7 strategy: recursive ─────────────────────────────────────────────────


def test_18_recursive_within_budget(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = "First short paragraph.\n\nSecond short paragraph."
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="recursive", max_tokens=50, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    assert len(chunks) == 1


def test_19a_recursive_paragraph_then_sentence(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    big_para = "AAA BBB CCC. DDD EEE FFF. GGG HHH III. JJJ KKK LLL."  # 12 tokens
    small_para = "Tiny."
    text = f"{big_para}\n\n{small_para}"
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="recursive", max_tokens=7, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    # the big paragraph splits at sentence boundaries
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= cfg.max_tokens


def test_19b_recursive_sentence_then_words(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = " ".join(f"w{i}" for i in range(15)) + "."
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="recursive", max_tokens=5, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.token_count <= cfg.max_tokens


def test_19c_recursive_word_then_token_windows(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    # The offline tokenizer has a single [UNK] for any unknown word, so even a
    # "monoword" still counts as 1 token. Simulate a long word by joining tokens.
    # In practice the token-window fallback fires when a single whitespace-bounded
    # piece tokenizes to > max_tokens — exercise the fallback path by giving the
    # recursive cascade nothing to split below words.
    word = "x" * 200  # one token under WhitespaceSplit
    doc = make_doc(page_count=1)
    pages = [make_page(text=word)]
    cfg = make_config(strategy="recursive", max_tokens=1, min_chunk_tokens=0)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    # one token, max=1 → exactly one chunk; no crash from the token-window fallback
    assert len(chunks) == 1


def test_20_recursive_greedy_preserves_order(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    text = "AAA. BBB. CCC. DDD. EEE."
    doc = make_doc(page_count=1)
    pages = [make_page(text=text)]
    cfg = make_config(strategy="recursive", max_tokens=2, min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs, regex_sentences)
    starts = [c.text.split()[0] for c in chunks]
    # must appear in original order
    expected_order = ["AAA.", "BBB.", "CCC.", "DDD.", "EEE."]
    seen_order = [w for w in expected_order if w in starts]
    assert seen_order == [w for w in starts if w in expected_order]


# ─── 5.8 section index and section strategy ──────────────────────────────────


def test_21a_section_index_basic() -> None:
    toc = [
        ch.TocEntry(text="Ch 1", level=0, page_index=0),
        ch.TocEntry(text="1.1", level=1, page_index=1),
        ch.TocEntry(text="Ch 2", level=0, page_index=3),
    ]
    idx = ch.build_section_index(toc, page_count=5)
    assert idx[0] == ["Ch 1"]
    assert idx[1] == ["Ch 1", "1.1"]
    assert idx[2] == ["Ch 1", "1.1"]
    assert idx[3] == ["Ch 2"]
    assert idx[4] == ["Ch 2"]


def test_21b_section_index_pages_before_first_entry() -> None:
    toc = [ch.TocEntry(text="Ch 1", level=0, page_index=2)]
    idx = ch.build_section_index(toc, page_count=4)
    assert idx[0] == [] and idx[1] == []
    assert idx[2] == ["Ch 1"] and idx[3] == ["Ch 1"]


def test_21c_section_index_null_page_ignored() -> None:
    toc = [
        ch.TocEntry(text="Intro", level=0, page_index=None),
        ch.TocEntry(text="Ch 1", level=0, page_index=0),
    ]
    idx = ch.build_section_index(toc, page_count=2)
    # Intro is ignored for mapping; Ch 1 takes over from page 0
    assert idx[0] == ["Ch 1"] and idx[1] == ["Ch 1"]


def test_21d_section_index_level_jumps() -> None:
    toc = [
        ch.TocEntry(text="Top", level=0, page_index=0),
        ch.TocEntry(text="Deep", level=2, page_index=1),  # jumps from 0 to 2
    ]
    idx = ch.build_section_index(toc, page_count=2)
    assert idx[0] == ["Top"]
    # stack truncates to entry's level; the entry is placed at its level
    assert idx[1][-1] == "Deep"
    # does not crash and does not duplicate ancestors
    assert len(idx[1]) <= 3


def test_21e_section_index_empty_toc() -> None:
    idx = ch.build_section_index([], page_count=3)
    assert idx == [[], [], []]


def test_22_section_cross_page_subchunks(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    # one section, two pages, joint text exceeds budget → sub-chunked
    toc = [{"text": "Ch 1", "level": 0, "page_index": 0}]
    doc = make_doc(page_count=2, toc=toc)
    p0 = make_page(page_index=0, text="AAA BBB CCC DDD EEE.")  # 5 tokens
    p1 = make_page(page_index=1, text="FFF GGG HHH III JJJ.")  # 5 tokens
    cfg = make_config(strategy="section", max_tokens=4, min_chunk_tokens=1, cross_page=True)
    chunks, _ = chunk(doc, [p0, p1], cfg, funcs, regex_sentences)
    assert len(chunks) >= 2
    # every chunk stamped with the section path
    for c in chunks:
        assert c.section_path == ["Ch 1"]


def test_23_section_no_cross_page_respects_pages(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    toc = [{"text": "Ch 1", "level": 0, "page_index": 0}]
    doc = make_doc(page_count=2, toc=toc)
    p0 = make_page(page_index=0, text="alpha beta gamma")
    p1 = make_page(page_index=1, text="delta epsilon zeta")
    cfg = make_config(strategy="section", max_tokens=50, min_chunk_tokens=1, cross_page=False)
    chunks, _ = chunk(doc, [p0, p1], cfg, funcs, regex_sentences)
    # each page is its own unit but tagged with the same section
    for c in chunks:
        assert c.section_path == ["Ch 1"]
        # never crosses pages
        assert len({s.page_index for s in c.spans}) == 1


def test_24_other_strategies_attach_section_path(
    funcs: tuple[ch.TokenCounter, ch.TokenOffsets],
) -> None:
    toc = [
        {"text": "Ch 1", "level": 0, "page_index": 0},
        {"text": "Ch 2", "level": 0, "page_index": 1},
    ]
    doc = make_doc(page_count=2, toc=toc)
    pages = [make_page(page_index=i, text=f"text on page {i}") for i in range(2)]

    cfg = make_config(strategy="page", min_chunk_tokens=1)
    chunks, _ = chunk(doc, pages, cfg, funcs)
    assert chunks[0].section_path == ["Ch 1"]
    assert chunks[1].section_path == ["Ch 2"]

    cfg2 = make_config(strategy="fixed-size", max_tokens=50, min_chunk_tokens=1)
    chunks2, _ = chunk(doc, pages, cfg2, funcs)
    assert chunks2[0].section_path == ["Ch 1"]
    assert chunks2[1].section_path == ["Ch 2"]


# ─── 5.9 pipeline / OCR / IO / manifest ──────────────────────────────────────


def _records_to_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _run(
    in_text: str, cfg: ch.ChunkingConfig, created_at: str = "2026-06-12T00:00:00Z"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = cfg.model_copy(update={"tokenizer_model": _tokenizer_path()})
    out = io.StringIO()
    ch.process_stream(io.StringIO(in_text), out, cfg, created_at=created_at)
    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    manifest = lines[-1]
    chunks = lines[:-1]
    return chunks, manifest


def test_25_ocr_policies() -> None:
    doc = make_doc(page_count=2).model_dump()
    p0 = make_page(page_index=0, text="real content").model_dump()
    p1_ocr = make_page(page_index=1, text="x", needs_ocr=True).model_dump()
    payload = _records_to_jsonl([doc, p0, p1_ocr])

    # skip
    chunks, manifest = _run(
        payload, make_config(strategy="page", on_ocr_needed="skip", min_chunk_tokens=1)
    )
    assert len(chunks) == 1
    assert any(
        s["reason"] == "needs_ocr" and s["page_index"] == 1 for s in manifest["skipped_pages"]
    )

    # include
    chunks, _ = _run(
        payload, make_config(strategy="page", on_ocr_needed="include", min_chunk_tokens=1)
    )
    assert len(chunks) == 2

    # error
    with pytest.raises(ch.ChunkingError) as exc:
        _run(payload, make_config(strategy="page", on_ocr_needed="error", min_chunk_tokens=1))
    msg = str(exc.value)
    assert "d1" in msg and "1" in msg


def test_26_multi_document_and_orphan_pages() -> None:
    doc1 = make_doc("d1", page_count=1).model_dump()
    p1 = make_page("d1", 0, "page from doc1").model_dump()
    doc2 = make_doc("d2", page_count=1).model_dump()
    p2 = make_page("d2", 0, "page from doc2").model_dump()
    payload = _records_to_jsonl([doc1, p1, doc2, p2])
    chunks, _manifest = _run(payload, make_config(strategy="page", min_chunk_tokens=1))
    assert len(chunks) == 2
    assert {c["doc_id"] for c in chunks} == {"d1", "d2"}

    # orphan page: arrives before any document — counted invalid, skipped
    orphan = make_page("d99", 0, "orphan").model_dump()
    payload2 = _records_to_jsonl([orphan, doc1, p1])
    chunks2, manifest2 = _run(payload2, make_config(strategy="page", min_chunk_tokens=1))
    assert len(chunks2) == 1
    assert manifest2["stats"]["invalid_lines"] >= 1


def test_27_malformed_lines() -> None:
    good_doc = make_doc().model_dump()
    good_page = make_page(text="hello").model_dump()
    bad_json = "{not json"
    bad_schema = json.dumps({"record_type": "page", "missing": "fields"})
    payload = (
        json.dumps(good_doc)
        + "\n"
        + bad_json
        + "\n"
        + bad_schema
        + "\n"
        + json.dumps(good_page)
        + "\n"
    )
    # non-strict: continue
    chunks, manifest = _run(payload, make_config(strategy="page", min_chunk_tokens=1))
    assert len(chunks) == 1
    assert manifest["stats"]["invalid_lines"] == 2

    # strict: raise with line number
    with pytest.raises(ch.ChunkingError) as exc:
        _run(payload, make_config(strategy="page", strict=True, min_chunk_tokens=1))
    assert "line 2" in str(exc.value)


def test_28_manifest_layout() -> None:
    doc = make_doc(page_count=2).model_dump()
    p0 = make_page(page_index=0, text="content one").model_dump()
    p1 = make_page(page_index=1, text="content two", needs_ocr=True).model_dump()
    payload = _records_to_jsonl([doc, p0, p1])
    chunks, manifest = _run(payload, make_config(strategy="page", min_chunk_tokens=1))
    assert manifest["record_type"] == "manifest"
    assert manifest["stats"]["chunks_written"] == len(chunks)
    assert manifest["stats"]["pages_seen"] == 2
    assert manifest["stats"]["pages_chunked"] == 1
    assert manifest["config"]["strategy"] == "page"
    assert manifest["config"]["on_ocr_needed"] == "skip"
    assert manifest["skipped_pages"] == [{"doc_id": "d1", "page_index": 1, "reason": "needs_ocr"}]


def test_29_contract_against_output_schema() -> None:
    pytest.importorskip("jsonschema")
    import jsonschema

    schema_path = Path(__file__).resolve().parent.parent / "output-schema.json"
    schema = json.loads(schema_path.read_text())

    doc = make_doc(page_count=3, toc=[{"text": "Ch 1", "level": 0, "page_index": 0}]).model_dump()
    p0 = make_page(page_index=0, text="alpha beta gamma delta epsilon").model_dump()
    p1 = make_page(page_index=1, text="more content here on page two").model_dump()
    p2 = make_page(page_index=2, text="x", needs_ocr=True).model_dump()
    payload = _records_to_jsonl([doc, p0, p1, p2])
    cfg = make_config(
        strategy="recursive",
        max_tokens=4,
        overlap=1,
        min_chunk_tokens=2,
    )
    chunks, manifest = _run(payload, cfg)
    validator = jsonschema.Draft202012Validator(schema)
    for c in chunks:
        errs = list(validator.iter_errors(c))
        assert not errs, errs
    errs = list(validator.iter_errors(manifest))
    assert not errs, errs


def test_30_determinism() -> None:
    doc = make_doc(page_count=2).model_dump()
    p0 = make_page(page_index=0, text="alpha beta gamma").model_dump()
    p1 = make_page(page_index=1, text="delta epsilon zeta").model_dump()
    payload = _records_to_jsonl([doc, p0, p1])
    cfg = make_config(strategy="fixed-size", max_tokens=2, overlap=0, min_chunk_tokens=1)
    cfg = cfg.model_copy(update={"tokenizer_model": _tokenizer_path()})

    out1 = io.StringIO()
    ch.process_stream(io.StringIO(payload), out1, cfg, created_at="2026-06-12T00:00:00Z")
    out2 = io.StringIO()
    ch.process_stream(io.StringIO(payload), out2, cfg, created_at="2026-06-12T00:00:00Z")
    assert out1.getvalue() == out2.getvalue()


# ─── 5.10 CLI and tool ───────────────────────────────────────────────────────


def test_31_cli_parses_defaults() -> None:
    ns = cli.parse_args([])
    defaults = ch.ChunkingConfig()
    assert ns.strategy == defaults.strategy
    assert ns.max_tokens == defaults.max_tokens
    assert ns.overlap == defaults.overlap
    assert ns.min_chunk_tokens == defaults.min_chunk_tokens
    assert ns.tokenizer_model == defaults.tokenizer_model

    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--strategy", "bogus"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit):
        cli.parse_args(["--max-tokens", "0"])


def test_32_cli_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = make_doc(page_count=1).model_dump()
    page = make_page(text="alpha beta gamma delta").model_dump()
    in_path = tmp_path / "in.jsonl"
    in_path.write_text(_records_to_jsonl([doc, page]))
    out_path = tmp_path / "out.jsonl"

    rc = cli.main(
        [
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--strategy",
            "page",
            "--min-chunk-tokens",
            "1",
            "--tokenizer-model",
            _tokenizer_path(),
        ]
    )
    assert rc == 0
    contents = out_path.read_text().strip().splitlines()
    assert json.loads(contents[-1])["record_type"] == "manifest"


def test_33_cli_stdin_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = make_doc(page_count=1).model_dump()
    page = make_page(text="hello world").model_dump()
    payload = _records_to_jsonl([doc, page])

    stdin = io.StringIO(payload)
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    rc = cli.main(
        ["--strategy", "page", "--min-chunk-tokens", "1", "--tokenizer-model", _tokenizer_path()]
    )
    assert rc == 0
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert json.loads(lines[-1])["record_type"] == "manifest"


def test_34_tool_spec_shape() -> None:
    spec = tl.TOOL_SPEC
    # JSON-serializable
    json.dumps(spec)
    assert spec["name"]
    assert spec["description"]
    schema = spec["input_schema"]
    props = schema["properties"]
    # must include all config fields plus input/output paths
    cfg_fields = set(ch.ChunkingConfig.model_fields.keys()) - {"strict"}
    expected_keys = cfg_fields | {"input_path", "output_path"}
    assert expected_keys.issubset(set(props.keys()))


def test_35_tool_run_happy_and_error(tmp_path: Path) -> None:
    doc = make_doc(page_count=1).model_dump()
    page = make_page(text="hello there").model_dump()
    in_path = tmp_path / "in.jsonl"
    in_path.write_text(_records_to_jsonl([doc, page]))
    out_path = tmp_path / "out.jsonl"

    result = tl.run_chunker_tool(
        {
            "input_path": str(in_path),
            "output_path": str(out_path),
            "strategy": "page",
            "min_chunk_tokens": 1,
            "tokenizer_model": _tokenizer_path(),
        }
    )
    assert result["ok"] is True
    assert result["output_path"] == str(out_path)
    assert result["chunks_written"] >= 1

    bad = tl.run_chunker_tool({"input_path": "/nope/missing.jsonl"})
    assert bad["ok"] is False
    assert "error" in bad


# ─── 5.11 NLTK integration ───────────────────────────────────────────────────


@nltk_ready
def test_36_nltk_handles_abbreviation() -> None:
    spans = ch.split_sentences_nltk("Dr. Smith went home. He slept.")
    assert len(spans) == 2


@nltk_ready
def test_37_nltk_spans_roundtrip() -> None:
    text = "First sentence here. Then another one. And one more."
    spans = ch.split_sentences_nltk(text)
    assert len(spans) == 3
    for s, e in spans:
        assert text[s:e].strip() != ""
