# chunker

A Python tool that consumes JSONL records produced by PDF extractor and emits 
JSONL chunks ready for an embedding tool.

Five chunking strategies, a real HuggingFace tokenizer for accurate token
budgeting, NLTK for sentence boundaries, full provenance back to source-page
character offsets, and a procedural API designed for both batch jobs and agent
tool-calling.

## Install

Python ≥3.10. Install with development extras for tests and lint:

```bash
pip install -e ".[dev]"
```

Three runtime dependencies: `tokenizers`, `pydantic`, `nltk`. On first use of a
sentence-aware strategy, NLTK's `punkt_tab` data is downloaded automatically.
For air-gapped runs, provision it ahead of time:

```bash
python -m nltk.downloader punkt_tab
```

## Quick start (CLI)

```bash
chunker \
    --input extraction.jsonl \
    --output chunks.jsonl \
    --strategy recursive \
    --max-tokens 500 \
    --overlap 50 \
    --tokenizer-model BAAI/bge-small-en-v1.5
```

`--input`/`--output` default to stdin/stdout, so it pipes:

```bash
cat extraction.jsonl | chunker --strategy section > chunks.jsonl
```

A human-readable summary (chunk count, skipped pages with reasons) is written
to **stderr**; stdout stays pure JSONL.

## Strategies

| Strategy     | What it does                                                                                | When to use                                                  |
|--------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `page`       | One chunk per page. Never splits, even if a page exceeds `max_tokens`.                      | Citation-exact retrieval; corpora with one logical unit per page. |
| `fixed-size` | Sliding token-index windows of `max_tokens` with `overlap`.                                  | Uniform shapes; predictable cost; baseline.                  |
| `sentence`   | Greedy-packs whole sentences (NLTK) up to budget; over-budget sentences fall back to windows. | Prose where sentence-aligned retrieval matters.              |
| `recursive`  | Cascade: paragraphs → sentences → words → token windows. Default.                            | Mixed documents; preserves structure where possible.         |
| `section`    | Groups pages by TOC section, then sub-chunks each section recursively.                       | Books, reports, anything with a usable outline.              |

All strategies attach a `section_path` (TOC ancestry) to every chunk when the
document has an outline.

## Configuration

| Flag                          | Default                    | Notes                                                                       |
|-------------------------------|----------------------------|------------------------------------------------------------------------------|
| `--strategy`                  | `recursive`                | One of `page`, `fixed-size`, `recursive`, `section`, `sentence`.            |
| `--max-tokens`                | `512`                      | Content tokens; **leave headroom** for CLS/SEP at embedding time.           |
| `--overlap`                   | `64`                       | Must be < `max-tokens`. Ignored by `page`.                                  |
| `--min-chunk-tokens`          | `20`                       | A trailing piece below this is merged into the previous chunk where allowed.|
| `--tokenizer-model`           | `BAAI/bge-small-en-v1.5`   | HF Hub repo id, or path to a local `tokenizer.json` (no network needed).    |
| `--cross-page/--no-cross-page`| `--no-cross-page`          | Whether chunks may span page boundaries. Off preserves citation locality.   |
| `--on-ocr-needed`             | `skip`                     | `skip` \| `include` \| `error` for pages flagged `needs_ocr: true`.          |
| `--strict`                    | off                        | Abort on malformed input lines instead of counting and continuing.          |

Token counting uses `add_special_tokens=False`. If your embedding model expects
512 tokens *including* `[CLS]`/`[SEP]`, set `--max-tokens 500` (or whatever the
model's pair add to).

**Match the tokenizer to your embedding model.** Counts differ across BERT-style,
SentencePiece, and BPE tokenizers; the same text can be 10–30% longer or shorter
under a different tokenizer.

## Output

JSONL with two record kinds (dispatch on `record_type`). The full schema is in
`output-schema.json`.

* `chunk` — one per chunk; the embedding input is the `text` field. Carries
  stable `chunk_id` (`{doc_id}::c{NNNN}`), token count, `section_path`,
  per-source-page `spans` (`page_index`, `char_start`, `char_end`), and
  citation-ready `source_doc` (title + path).
* `manifest` — exactly one, **last line** of the file. Run provenance:
  effective config, run stats (documents/pages/chunks counts, invalid lines),
  and a list of pages that produced no chunks (`needs_ocr`, `empty_text`,
  `invalid_record`) so a side process can fill them in and you can rerun.

Trimmed example chunk:

```json
{
  "record_type": "chunk",
  "schema_version": "1.0.0",
  "chunk_id": "paper-acme-2024::c0007",
  "doc_id": "paper-acme-2024",
  "chunk_index": 7,
  "text": "Our experiments show...",
  "strategy": "recursive",
  "token_count": 412,
  "tokenizer_model": "BAAI/bge-small-en-v1.5",
  "spans": [
    {"page_index": 3, "page_label": "4", "char_start": 0, "char_end": 1840}
  ],
  "section_path": ["3. Experiments", "3.2 Setup"],
  "source_doc": {"title": "Acme Methods", "source_path": "/papers/acme.pdf"}
}
```

`page.text[char_start:char_end]` is the exact substring this chunk came from.

## Library use

```python
import chunker

with open("extraction.jsonl") as in_f, open("chunks.jsonl", "w") as out_f:
    cfg = chunker.ChunkingConfig(
        strategy="recursive",
        max_tokens=500,
        overlap=50,
        tokenizer_model="BAAI/bge-small-en-v1.5",
    )
    stats, skipped = chunker.process_stream(in_f, out_f, cfg)

print(f"{stats.chunks_written} chunks; {len(skipped)} pages skipped")
```

The library streams a document at a time, so memory usage doesn't grow with the
size of the input file.

For finer control, `chunker.chunk_document(doc, pages, cfg, count, offsets,
sentences)` operates on already-parsed records and lets you inject tokenizer and
sentence-splitter callables. See `chunker.py` and the test suite for examples.

## Agent / tool use

`tool.py` exposes a machine-readable spec and a never-raising entry point:

```python
from tool import TOOL_SPEC, run_chunker_tool

# TOOL_SPEC is generated from ChunkingConfig — it can't drift from the library.
# Pass it to your LLM as a tool definition.

result = run_chunker_tool({
    "input_path": "extraction.jsonl",
    "output_path": "chunks.jsonl",
    "strategy": "section",
    "max_tokens": 500,
    "overlap": 50,
})
# success: {"ok": True, "output_path": ..., "chunks_written": N, "stats": {...}}
# failure: {"ok": False, "error": "<message>"}
```

`run_chunker_tool` returns structured errors rather than raising — agents
handle structured failures better than tracebacks. `strict` is intentionally
omitted from the agent surface (agents always get the lenient code path).

## Development

```bash
ruff format .
ruff check .
mypy chunker.py cli.py tool.py tests/
pytest
```

All four are part of the definition-of-done. `mypy` runs strict with the
`pydantic.mypy` plugin. The test suite is one file (`tests/tests.py`); offline
fixtures keep the run deterministic and network-free.

## Determinism

Given the same input file, the same `ChunkingConfig`, and the same
`created_at` (process_stream accepts it as a keyword), output is byte-identical
across runs. The CLI/tool fill in `created_at` from the wall clock; library
callers can pin it explicitly for caching or content-addressed pipelines.

## Non-goals

* **Embedding** — out of scope; emit chunks, hand to your embedder.
* **Image and caption chunks** — owned by the image tool in a separate stage.
* **OCR execution** — `needs_ocr` pages are reported in the manifest, not
  re-processed here.
* **Embedding-based / semantic chunking** — deliberately deferred; the
  pipeline doesn't carry an embedder.
* **Parallelism** — the per-document streaming design leaves room for
  `multiprocessing` over document groups later without API changes.

## See also

* `output-schema.json` — formal output contract.
* `guide.md` — implementation guide, design rationale, and rebuild plan.
