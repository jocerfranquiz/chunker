"""Agent-callable interface. One function, one machine-readable spec, structured
results. Never raises; agents prefer structured errors to tracebacks.

The TOOL_SPEC is generated from ChunkingConfig so the agent surface cannot drift
from the library's actual options. `strict` is intentionally omitted from the
agent surface (agents should always get the lenient code path).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

import chunker as ch

# Fields exposed to agents. `strict` is hidden by design (see module docstring).
_AGENT_FIELDS: tuple[str, ...] = (
    "strategy",
    "max_tokens",
    "overlap",
    "min_chunk_tokens",
    "tokenizer_model",
    "cross_page",
    "on_ocr_needed",
)


def _build_input_schema() -> dict[str, Any]:
    """Build the input_schema from ChunkingConfig so it stays in lockstep."""
    cfg_schema = ch.ChunkingConfig.model_json_schema()
    cfg_props = cfg_schema.get("properties", {})
    # Pydantic encodes Literals as either `enum` directly or via $defs; strip
    # $ref indirection so the spec is self-contained.
    defs = cfg_schema.get("$defs", {})

    def resolve(prop: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in prop:
            name = prop["$ref"].split("/")[-1]
            return dict(defs.get(name, {}))
        return prop

    properties: dict[str, Any] = {
        "input_path": {
            "type": "string",
            "description": "Path to extraction JSONL.",
        },
        "output_path": {
            "type": "string",
            "description": "Path to write chunk JSONL.",
        },
    }
    for name in _AGENT_FIELDS:
        if name in cfg_props:
            properties[name] = resolve(cfg_props[name])

    return {
        "type": "object",
        "properties": properties,
        "required": ["input_path", "output_path"],
    }


TOOL_SPEC: dict[str, Any] = {
    "name": "chunk_pdf_extraction",
    "description": (
        "Chunk JSONL records produced by the PDF extraction stage into "
        "embedding-ready chunk records (see output-schema.json). "
        "Strategies: page | fixed-size | recursive | section | sentence."
    ),
    "input_schema": _build_input_schema(),
}


def run_chunker_tool(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Run the chunker from agent-supplied arguments. Always returns a dict:

    success: {"ok": True, "output_path": str, "chunks_written": int,
              "pages_skipped": [{...}], "stats": {...}}
    failure: {"ok": False, "error": "<message>"}
    """
    try:
        input_path = str(arguments["input_path"])
        output_path = str(arguments["output_path"])
    except KeyError as e:
        return {"ok": False, "error": f"missing required argument: {e.args[0]}"}

    cfg_kwargs: dict[str, Any] = {}
    for name in _AGENT_FIELDS:
        if name in arguments:
            cfg_kwargs[name] = arguments[name]
    try:
        cfg = ch.ChunkingConfig(**cfg_kwargs)
    except ValidationError as e:
        return {"ok": False, "error": f"invalid configuration: {e}"}

    try:
        with (
            open(input_path, encoding="utf-8") as in_f,
            open(output_path, "w", encoding="utf-8") as out_f,
        ):
            stats, skipped = ch.process_stream(in_f, out_f, cfg)
    except FileNotFoundError as e:
        return {"ok": False, "error": f"file not found: {e.filename}"}
    except ch.ChunkingError as e:
        return {"ok": False, "error": str(e)}
    except OSError as e:
        return {"ok": False, "error": f"IO error: {e}"}

    return {
        "ok": True,
        "output_path": output_path,
        "chunks_written": stats.chunks_written,
        "pages_skipped": [s.model_dump() for s in skipped],
        "stats": stats.model_dump(),
    }


__all__ = ["TOOL_SPEC", "run_chunker_tool"]
