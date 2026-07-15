# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed OpenAI-compatible HTTP endpoint for MLX inference — the engine-side serve verb.

Same Safety-first (Rule #1) discipline as ``generate``: the safety gate runs *before* any
weight load. A vetoed model prints one JSON refusal to stdout and exits 1 immediately. Only
a safe (model, context) pair reaches mlx_lm. After binding, a single ready-handshake JSON
line is flushed to stdout (the IPC signal for ARA). All logging goes to **stderr**; stdout
carries only the handshake or refusal JSON.

Per-request governance enforces the context ceiling for every call: no single request can
exceed it, regardless of what the caller requests. Prompt tokens are counted with the loaded
tokenizer (no weights re-accessed). The ceiling is a HARD cap — requests that need more are
refused with HTTP 400 before generation starts.

    refused: {"refused": true, "reason": "<why>"}
    ready:   {"ready": true, "url": "http://127.0.0.1:<PORT>", "context": <ceiling>}

Usage:
    python -m ara_engine_mlx.serve <hf_id> <ctx_ceiling> --margin G --overhead G --port N [--kv-bits N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import measure_one, models, system

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure governance helpers — unit-testable without weights or sockets
# --------------------------------------------------------------------------- #

def governed_max_tokens(prompt_tokens: int, requested_max_tokens: int,
                        ceiling: int) -> int | None:
    """Return the allowed max_tokens, or None if the request must be rejected.

    None is returned when the prompt alone fills the ceiling (``prompt_tokens >= ceiling``)
    or when ``prompt_tokens + requested_max_tokens > ceiling``. Otherwise returns
    ``min(requested_max_tokens, ceiling - prompt_tokens)``, which equals
    ``requested_max_tokens`` in the acceptance branch — the clamp is a conservative
    safety belt for callers that compute their own budget.
    """
    if prompt_tokens >= ceiling:
        return None
    if prompt_tokens + requested_max_tokens > ceiling:
        return None
    return min(requested_max_tokens, ceiling - prompt_tokens)


# Well-known instruct turn-end tokens. mlx_lm registers only the scalar tokenizer.eos_token_id
# (e.g. <eos> id 1 for gemma-3), which is NOT the token instruct models emit to end a turn
# (gemma uses <end_of_turn> id 106). Without it, generation never self-stops — it runs to
# max_tokens and rambles past the answer, tanking benchmark scoring (#107). The model-author
# source (generation_config.json eos_token_id) is often stripped from quantized community
# repos, so we register any well-known terminator that actually exists in this tokenizer's
# vocab — general (gemma/chatml/llama3/phi), not a per-model hardcode.
_TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>")


def register_turn_end_tokens(tokenizer) -> list[int]:
    """Add known instruct turn-end tokens (those present in the vocab) to *tokenizer*'s eos set.

    Only tokens that resolve to a real (non-unk) id are added, so it is a safe no-op for models
    that don't use a given terminator. Idempotent. Returns the ids added. A no-op on tokenizers
    without mlx_lm's ``add_eos_token`` (e.g. a bare transformers tokenizer)."""
    add = getattr(tokenizer, "add_eos_token", None)
    if not callable(add):
        return []
    unk = getattr(tokenizer, "unk_token_id", None)
    added: list[int] = []
    for tok_str in _TURN_END_TOKENS:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
        except Exception:
            continue
        if tid is None or tid == unk:
            continue
        try:
            add(tok_str)
            added.append(tid)
        except Exception:
            continue
    return added


# --------------------------------------------------------------------------- #
# Pre-load gate — mirrors generate.py (Rule #1: never load if gate fails)
# --------------------------------------------------------------------------- #

def _pre_load_gate(hf_id: str, ceiling: int, *, margin_gb: float, overhead_gb: float,
                   kv_bits: int | None = None,
                   measured_slope_gb_per_k: float | None = None) -> tuple[dict | None, int | None]:
    """Validate model + run memory gate before any weight load.

    Returns ``(refusal_dict, None)`` on any failure; ``(None, effective_kv_bits)`` when safe.
    ``effective_kv_bits`` reflects :func:`measure_one._effective_kv_bits` — fp16 (None) for
    non-quantizable (RotatingKVCache) models regardless of what the caller passed.

    ``measured_slope_gb_per_k`` (when ARA serves this model's own measured ceiling) makes the
    gate predict with the measured ramp slope instead of the conservative a-priori one — see
    :func:`measure_one.safety_gate`. Absent → a-priori (unchanged).
    """
    info = models.describe(hf_id)
    if info is None:
        return ({"refused": True, "reason": f"model not found in HF cache: {hf_id}"}, None)
    if not info.is_causal:
        return ({"refused": True,
                 "reason": f"{hf_id} is not a supported causal language model"}, None)

    effective_kv = measure_one._effective_kv_bits(info, kv_bits)
    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    reason = measure_one.safety_gate(info, limits, ceiling, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base,
                                     kv_bits=effective_kv,
                                     measured_slope_gb_per_k=measured_slope_gb_per_k)
    if reason is not None:
        return ({"refused": True, "reason": reason}, None)
    return (None, effective_kv)


# --------------------------------------------------------------------------- #
# Per-request decision — pure function, no HTTP plumbing
# --------------------------------------------------------------------------- #

def _render_messages(tokenizer, messages: list[dict], *, tools=None) -> list[int]:
    """Render chat messages to TOKEN IDS via the model's chat template (single ``<bos>``).

    Returns token ids, NOT a string: passing an already-templated string to mlx generation
    re-encodes it and prepends a SECOND ``<bos>`` (gemma-3: ``[2,2,...]`` vs ``[2,...]``),
    degrading output (#107). ``tools`` is passed to the template when provided (function-calling
    models). Falls back to encoding a plain ``role: content`` join for tokenizers without a
    chat template.
    """
    try:
        kwargs = {}
        if tools is not None:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **kwargs
        )  # tokenize=True (default) -> token ids, single <bos>
    except Exception:
        return tokenizer.encode(
            "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        )


def _handle_completions(body: dict, tokenizer, model, ceiling: int,
                        kv_bits: int | None, hf_id: str) -> tuple[int, dict]:
    """Pure request-decision function — no HTTP plumbing, fully unit-testable.

    Tokenizes the rendered prompt (tokenizer only, no weights), applies governance, then
    either returns ``(400, error_body)`` or generates and returns ``(200, completion_body)``.
    The error body follows the OpenAI error envelope; the success body is a minimal
    OpenAI-compatible chat completion object.

    When the body carries ``tools`` and the tokenizer supports tool-calling, a tool-use
    response (``finish_reason: "tool_calls"``) is returned if the model emits tool markers.
    """
    messages = body.get("messages", [])
    _raw_max = body.get("max_tokens", 512)
    if not isinstance(_raw_max, int) or isinstance(_raw_max, bool):
        return (400, {
            "error": {
                "message": "max_tokens must be a positive integer",
                "type": "invalid_request_error",
            }
        })
    requested_max = _raw_max
    if requested_max <= 0:
        return (400, {
            "error": {
                "message": "max_tokens must be positive",
                "type": "invalid_request_error",
            }
        })
    tools = body.get("tools") or None

    prompt = _render_messages(tokenizer, messages, tools=tools)   # token ids (single <bos>, #107)
    prompt_tokens = len(prompt)

    allowed = governed_max_tokens(prompt_tokens, requested_max, ceiling)
    if allowed is None:
        need = prompt_tokens + requested_max
        return (400, {
            "error": {
                "message": (f"request exceeds governed context ceiling {ceiling} "
                            f"(needed {need})"),
                "type": "context_length_exceeded",
            }
        })

    # Lazy import so the module loads without mlx installed and tests can monkeypatch.
    from mlx_lm import generate as mlx_generate  # type: ignore[import]

    kv_kwargs = ({} if kv_bits is None
                 else {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 5000})
    text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=allowed, **kv_kwargs)

    # Tool-calling: scan ALL <tool_call>…</tool_call> blocks (parallel calls).
    # If the start marker is present but no end marker is found, fall through to text.
    if (tools and getattr(tokenizer, "has_tool_calling", False)
            and tokenizer.tool_call_start in text):
        try:
            tool_calls = []
            search_start = 0
            while True:
                start_idx = text.find(tokenizer.tool_call_start, search_start)
                if start_idx == -1:
                    break
                content_start = start_idx + len(tokenizer.tool_call_start)
                end_idx = text.find(tokenizer.tool_call_end, content_start)
                if end_idx == -1:
                    break  # unterminated marker — stop scanning, fall through to text
                tool_text = text[content_start:end_idx]
                parsed = tokenizer.tool_parser(tool_text, tools)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                for tc in parsed:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    })
                search_start = end_idx + len(tokenizer.tool_call_end)
            if tool_calls:
                return (200, {
                    "object": "chat.completion",
                    "model": hf_id,
                    "choices": [
                        {"index": 0,
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": tool_calls},
                         "finish_reason": "tool_calls"},
                    ],
                })
        except Exception:
            pass  # Parse failed — fall through to plain text response.

    return (200, {
        "object": "chat.completion",
        "model": hf_id,
        "choices": [
            {"index": 0,
             "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"},
        ],
    })


def _stream_completions(body: dict, tokenizer, model, ceiling: int,
                        kv_bits: int | None, hf_id: str, handler) -> tuple[int, dict] | None:
    """SSE streaming completions. Governance runs FIRST.

    Returns ``(400, error_body)`` when governance rejects the request (caller must send
    the error before any SSE header). Returns ``None`` after streaming is complete — the
    caller must not call ``_json`` in that case.
    """
    messages = body.get("messages", [])
    _raw_max = body.get("max_tokens", 512)
    if not isinstance(_raw_max, int) or isinstance(_raw_max, bool):
        return (400, {
            "error": {
                "message": "max_tokens must be a positive integer",
                "type": "invalid_request_error",
            }
        })
    requested_max = _raw_max
    if requested_max <= 0:
        return (400, {
            "error": {
                "message": "max_tokens must be positive",
                "type": "invalid_request_error",
            }
        })
    tools = body.get("tools") or None

    prompt = _render_messages(tokenizer, messages, tools=tools)   # token ids (single <bos>, #107)
    prompt_tokens = len(prompt)

    allowed = governed_max_tokens(prompt_tokens, requested_max, ceiling)
    if allowed is None:
        need = prompt_tokens + requested_max
        return (400, {
            "error": {
                "message": (f"request exceeds governed context ceiling {ceiling} "
                            f"(needed {need})"),
                "type": "context_length_exceeded",
            }
        })

    # Governance passed — commit to SSE; no Content-Length (chunked streaming).
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()

    # Lazy import so module loads without mlx and tests can monkeypatch.
    from mlx_lm import stream_generate  # type: ignore[import]

    kv_kwargs = ({} if kv_bits is None
                 else {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 5000})

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    finish_reason = "stop"

    # Initial role chunk (OpenAI shape): role must arrive before any content deltas.
    role_chunk = json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": hf_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })
    handler.wfile.write(f"data: {role_chunk}\n\n".encode())
    handler.wfile.flush()

    try:
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=allowed, **kv_kwargs):
            # Emit content delta whenever text is non-empty — including the final yield from
            # detokenizer.finalize() which carries text AND a non-None finish_reason.
            if resp.text:
                chunk = json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": hf_id,
                    "choices": [
                        {"index": 0, "delta": {"content": resp.text}, "finish_reason": None},
                    ],
                })
                handler.wfile.write(f"data: {chunk}\n\n".encode())
                handler.wfile.flush()
            if resp.finish_reason is not None:
                finish_reason = resp.finish_reason
    except Exception:
        # Best-effort [DONE] before propagating so clients can cleanly end the stream.
        try:
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except Exception:
            pass
        raise

    # Final chunk carries finish_reason; then the sentinel.
    final = json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": hf_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })
    handler.wfile.write(f"data: {final}\n\n".encode())
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()
    return None


# --------------------------------------------------------------------------- #
# HTTP handler factory
# --------------------------------------------------------------------------- #

def _make_handler(model, tokenizer, ceiling: int, kv_bits: int | None, hf_id: str):
    """Return a BaseHTTPRequestHandler subclass closed over the loaded model + config."""

    class _Handler(BaseHTTPRequestHandler):
        # Bound the per-request socket read. This server is single-threaded on purpose, so a client
        # that sends a valid Content-Length then stalls the body would otherwise hang rfile.read()
        # forever and wedge the endpoint for every other caller (a slowloris — the same failure
        # class as the negative-Content-Length hang, via a valid-looking header). With a timeout the
        # stalled read raises socket.timeout, caught by do_POST's body try/except → a clean HTTP 400.
        timeout = 30.0

        def log_message(self, fmt, *args):  # noqa: N802
            log.debug(fmt, *args)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": {"message": f"not found: {self.path}",
                                           "type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                self._json(400, {"error": {"message": "invalid Content-Length",
                                           "type": "invalid_request_error"}})
                return
            if length < 0:
                self._json(400, {"error": {"message": "invalid Content-Length",
                                           "type": "invalid_request_error"}})
                return
            if length > 1_048_576:
                self._json(413, {"error": {"message": "request body too large",
                                           "type": "invalid_request_error"}})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except Exception as exc:
                self._json(400, {"error": {"message": str(exc),
                                           "type": "invalid_request_error"}})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": {"message": "request body must be a JSON object",
                                           "type": "invalid_request_error"}})
                return

            if body.get("stream"):
                result = _stream_completions(body, tokenizer, model, ceiling, kv_bits,
                                             hf_id, self)
                if result is not None:
                    status, err = result
                    self._json(status, err)
            else:
                status, resp = _handle_completions(body, tokenizer, model, ceiling,
                                                   kv_bits, hf_id)
                self._json(status, resp)

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return _Handler


# --------------------------------------------------------------------------- #
# Server entry-point
# --------------------------------------------------------------------------- #

def serve(hf_id: str, ceiling: int, *, margin_gb: float, overhead_gb: float,
          port: int, kv_bits: int | None = None,
          measured_slope_gb_per_k: float | None = None) -> None:
    """Gate, load, bind, handshake, serve. Exits 1 (never returns normally) on veto.

    Sequence:
    1. Safety gate BEFORE loading weights (Rule #1). Veto → stdout refusal + exit 1.
    2. ``mlx_lm.load(hf_id)`` — only reached when the gate passes.
    3. Bind a single-threaded ``HTTPServer`` on ``127.0.0.1:<port>``.
    4. Print ready-handshake JSON to stdout and flush (ARA blocks on this line).
    5. ``serve_forever()`` — runs until the process is killed.
    """
    refusal, effective_kv = _pre_load_gate(hf_id, ceiling, margin_gb=margin_gb,
                                           overhead_gb=overhead_gb, kv_bits=kv_bits,
                                           measured_slope_gb_per_k=measured_slope_gb_per_k)
    if refusal is not None:
        print(json.dumps({"refused": True, "reason": refusal["reason"]}), flush=True)
        sys.exit(1)

    # Lazy import — preserves refuse-before-load (Rule #1) and allows test monkeypatching.
    from mlx_lm import load  # type: ignore[import]

    try:
        model, tokenizer = load(hf_id)
    except Exception as exc:
        first = str(exc).splitlines()[0][:200] if str(exc) else ""
        print(json.dumps({"refused": True,
                          "reason": f"failed to load {hf_id}: {type(exc).__name__}: {first}"}),
              flush=True)
        sys.exit(1)

    # Register instruct turn-end tokens so generation self-stops (mlx_lm registers only the
    # scalar <eos>, not <end_of_turn> etc.) — else served instruct models ramble to max_tokens (#107).
    register_turn_end_tokens(tokenizer)

    handler_class = _make_handler(model, tokenizer, ceiling, effective_kv, hf_id)
    # Single-threaded on purpose: MLX's GPU stream is thread-local, so generation must run on the
    # same thread that loaded the model. Requests serialize — correct for a local one-model endpoint.
    server = HTTPServer(("127.0.0.1", port), handler_class)

    # IPC ready handshake — ARA blocks waiting for this exact line on stdout.
    print(json.dumps({"ready": True, "url": f"http://127.0.0.1:{port}",
                      "context": ceiling}), flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Governed OpenAI-compatible MLX endpoint."
    )
    ap.add_argument("hf_id")
    ap.add_argument("ctx_ceiling", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--kv-bits", type=int, default=None,
        help="KV-cache quantization bits (8 or 4); omit for fp16. "
             "Ignored for non-quantizable (RotatingKVCache) models.",
    )
    ap.add_argument(
        "--measured-slope", type=float, default=None,
        help="Measured ramp slope (GB per 1k tokens) fit from this model's characterization on "
             "this machine. When ARA serves the model's own measured ceiling it passes this so the "
             "gate predicts with the real growth, not the conservative a-priori slope. Omit for "
             "the a-priori gate (uncharacterized / --ctx serves).",
    )
    args = ap.parse_args(argv)
    serve(args.hf_id, args.ctx_ceiling, margin_gb=args.margin, overhead_gb=args.overhead,
          port=args.port, kv_bits=args.kv_bits,
          measured_slope_gb_per_k=args.measured_slope)


if __name__ == "__main__":
    main()
