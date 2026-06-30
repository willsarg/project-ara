# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Safe launch planning for mlx_lm.generate.

Two failure modes a naive launcher hits, and how we avoid them:
  1. Forcing `--kv-bits 4` on every model crashes RotatingKVCache models (Gemma,
     GPT-OSS) past 5000 tokens. We quantize ONLY models whose cache supports it.
  2. Budgeting against TOTAL RAM ignores the real wired wall (the live, per-machine
     working-set limit — 17.18 GB on the testbed) and the prefill spike. We budget against
     the measured per-model curve and the live system baseline, refusing to launch if the
     model can't even load without breaching the wall.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import config, db, models, profiles
from .system import read_limits, sample_settled_baseline

PREFILL_SPIKE_MULT = models.PREFILL_SPIKE_MULT  # source of truth: models.py
MIN_USEFUL_CTX = 512
KV_BITS = 4
KV_GROUP_SIZE = 64
QUANTIZED_KV_START = 5000
PROMPT_WARNING_FRACTION = 0.8


def _estimated_slope_gb_per_k(info: models.ModelInfo, kv_bits: int | None = None) -> float:
    """Conservative os-wired slope estimate for an uncharacterized model (GB per 1k tokens),
    at the given KV precision (fp16 by default)."""
    return info.estimated_slope_gb_per_k(kv_bits)


@dataclass(frozen=True)
class Prediction:
    base_abs_gb: float     # live baseline + model's own footprint at context -> 0
    headroom_gb: float     # threshold - base_abs (negative => won't even load safely)
    breaches_wall: bool    # base_abs alone crosses the hard wall — can't load at all
    safe_ctx: int          # max context before predicted peak hits the threshold


def predict(*, model_base_gb: float, slope_gb_per_k: float, live_base_gb: float,
            threshold_gb: float, wall_gb: float, model_max: int | None) -> Prediction:
    """Single source of truth for the crash-prediction math, shared by `plan` and `health`.

    Given a model's curve (base footprint + slope) and the live system baseline, work out
    the absolute base load and the safe context ceiling under the threshold. Keeping this
    in one place means `health`'s verdict can never drift from what `run` actually does.
    """
    base_abs = live_base_gb + model_base_gb
    headroom = threshold_gb - base_abs
    if headroom <= 0:
        cap = 0
    elif slope_gb_per_k > 0:
        cap = int(headroom / slope_gb_per_k * 1000)
    else:
        cap = 0
    if model_max:
        cap = min(cap, model_max)
    return Prediction(base_abs_gb=base_abs, headroom_gb=headroom,
                      breaches_wall=base_abs >= wall_gb, safe_ctx=cap)


def plan(hf_id: str, *, margin_gb: float | None = None, kv_bits: int | None = None) -> dict:
    """Decide a safe --max-kv-size for a launch at the given KV precision, or refuse.

    KV defaults to fp16 (``kv_bits=None``) — the memory-conservative, lossless choice; the
    caller opts into quant (8/4) explicitly. A non-quantizable cache type is forced to fp16
    regardless (quantizing a RotatingKVCache crashes past the quant threshold). A stored fit is
    reused only when it was measured at the *same* kv_bits; otherwise the q4-vs-fp16 slope
    mismatch would mis-predict memory, so we fall back to the conservative a-priori estimate.
    """
    info = models.describe(hf_id)
    if info is None:
        return {"error": f"model not found in HF cache: {hf_id}"}
    if not info.is_causal:
        return {"error": f"Model {hf_id} is not a supported causal language model."}

    margin_gb = config.margin_gb(margin_gb)
    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)

    wall = limits.wall_gb
    live_base = sample_settled_baseline()
    kv_bits = kv_bits if info.can_quantize_kv else None  # fp16 forced for non-quantizable caches

    con = db.connect()
    fit = db.latest_fit(con, hf_id)
    cold_source = None
    if fit and fit.get("slope_gb_per_k") and fit.get("fit_kv_bits") == kv_bits:
        model_base = float(fit["model_base_gb"])
        slope = float(fit["slope_gb_per_k"])
        source = "measured"
        fit_stale = models.fit_is_stale(hf_id, fit.get("characterized_at"))
    else:
        factor, overhead, cold_source = profiles.cold_start_constants(con)
        model_base = info.weights_gb * factor + overhead
        slope = _estimated_slope_gb_per_k(info, kv_bits)
        source = "estimated"
        fit_stale = False

    pred = predict(model_base_gb=model_base, slope_gb_per_k=slope, live_base_gb=live_base,
                   threshold_gb=threshold, wall_gb=wall, model_max=info.max_context)
    p = {
        "hf_id": hf_id, "kv_bits": kv_bits, "can_quantize": info.can_quantize_kv,
        "source": source,
        "fit_stale": fit_stale, "cold_start_profile": cold_source,
        "max_kv_size_enforced": info.max_kv_size_enforced,
        "kv_group_size": KV_GROUP_SIZE, "quantized_kv_start": QUANTIZED_KV_START,
        "cache_type": info.cache_type, "model_max": info.max_context,
        "live_base_gb": round(live_base, 2), "model_base_gb": round(model_base, 2),
        "base_abs_gb": round(pred.base_abs_gb, 2), "slope_gb_per_k": round(slope, 5),
        "threshold_gb": round(threshold, 2), "wall_gb": round(wall, 2),
    }

    # RULE #1: would it breach the wall just to load?
    if pred.breaches_wall:
        p["refuse"] = True
        p["reason"] = (f"weights+baseline ≈ {pred.base_abs_gb:.2f}GB ≥ wall {wall:.2f}GB — "
                       f"would breach the wall on load. Cannot run safely on this machine.")
        p["max_kv_size"] = 0
        return p

    cap = pred.safe_ctx
    p["max_kv_size"] = cap
    if cap < MIN_USEFUL_CTX:
        p["refuse"] = True
        p["reason"] = (f"safe cap {cap:,} tok < {MIN_USEFUL_CTX} — base leaves no useful "
                       f"context headroom.")
    else:
        p["refuse"] = False
    return p


class LaunchArgumentError(ValueError):
    """A passthrough argument violates the launcher's safety policy."""


@dataclass(frozen=True)
class PromptCheck:
    tokens: int
    cap: int
    warn: bool


def _option_values(argv: list[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    prefix = option + "="
    for index, arg in enumerate(argv):
        if arg == option:
            value = argv[index + 1] if index + 1 < len(argv) else None
            values.append(None if value is not None and value.startswith("--") else value)
        elif arg.startswith(prefix):
            values.append(arg[len(prefix):])
    return values


def _single_int_option(argv: list[str], option: str) -> int | None:
    values = _option_values(argv, option)
    if len(values) > 1:
        raise LaunchArgumentError(f"{option} may be provided only once")
    if not values:
        return None
    value = values[0]
    try:
        parsed = int(value) if value is not None else None
    except ValueError as exc:
        raise LaunchArgumentError(f"{option} requires an integer") from exc
    if parsed is None or parsed < 0:
        raise LaunchArgumentError(f"{option} requires a non-negative integer")
    return parsed


def _single_option(argv: list[str], *options: str) -> str | None:
    values: list[str | None] = []
    for option in options:
        values.extend(_option_values(argv, option))
    if len(values) > 1:
        raise LaunchArgumentError(f"{'/'.join(options)} may be provided only once")
    if not values:
        return None
    if values[0] is None:
        raise LaunchArgumentError(f"{options[0]} requires a value")
    return values[0]


def check_prompt(argv: list[str], p: dict, tokenizer) -> PromptCheck:
    """Count the prompt exactly as mlx_lm.generate prepares it, without loading weights."""
    if _option_values(argv, "--prompt-cache-file"):
        raise LaunchArgumentError(
            "prompt length cannot be verified with --prompt-cache-file; pass --force to override"
        )

    prompt = _single_option(argv, "--prompt", "-p")
    if prompt is None:
        from mlx_lm.generate import DEFAULT_PROMPT
        prompt = DEFAULT_PROMPT
    if prompt == "-":
        raise LaunchArgumentError(
            "prompt length cannot be verified from stdin; pass --force to override"
        )
    prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")

    ignore_template = bool(_option_values(argv, "--ignore-chat-template"))
    if not ignore_template and tokenizer.has_chat_template:
        system_prompt = _single_option(argv, "--system-prompt")
        prefill = _single_option(argv, "--prefill-response")
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if prefill is not None:
            messages.append({"role": "assistant", "content": prefill})
        template_config = _single_option(argv, "--chat-template-config")
        try:
            template_kwargs = json.loads(template_config) if template_config else {}
        except json.JSONDecodeError as exc:
            raise LaunchArgumentError("--chat-template-config requires valid JSON") from exc
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=prefill is not None,
            add_generation_prompt=prefill is None,
            **template_kwargs,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        tokens = tokenizer.encode(prompt)

    cap = effective_max_kv_size(argv, p)
    count = len(tokens)
    return PromptCheck(
        tokens=count,
        cap=cap,
        warn=count > cap * PROMPT_WARNING_FRACTION,
    )


def effective_max_kv_size(argv: list[str], p: dict) -> int:
    """Return the cap mlx_lm.generate will receive after launcher validation."""
    user_cap = _single_int_option(argv, "--max-kv-size")
    return user_cap if user_cap is not None else p["max_kv_size"]


def tokenizer_config(argv: list[str]) -> dict:
    """Mirror the tokenizer-specific configuration used by mlx_lm.generate."""
    return {
        "trust_remote_code": True
        if _option_values(argv, "--trust-remote-code")
        else None
    }


def build_argv(rest: list[str], p: dict, *, force: bool = False) -> list[str]:
    """Validate safety-sensitive passthrough args and inject planned defaults."""
    argv = list(rest)
    user_kv_bits = _single_int_option(argv, "--kv-bits")
    user_kv_group_size = _single_int_option(argv, "--kv-group-size")
    user_quantized_kv_start = _single_int_option(argv, "--quantized-kv-start")
    user_max_kv = _single_int_option(argv, "--max-kv-size")

    if not p.get("max_kv_size_enforced", True) and not force:
        raise LaunchArgumentError(
            "this model's custom MLX cache does not enforce --max-kv-size; "
            "pass --force to launch without a verified runtime cap"
        )

    kv_options = {
        "--kv-bits": user_kv_bits,
        "--kv-group-size": user_kv_group_size,
        "--quantized-kv-start": user_quantized_kv_start,
    }
    if not p["can_quantize"]:
        supplied = [option for option, value in kv_options.items() if value is not None]
        if supplied:
            raise LaunchArgumentError(
                f"{', '.join(supplied)} not supported because this model's cache "
                "is not quantizable"
            )
    elif not force:
        expected = {
            "--kv-bits": p["kv_bits"],
            "--kv-group-size": p["kv_group_size"],
            "--quantized-kv-start": p["quantized_kv_start"],
        }
        for option, value in kv_options.items():
            if value is not None and value != expected[option]:
                raise LaunchArgumentError(
                    f"{option} {value} does not match characterized setting "
                    f"{expected[option]}; pass --force to override"
                )

    if user_max_kv is not None and user_max_kv > p["max_kv_size"] and not force:
        raise LaunchArgumentError(
            f"--max-kv-size {user_max_kv:,} exceeds planned cap "
            f"{p['max_kv_size']:,}; pass --force to override"
        )

    for option in ("--draft-model", "--prompt-cache-file", "--adapter-path"):
        if _option_values(argv, option) and not force:
            raise LaunchArgumentError(
                f"{option} changes unmeasured memory behavior; pass --force to override"
            )

    if p["kv_bits"] is not None and user_kv_bits is None:
        argv = ["--kv-bits", str(p["kv_bits"])] + argv
    if p["kv_bits"] is not None and user_kv_group_size is None:
        argv = ["--kv-group-size", str(p["kv_group_size"])] + argv
    if p["kv_bits"] is not None and user_quantized_kv_start is None:
        argv = ["--quantized-kv-start", str(p["quantized_kv_start"])] + argv
    if user_max_kv is None:
        argv = ["--max-kv-size", str(p["max_kv_size"])] + argv
    return argv
