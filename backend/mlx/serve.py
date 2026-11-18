import json
import argparse
from threading import Lock
from typing import Any
from uuid import UUID
from contextlib import asynccontextmanager
from copy import copy
from os.path import commonprefix

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from mlx_vlm import generate, load
from mlx_vlm.apc import APCManager
from mlx_vlm.models.cache import KVCache, RotatingKVCache, cache_nbytes
from mlx_vlm.structured import ThinkingAwareLogitsProcessor, build_json_schema_logits_processor
from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import StoppingCriteria, get_model_path, load_config, load_model
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODEL = "qwen3.8:27b-3bit-mlx"
DEFAULT_WINDOW = 4096
DEFAULT_MAX_OUTPUT = 300
DEFAULT_THINKING_BUDGET = 192
THINKING_MARKER_RESERVE = 8
generation_lock = Lock()


class ConversationPrefixCache(APCManager):
    """For long prompts, retain the shared transcript before the criterion suffix."""

    stable_prefix_length = 0
    transcript_checkpoint = None

    def resident_bytes(self):
        if self.transcript_checkpoint is not None:
            return cache_nbytes(self.transcript_checkpoint[2])
        return super().resident_bytes()

    def checkpoint_prefix_length(self, token_ids, *, extra_hash=0):
        return self.stable_prefix_length or super().checkpoint_prefix_length(token_ids, extra_hash=extra_hash)

    def store_exact_cache(self, token_ids, prompt_cache, *, extra_hash=0):
        if not self.stable_prefix_length:
            return super().store_exact_cache(token_ids, prompt_cache, extra_hash=extra_hash)
        if len(token_ids) != self.stable_prefix_length:
            return False
        self.transcript_checkpoint = None
        if any(type(layer) not in (KVCache, RotatingKVCache) for layer in prompt_cache):
            return False
        if not self.has_room_for_transcript(prompt_cache):
            self.stats.memory_skips += 1
            return False
        self.transcript_checkpoint = (tuple(token_ids), extra_hash, self.fork_layers(prompt_cache))
        self.stats.exact_stores += 1
        return True

    def lookup_exact_cache(self, token_ids, extra_hash=0, max_prefix_tokens=None, min_prefix_tokens=0, *,
                           defer_on_pressure=False):
        if not self.stable_prefix_length:
            return super().lookup_exact_cache(
                token_ids, extra_hash, max_prefix_tokens, min_prefix_tokens,
                defer_on_pressure=defer_on_pressure,
            )
        if self.transcript_checkpoint is None:
            return None, 0
        prefix, stored_hash, layers = self.transcript_checkpoint
        bound = min(max_prefix_tokens or len(token_ids), len(token_ids) - 1)
        if stored_hash != extra_hash or not min_prefix_tokens < len(prefix) <= bound:
            return None, 0
        if tuple(token_ids[:len(prefix)]) != prefix:
            return None, 0
        if not self.has_room_for_transcript(layers):
            self.transcript_checkpoint = None
            self.stats.memory_skips += 1
            return None, 0
        self.stats.exact_hits += 1
        self.stats.hits += 1
        self.stats.matched_tokens += len(prefix)
        return self.fork_layers(layers), len(prefix)

    def has_room_for_transcript(self, layers):
        # Reserve room for a full copy as individual layers diverge on writes.
        size = cache_nbytes(layers)
        mx.clear_cache()
        return size <= self.memory_max_bytes and size + self.memory_reserve_bytes <= self._memory_headroom()

    @staticmethod
    def fork_layers(layers):
        forks = []
        for layer in layers:
            fork = copy(layer)
            # Distinct MLX array handles share the evaluated buffers until a
            # branch writes. Sharing the handles themselves would mutate both.
            if layer.keys is not None:
                fork.keys = mx.array(layer.keys)
                fork.values = mx.array(layer.values)
            forks.append(fork)
        return forks


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    format: dict[str, Any]
    options: dict[str, Any] = Field(default_factory=dict)
    cache_id: UUID | None = None
    think: bool = False

def load_selected_model(app, model_id):
    state = app.state
    if state.model_id == model_id:
        return
    model_path = get_model_path(state.model_paths[model_id])
    gemma = load_config(model_path)["model_type"] in {"gemma4", "gemma4_unified"}
    # Keep only the selected weights and discard model-specific KV cache.
    state.model_id = state.model = state.processor = None
    state.cache_id = state.cache = None
    mx.clear_cache()
    if gemma:
        # The QA endpoint only accepts text. Gemma's multimodal processor also
        # loads Torchvision, which is unnecessary when we supply token IDs.
        model = load_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.detokenizer = load_tokenizer(model_path, return_tokenizer=False)(tokenizer)
        tokenizer.stopping_criteria = StoppingCriteria(model.config.eos_token_id, tokenizer)
        processor = tokenizer
    else:
        model, processor = load(model_path)
    state.model, state.processor, state.gemma, state.model_id = model, processor, gemma, model_id


@asynccontextmanager
async def lifespan(app):
    app.state.model_id = None
    load_selected_model(app, app.state.default_model_id)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/api/version")
def version():
    return {"version": "mlx-vlm", "model": app.state.model_id, "context_length": app.state.context_length,
            "available_models": list(app.state.model_paths),
            "thinking_budget": app.state.thinking_budget, "max_output_tokens": app.state.max_output_tokens}

@app.post("/api/chat")
def chat(body: ChatRequest):
    if body.model not in app.state.model_paths:
        raise HTTPException(404, "Model is not configured")
    limit = app.state.max_output_tokens
    max_tokens = body.options.get("num_predict", min(DEFAULT_MAX_OUTPUT, limit))
    if type(max_tokens) is not int or not 1 <= max_tokens <= limit:
        raise HTTPException(422, f"num_predict must be between 1 and {limit}")
    window = body.options.get("num_ctx", app.state.context_length)
    if type(window) is not int or not 1 <= window <= app.state.context_length:
        raise HTTPException(422, f"num_ctx must be between 1 and {app.state.context_length}")
    reasoning = body.options.get("thinking_budget", app.state.thinking_budget)
    if type(reasoning) is not int or not 1 <= reasoning <= app.state.thinking_budget:
        raise HTTPException(422, f"thinking_budget must be between 1 and {app.state.thinking_budget}")
    with generation_lock:
        load_selected_model(app, body.model)
        if app.state.cache_id != body.cache_id:
            app.state.cache_id = body.cache_id
            app.state.cache = (
                ConversationPrefixCache(num_blocks=0, overrides={
                    "checkpoint_entries": 2,
                    "memory_reserve_gb": 0.25,
                    "checkpoint_interval_tokens": 512,
                }) if body.cache_id else None
            )
        return generate_answer(body, max_tokens, window, reasoning)


@app.delete("/api/prompt-cache/{cache_id}", status_code=204)
def release_prompt_cache(cache_id: UUID):
    with generation_lock:
        if app.state.cache_id == cache_id:
            app.state.cache_id = None
            app.state.cache = None
            mx.clear_cache()


def generate_answer(body: ChatRequest, max_tokens: int, window: int, reasoning: int):
    gemma = app.state.gemma
    tokenizer = app.state.processor if gemma else app.state.processor.tokenizer
    thinking_start, thinking_end = ("<|channel>", "<channel|>") if gemma else ("<think>", "</think>")
    messages = body.messages
    sources_only = None
    if body.cache_id and messages and messages[-1]["role"] == "user":
        payload = json.loads(messages[-1]["content"])
        sources = payload.pop("sources")
        messages = [*messages[:-1], {
            **messages[-1], "content": json.dumps({"sources": sources, **payload}, ensure_ascii=False),
        }]
        if gemma and window > 8192:
            sources_only = [*messages[:-1], {
                **messages[-1], "content": json.dumps({"sources": sources}, ensure_ascii=False),
            }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=body.think,
    )
    # Gemma's chat template already includes its BOS token.
    token_ids = tokenizer.encode(prompt, add_special_tokens=not gemma)
    available = window - len(token_ids) - max_tokens
    thinking_budget = min(reasoning, available - THINKING_MARKER_RESERVE) if body.think else 0
    generation_budget = max_tokens + (thinking_budget + THINKING_MARKER_RESERVE if body.think else 0)
    if available < 0 or (body.think and thinking_budget < 1):
        raise HTTPException(422, f"Input and output budget exceed the {window}-token context")
    if app.state.cache is not None:
        app.state.cache.stable_prefix_length = 0
        if sources_only and len(token_ids) > 8192:
            prefix_prompt = tokenizer.apply_chat_template(
                sources_only, tokenize=False, add_generation_prompt=True, enable_thinking=body.think,
            )
            prefix_ids = tokenizer.encode(prefix_prompt, add_special_tokens=False)
            app.state.cache.stable_prefix_length = len(commonprefix((token_ids, prefix_ids)))
    grammar = build_json_schema_logits_processor(tokenizer, body.format)
    thinking_options = {}
    if body.think:
        grammar = ThinkingAwareLogitsProcessor(
            grammar, tokenizer, thinking_start_token=thinking_start,
            thinking_end_token=thinking_end, enable_thinking=True,
        )
        thinking_options = {
            "enable_thinking": True, "thinking_budget": thinking_budget,
            "thinking_start_token": thinking_start, "thinking_end_token": thinking_end,
        }
    mx.clear_cache()
    mx.reset_peak_memory()
    result = generate(
        app.state.model, app.state.processor, prompt, max_tokens=generation_budget,
        temperature=0, prefill_step_size=512, logits_processors=[grammar],
        apc_manager=app.state.cache,
        **thinking_options,
        # This text-only request has no padding. An all-ones attention mask
        # would make MLX-VLM hash the full prompt length into the prefix key.
        **({"input_ids": mx.array([token_ids])} if gemma or app.state.cache is not None else {}),
    )
    content = result.text
    reasoning_tokens = 0
    if body.think:
        end_token_id = tokenizer.encode(thinking_end, add_special_tokens=False)[-1]
        if end_token_id not in result.token_ids:
            raise HTTPException(422, "Model did not finish its reasoning within the generation budget")
        boundary = result.token_ids.index(end_token_id) + 1
        reasoning_tokens = boundary
        content = tokenizer.decode(result.token_ids[boundary:], skip_special_tokens=True).strip()
    print(json.dumps({
        "input_tokens": result.prompt_tokens, "output_tokens": result.generation_tokens,
        "finish_reason": result.finish_reason,
        "reasoning_tokens": reasoning_tokens,
        "thinking_budget": thinking_budget,
        "cached_tokens": result.cached_tokens,
        "cache_stores": app.state.cache.stats.exact_stores if app.state.cache else 0,
        "cache_memory_skips": app.state.cache.stats.memory_skips if app.state.cache else 0,
        "cache_resident_bytes": app.state.cache.resident_bytes() if app.state.cache else 0,
        "prefill_tps": round(result.prompt_tps, 2), "decode_tps": round(result.generation_tps, 2),
        "peak_memory_gb": round(result.peak_memory, 2),
    }), flush=True)
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": result.prompt_tokens, "eval_count": result.generation_tokens,
        "prompt_eval_cached_count": result.cached_tokens,
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local QA evaluator on Apple MLX")
    parser.add_argument("model_path", help="Local directory containing the quantized model")
    parser.add_argument("--model-id", default=MODEL, help="Model identifier exposed by the QA API")
    parser.add_argument("--extra-model", metavar="ID=PATH", action="append", default=[],
                        help="Additional model to load on demand from a local directory")
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--context-length", type=int, choices=(4096, 8192, 49152, 65536), default=DEFAULT_WINDOW)
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT,
                        help="Maximum num_predict for the structured answer, excluding the thinking budget")
    args = parser.parse_args()
    if args.thinking_budget <= 0:
        parser.error("--thinking-budget must be positive")
    if not 1 <= args.max_output_tokens < args.context_length:
        parser.error("--max-output-tokens must be positive and smaller than the context")
    model_paths = {args.model_id: args.model_path}
    for entry in args.extra_model:
        model_id, _, path = entry.partition("=")
        if not model_id or not path or model_id in model_paths:
            parser.error("--extra-model requires a unique ID and a path in ID=PATH format")
        model_paths[model_id] = path
    app.state.thinking_budget = args.thinking_budget
    app.state.max_output_tokens = args.max_output_tokens
    app.state.context_length = args.context_length
    app.state.default_model_id = args.model_id
    app.state.model_paths = model_paths
    uvicorn.run(app, host="127.0.0.1", port=args.port)
