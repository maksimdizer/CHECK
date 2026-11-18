import json
from urllib.parse import quote

import httpx

from .evaluation import OllamaProvider
from .settings import OLLAMA_TIMEOUT_SECONDS

CLOUD_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
}

ANTHROPIC_SCHEMA_LIMITS = {
    "minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
}
GEMINI_SCHEMA_LIMITS = {"const", "default", "minLength", "maxLength"}


def _provider_schema(schema, excluded):
    result = {name: value for name, value in schema.items() if name not in excluded}
    for name in ("properties", "$defs"):
        if name in result:
            result[name] = {key: _provider_schema(value, excluded) for key, value in result[name].items()}
    if isinstance(result.get("items"), dict):
        result["items"] = _provider_schema(result["items"], excluded)
    for name in ("anyOf", "oneOf", "allOf"):
        if name in result:
            result[name] = [_provider_schema(value, excluded) for value in result[name]]
    return result


class CloudChatProvider(OllamaProvider):
    def __init__(self, model, *, provider, api_key, **options):
        super().__init__(model, **options)
        self.provider = provider
        self.api_key = api_key
        self.cache_id = None

    def _generate(self, request, contract, prompt, *, schema=None, think=None):
        schema = contract.model_json_schema() if schema is None else schema
        messages = self._messages(request, prompt)
        system, user = (message["content"] for message in messages)
        instruction = "\nReturn one JSON object conforming to this schema: " + json.dumps(schema, ensure_ascii=False)
        endpoint = CLOUD_ENDPOINTS[self.provider]
        if self.provider == "anthropic":
            data = self._post(endpoint, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, {
                "model": self.model,
                "system": system + instruction,
                "messages": [messages[1]],
                "max_tokens": self.num_predict,
                "output_config": {"format": {"type": "json_schema",
                                             "schema": _provider_schema(schema, ANTHROPIC_SCHEMA_LIMITS)}},
            })
            blocks = data.get("content") or []
            content = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            usage = data.get("usage") or {}
            cached = usage.get("cache_read_input_tokens", 0)
            tokens = {
                "prompt_eval_count": usage.get("input_tokens", 0) + cached + usage.get("cache_creation_input_tokens", 0),
                "prompt_eval_cached_count": cached,
                "eval_count": usage.get("output_tokens", 0),
            }
        elif self.provider == "google":
            model = quote(self.model.removeprefix("models/"), safe="")
            data = self._post(f"{endpoint}/{model}:generateContent", {"x-goog-api-key": self.api_key}, {
                "systemInstruction": {"parts": [{"text": system + instruction}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "maxOutputTokens": self.num_predict,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": _provider_schema(schema, GEMINI_SCHEMA_LIMITS),
                },
            })
            candidates = data.get("candidates") or []
            first = candidates[0] if candidates else {}
            parts = (first.get("content") or {}).get("parts") or []
            content = "".join(part.get("text", "") for part in parts if not part.get("thought"))
            usage = data.get("usageMetadata") or {}
            tokens = {
                "prompt_eval_count": usage.get("promptTokenCount", 0),
                "prompt_eval_cached_count": usage.get("cachedContentTokenCount", 0),
                "eval_count": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            }
        else:
            messages[0]["content"] += instruction
            data = self._post(endpoint, {"Authorization": "Bearer " + self.api_key}, {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.num_predict,
                "response_format": {"type": "json_object"},
                **({"store": False} if self.provider == "openai" else {"provider": {"require_parameters": True}}),
            })
            choices = data.get("choices") or []
            message = choices[0].get("message") if choices else None
            content = message.get("content") if isinstance(message, dict) else None
            usage = data.get("usage") or {}
            tokens = {"prompt_eval_count": usage.get("prompt_tokens", 0), "eval_count": usage.get("completion_tokens", 0)}
            details = usage.get("prompt_tokens_details") or {}
            if "cached_tokens" in details:
                tokens["prompt_eval_cached_count"] = details["cached_tokens"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The cloud model did not return a JSON answer.")
        return contract.model_validate_json(content), tokens

    @staticmethod
    def _post(endpoint, headers, payload):
        response = httpx.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=5),
            trust_env=False,
        )
        response.raise_for_status()
        return response.json()
