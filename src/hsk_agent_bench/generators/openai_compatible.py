"""Non-streaming OpenAI-compatible HTTP transport, with no implicit retries."""
import asyncio
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import Generator, ProviderError, seed_rejection_parameter
from ..schemas import GenerationRequest


class OpenAICompatibleGenerator(Generator):
    default_base_url: str
    token_limit_parameter = "max_tokens"

    def _payload(self, request: GenerationRequest, seed_applied: bool) -> dict:
        params = request.parameters
        payload = {
            "model": request.model_id, "messages": [message.model_dump() for message in request.messages],
            "temperature": params.temperature, "top_p": params.top_p,
            self.token_limit_parameter: params.max_output_tokens,
        }
        if seed_applied:
            payload["seed"] = params.seed
        if self.model.capabilities.json_output:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _send(self, payload: dict) -> dict:
        return await asyncio.to_thread(self._post, payload)

    def _post(self, payload: dict) -> dict:
        key = os.environ.get(self.model.api_key_env)
        if not key:
            raise ValueError(f"missing credential environment variable {self.model.api_key_env}")
        endpoint = (self.model.base_url or self.default_base_url).rstrip("/") + "/chat/completions"
        request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
        try:
            with urlopen(request, timeout=120) as response:
                raw = json.load(response)
                if isinstance(raw, dict) and response.headers.get("x-request-id"):
                    raw["_request_id"] = response.headers["x-request-id"]
                return raw
        except HTTPError as error:
            try:
                raw = json.loads(error.read().decode("utf-8"))
            except (ValueError, UnicodeError):
                raw = {}
            detail = raw.get("error", {}) if isinstance(raw, dict) else {}
            if not isinstance(detail, dict):
                detail = {}
            message = detail.get("message", f"HTTP {error.code}")
            raise ProviderError(message, status_code=error.code,
                                parameter=seed_rejection_parameter(message, detail.get("param")),
                                raw_response=raw if isinstance(raw, dict) else None) from error
        except (URLError, TimeoutError, OSError) as error:
            raise ProviderError("provider transport failed") from error
        except (ValueError, UnicodeError) as error:
            raise ProviderError("malformed provider response JSON") from error

    def _normalize(self, raw: dict) -> dict:
        choice = raw["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("response content must be text")
        usage = raw.get("usage") or {}
        return {
            "text": text, "request_id": raw.get("_request_id") or raw.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": {"input_tokens": usage.get("prompt_tokens"),
                      "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")},
        }
