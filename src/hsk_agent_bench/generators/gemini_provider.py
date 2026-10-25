"""Google Gen AI SDK adapter, separate from OpenAI-compatible providers.

Live use requires `google-genai>=1.52.0,<2`. Imports and credentials are deferred until
a real call; fixture tests never need either.
"""
import os

from .base import Generator, ProviderError, seed_rejection_parameter
from ..schemas import GenerationRequest


class GeminiGenerator(Generator):
    provider = "gemini"

    def _payload(self, request: GenerationRequest, seed_applied: bool) -> dict:
        config = {"temperature": request.parameters.temperature, "top_p": request.parameters.top_p,
                  "max_output_tokens": request.parameters.max_output_tokens}
        system = [message.content for message in request.messages if message.role == "system"]
        if system:
            config["system_instruction"] = "\n\n".join(system)
        if seed_applied:
            config["seed"] = request.parameters.seed
        if self.model.capabilities.json_output:
            config["response_mime_type"] = "application/json"
        return {"model": request.model_id, "config": config, "contents": [
            {"role": "model" if message.role == "assistant" else "user", "parts": [{"text": message.content}]}
            for message in request.messages if message.role != "system"
        ]}

    async def _send(self, payload: dict) -> dict:
        try:
            import httpx
            from google import genai
            from google.genai.errors import APIError
        except ImportError as error:
            raise RuntimeError('Gemini requires google-genai; install "hsk-agent-bench[providers]" before live collection') from error
        key = os.environ.get(self.model.api_key_env)
        if not key:
            raise ValueError(f"missing credential environment variable {self.model.api_key_env}")
        http_options = {"retry_options": {"attempts": 1}}
        if self.model.base_url:
            http_options["base_url"] = self.model.base_url
        try:
            # Supplying httpx explicitly prevents SDK versions from choosing an
            # installed aiohttp transport with an independent reconnect retry.
            async with httpx.AsyncClient(timeout=120, trust_env=True) as http_client:
                http_options["httpx_async_client"] = http_client
                async with genai.Client(api_key=key, http_options=http_options).aio as client:
                    response = await client.models.generate_content(**payload)
                    return response.model_dump(mode="json", exclude_none=True)
        except APIError as error:
            message = str(error)
            raise ProviderError(message, status_code=error.code, parameter=seed_rejection_parameter(message),
                                raw_response={"message": message, "status_code": error.code}) from error

    def _normalize(self, raw: dict) -> dict:
        candidate = raw["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))
        usage = raw.get("usage_metadata") or {}
        return {"text": text, "request_id": raw.get("response_id"), "finish_reason": candidate.get("finish_reason"),
                "usage": {"input_tokens": usage.get("prompt_token_count"),
                          "output_tokens": usage.get("candidates_token_count"), "total_tokens": usage.get("total_token_count")}}
