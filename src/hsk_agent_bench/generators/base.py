"""Provider-neutral asynchronous generation and explicit sampling provenance."""
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import re
from time import perf_counter
from typing import Any

from ..config import ModelConfig
from ..prompts.common import hash_messages
from ..schemas import GenerationRecord, GenerationRequest

Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def seed_rejection_parameter(message: str, parameter: str | None = None) -> str | None:
    """Recognize explicit seed errors without treating every HTTP 400 as one."""
    if parameter is not None:
        return parameter
    seed_error = re.search(
        r"(?:unsupported|unknown|unrecognized)(?:\s+(?:parameter|argument|field))?\s*['\"]?seed\b"
        r"|\bseed['\"]?(?:\s+(?:parameter|argument|field))?\s+(?:is\s+)?(?:not\s+(?:supported|allowed)|unsupported|unknown|unrecognized)",
        message, re.IGNORECASE,
    )
    return "seed" if seed_error else None


class ProviderError(Exception):
    """A failed provider call; only explicit HTTP statuses qualify for retries."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 parameter: str | None = None, raw_response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.parameter = parameter
        self.raw_response = raw_response


class Generator(ABC):
    provider: str
    supports_seed = True

    def __init__(self, model: ModelConfig, *, transport: Transport | None = None):
        if model.provider != self.provider:
            raise ValueError(f"expected provider {self.provider}")
        self.model = model.model_copy(deep=True)
        self.transport = transport or self._send

    @abstractmethod
    async def _send(self, payload: dict) -> dict: ...

    @abstractmethod
    def _payload(self, request: GenerationRequest, seed_applied: bool) -> dict: ...

    @abstractmethod
    def _normalize(self, raw: dict) -> dict: ...

    async def generate(self, request: GenerationRequest) -> GenerationRecord:
        """An accepted seed is best effort, not a promise of determinism.

        Only a seed-specific rejection permits a single seedless fallback.
        Other parameter errors propagate rather than changing the experiment.
        """
        for field in ("provider", "family", "origin", "model_id"):
            if getattr(request, field) != getattr(self.model, field):
                raise ValueError(f"request {field} does not match model configuration")
        if request.prompt_hash != hash_messages(request.messages):
            raise ValueError("request prompt hash does not match rendered messages")
        for parameter in ("temperature", "top_p"):
            if not getattr(self.model.capabilities, parameter):
                raise ValueError(f"model does not support requested {parameter}")
        seed = request.parameters.seed
        applied = seed is not None and self.supports_seed and self.model.capabilities.seed
        seed_status = "requested" if applied else ("not_requested" if seed is None else "unsupported")
        requested_at = datetime.now(timezone.utc)
        started = perf_counter()
        rejection = None
        try:
            raw = await self.transport(self._payload(request, applied))
        except ProviderError as error:
            if not (applied and error.status_code in {400, 422} and error.parameter == "seed"):
                raise
            applied, seed_status = False, "rejected"
            rejection = error.raw_response or {"message": str(error), "status_code": error.status_code}
            raw = await self.transport(self._payload(request, False))
        latency = (perf_counter() - started) * 1000
        if not isinstance(raw, dict):
            raise ProviderError("malformed provider response: expected an object", raw_response={"response": raw})
        try:
            normalized = self._normalize(raw)
            raw = dict(raw)
            raw["_benchmark"] = {
                "provider": self.provider, "seed_status": seed_status,
                "seed_rejection": rejection, "capabilities": self.model.capabilities.model_dump(),
            }
            return GenerationRecord(**request.model_dump(), **normalized, raw_response=raw,
                                    requested_at=requested_at, latency_ms=latency, seed_applied=applied)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ProviderError(f"malformed provider response: {error}", raw_response=raw) from error
