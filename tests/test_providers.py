"""Mock only network boundaries; payloads and normalization use real adapters."""
import asyncio
import copy
import io
import json
from pathlib import Path
import sys
import types
import tomllib
from urllib.error import HTTPError

import pytest

from hsk_agent_bench.config import ModelConfig
from hsk_agent_bench.generators import make_generator
from hsk_agent_bench.generators.base import ProviderError
from hsk_agent_bench.prompts.common import hash_messages
from hsk_agent_bench.schemas import ChatMessage, GenerationParameters, GenerationRequest


def model(provider, seed=True, **changes):
    values = dict(family=provider, origin="china" if provider in {"qwen", "deepseek"} else "global",
                  provider=provider, model_id=f"{provider}-frozen-id", api_key_env=f"{provider.upper()}_API_KEY",
                  capabilities={"seed": seed})
    values.update(changes)
    return ModelConfig(**values)


def request(config):
    messages = [ChatMessage(role="system", content="Return JSON."), ChatMessage(role="user", content="你好")]
    return GenerationRequest(generation_id="generation-1", prompt_id="p1", task="reading", hsk_level=3,
                             family=config.family, origin=config.origin, provider=config.provider,
                             model_id=config.model_id, prompt_hash=hash_messages(messages), messages=messages,
                             parameters=GenerationParameters(temperature=.4, top_p=.9, max_output_tokens=100, seed=7))


def fixture(provider):
    return json.loads((Path(__file__).parent / "fixtures/provider_responses" / f"{provider}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("provider", ["openai", "qwen", "deepseek", "gemini"])
def test_normalizes_provider_response_and_keeps_exact_request_metadata(provider):
    config = model(provider)
    calls = []
    async def transport(payload):
        calls.append(copy.deepcopy(payload))
        return fixture(provider)
    req = request(config)
    record = asyncio.run(make_generator(config, transport=transport).generate(req))
    assert record.text == "你好，世界。"
    assert record.usage.model_dump() == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
    assert record.request_id == ("gemini-response" if provider == "gemini" else f"chatcmpl-{ 'gpt' if provider == 'openai' else provider}")
    assert record.finish_reason == ("STOP" if provider == "gemini" else "stop")
    assert record.model_id == f"{provider}-frozen-id"
    assert record.provider == provider and record.prompt_hash == req.prompt_hash
    assert record.parameters.seed == 7 and record.seed_applied is (provider != "deepseek")
    assert record.requested_at.tzinfo is not None and record.latency_ms >= 0
    assert record.raw_response["_benchmark"]["provider"] == provider
    for key, value in fixture(provider).items():
        assert record.raw_response[key] == value
    payload = calls[0]
    assert payload["model"] == config.model_id
    if provider == "gemini":
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "你好"}]}]
        assert payload["config"]["system_instruction"] == "Return JSON."
        params = payload["config"]
        assert params["max_output_tokens"] == 100
    else:
        assert payload["messages"] == [m.model_dump() for m in req.messages]
        params = payload
        assert params["max_completion_tokens" if provider == "openai" else "max_tokens"] == 100
    assert params["temperature"] == .4 and params["top_p"] == .9
    assert params.get("seed") == (None if provider == "deepseek" else 7)


@pytest.mark.parametrize("provider", ["openai", "qwen", "gemini"])
def test_unsupported_seed_is_explicit_and_sampling_is_preserved(provider):
    config = model(provider, seed=False)
    async def transport(payload):
        assert "seed" not in payload.get("config", payload)
        return fixture(provider)
    record = asyncio.run(make_generator(config, transport=transport).generate(request(config)))
    assert record.seed_applied is False
    assert record.parameters.seed == 7
    assert record.raw_response["_benchmark"]["seed_status"] == "unsupported"


@pytest.mark.parametrize("provider", ["openai", "qwen", "gemini"])
def test_rejected_seed_is_recorded_and_only_seed_changes_on_fallback(provider):
    config = model(provider)
    calls = []
    async def transport(payload):
        calls.append(copy.deepcopy(payload))
        if len(calls) == 1:
            raise ProviderError("seed unsupported", status_code=400, parameter="seed", raw_response={"error": "seed unsupported"})
        return fixture(provider)
    record = asyncio.run(make_generator(config, transport=transport).generate(request(config)))
    assert record.seed_applied is False and record.parameters.seed == 7
    assert record.raw_response["_benchmark"]["seed_status"] == "rejected"
    assert record.raw_response["_benchmark"]["seed_rejection"] == {"error": "seed unsupported"}
    first = calls[0].get("config", calls[0])
    second = calls[1].get("config", calls[1])
    assert first.pop("seed") == 7 and first == second


def test_unrelated_bad_request_does_not_trigger_seed_fallback():
    config = model("openai")
    async def transport(payload):
        raise ProviderError("invalid schema", status_code=400, parameter="response_format")
    with pytest.raises(ProviderError, match="invalid schema"):
        asyncio.run(make_generator(config, transport=transport).generate(request(config)))


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_refuses_unsupported_sampling_or_mismatched_request_before_transport(provider):
    config = model(provider, capabilities={"temperature": False})
    async def transport(payload):
        pytest.fail("invalid request must not reach network")
    generator = make_generator(config, transport=transport)
    with pytest.raises(ValueError, match="temperature"):
        asyncio.run(generator.generate(request(config)))
    config = model(provider)
    req = request(config).model_copy(update={"model_id": "other-id"})
    with pytest.raises(ValueError, match="model"):
        asyncio.run(make_generator(config, transport=transport).generate(req))
    req = request(config).model_copy(update={"prompt_hash": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        asyncio.run(make_generator(config, transport=transport).generate(req))


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_malformed_response_is_a_permanent_provider_failure(provider):
    async def transport(payload):
        return {"unexpected": "payload"}
    config = model(provider)
    with pytest.raises(ProviderError, match="response") as error:
        asyncio.run(make_generator(config, transport=transport).generate(request(config)))
    assert error.value.status_code is None


def test_http_boundary_records_header_id_and_seed_rejection_without_param(monkeypatch):
    import hsk_agent_bench.generators.openai_compatible as http
    config = model("qwen", base_url="https://fixture.invalid/v1")
    monkeypatch.setenv("QWEN_API_KEY", "test-key-only")
    sent = []
    class Response(io.BytesIO):
        headers = {"x-request-id": "header-request-id"}
    def urlopen(req, *, timeout):
        assert req.full_url == "https://fixture.invalid/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer test-key-only"
        sent.append(json.loads(req.data))
        if len(sent) == 1:
            body = io.BytesIO(b'{"error":{"message":"seed is not supported","code":"InvalidParameter"}}')
            raise HTTPError(req.full_url, 400, "bad request", {}, body)
        return Response(json.dumps(fixture("qwen"), ensure_ascii=False).encode("utf-8"))
    monkeypatch.setattr(http, "urlopen", urlopen)
    record = asyncio.run(make_generator(config).generate(request(config)))
    assert record.text == "你好，世界。" and record.request_id == "header-request-id"
    assert record.seed_applied is False and record.raw_response["_benchmark"]["seed_status"] == "rejected"
    sent[0].pop("seed")
    assert sent[0] == sent[1]


def test_gemini_sdk_boundary_uses_own_config_no_sdk_retries_and_closes_client(monkeypatch):
    config = model("gemini", base_url="https://fixture.invalid")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-only")
    calls, clients, closed, http_clients = [], [], [], []
    class APIError(Exception):
        code = 400
    class Response:
        def model_dump(self, **options):
            return fixture("gemini")
    class Models:
        async def generate_content(self, **payload):
            calls.append(copy.deepcopy(payload))
            if len(calls) == 1:
                raise APIError("seed is not supported")
            return Response()
    class AsyncClient:
        models = Models()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            closed.append(True)
    class Client:
        def __init__(self, **options):
            clients.append(options)
            self.aio = AsyncClient()
    class HTTPClient:
        def __init__(self, **options):
            self.options = options
            self.closed = False
            http_clients.append(self)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            self.closed = True
    google, genai, errors = types.ModuleType("google"), types.ModuleType("google.genai"), types.ModuleType("google.genai.errors")
    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = HTTPClient
    google.genai, genai.Client, errors.APIError = genai, Client, APIError
    for name, module in [("google", google), ("google.genai", genai), ("google.genai.errors", errors), ("httpx", httpx)]:
        monkeypatch.setitem(sys.modules, name, module)
    record = asyncio.run(make_generator(config).generate(request(config)))
    assert record.text == "你好，世界。" and record.seed_applied is False
    assert len(closed) == 2
    assert len(http_clients) == 2 and all(client.closed for client in http_clients)
    assert clients[0]["http_options"] == {"retry_options": {"attempts": 1}, "base_url": "https://fixture.invalid",
                                          "httpx_async_client": http_clients[0]}
    calls[0]["config"].pop("seed")
    assert calls[0] == calls[1]


def test_gemini_missing_optional_extra_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    config = model("gemini")
    with pytest.raises(RuntimeError, match=r"hsk-agent-bench\[providers\]"):
        asyncio.run(make_generator(config).generate(request(config)))


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_nonobject_json_response_keeps_a_serializable_failure_payload(provider):
    async def transport(payload):
        return ["invalid", "response"]
    config = model(provider)
    with pytest.raises(ProviderError) as error:
        asyncio.run(make_generator(config, transport=transport).generate(request(config)))
    assert error.value.raw_response == {"response": ["invalid", "response"]}


def test_gemini_dependency_range_excludes_sdk_without_required_async_features():
    from packaging.requirements import Requirement
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(Requirement(value) for value in metadata["project"]["optional-dependencies"]["providers"]
                       if Requirement(value).name == "google-genai")
    # 1.0.0 lacks the async context manager; 1.52.0 is verified below against
    # real SDK classes, configuration validation, serializers, and HTTP routing.
    assert "1.0.0" not in requirement.specifier
    assert "1.51.0" not in requirement.specifier
    assert "1.52.0" in requirement.specifier
    assert "2.0.0" not in requirement.specifier


@pytest.mark.parametrize("reject_seed", [False, True])
def test_real_gemini_sdk_offline_http_serialization_and_seed_fallback(monkeypatch, reject_seed):
    pytest.importorskip("google.genai", reason="install the providers extra for real-SDK compatibility checks")
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    sent, clients = [], []
    def handler(wire_request):
        sent.append(json.loads(wire_request.content))
        assert wire_request.url.host == "fixture.invalid"
        assert wire_request.url.path.endswith("/models/gemini-frozen-id:generateContent")
        if reject_seed and len(sent) == 1:
            return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                       "message": "seed is not supported"}})
        return httpx.Response(200, json={
            "responseId": "offline-gemini-response", "modelVersion": "gemini-returned-version",
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [
                {"text": "hidden thought", "thought": True}, {"text": "你好，世界。"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8, "totalTokenCount": 20},
        })
    real_async_client = httpx.AsyncClient
    def offline_async_client(**options):
        client = real_async_client(**options, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client
    # Keep the real SDK and HTTP client; replace only the network transport.
    monkeypatch.setattr(httpx, "AsyncClient", offline_async_client)
    config = model("gemini", base_url="https://fixture.invalid", capabilities={"seed": True, "json_output": True})
    record = asyncio.run(make_generator(config).generate(request(config)))
    assert record.text == "你好，世界。" and record.request_id == "offline-gemini-response"
    assert record.finish_reason == "STOP" and record.seed_applied is not reject_seed
    assert record.usage.model_dump() == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
    assert sent[-1]["generationConfig"] == {"temperature": .4, "topP": .9, "maxOutputTokens": 100,
                                           "responseMimeType": "application/json", **({} if reject_seed else {"seed": 7})}
    assert sent[-1]["contents"] == [{"parts": [{"text": "你好"}], "role": "user"}]
    assert sent[-1]["systemInstruction"]["parts"] == [{"text": "Return JSON."}]
    assert len(sent) == (2 if reject_seed else 1)
    assert all(client.is_closed for client in clients)


def test_real_gemini_sdk_offline_rate_limit_has_no_implicit_retries(monkeypatch):
    pytest.importorskip("google.genai", reason="install the providers extra for real-SDK compatibility checks")
    import httpx
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    sent = []
    def handler(wire_request):
        sent.append(wire_request)
        return httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "rate limited"}})
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **options: real_async_client(**options, transport=httpx.MockTransport(handler)))
    config = model("gemini", base_url="https://fixture.invalid")
    with pytest.raises(ProviderError) as error:
        asyncio.run(make_generator(config).generate(request(config)))
    assert error.value.status_code == 429
    assert len(sent) == 1
