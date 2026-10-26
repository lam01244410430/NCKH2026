"""Factory for the four explicitly supported generation providers."""
from ..config import ModelConfig
from .base import Generator, ProviderError, Transport
from .deepseek_provider import DeepSeekGenerator
from .gemini_provider import GeminiGenerator
from .openai_provider import OpenAIGenerator
from .qwen_provider import QwenGenerator


def make_generator(model: ModelConfig, *, transport: Transport | None = None) -> Generator:
    adapters = {"openai": OpenAIGenerator, "gemini": GeminiGenerator, "qwen": QwenGenerator, "deepseek": DeepSeekGenerator}
    return adapters[model.provider](model, transport=transport)


__all__ = ["Generator", "ProviderError", "make_generator"]
