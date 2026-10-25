"""OpenAI Chat Completions adapter; flags must match the frozen model ID."""
from .openai_compatible import OpenAICompatibleGenerator


class OpenAIGenerator(OpenAICompatibleGenerator):
    provider = "openai"
    default_base_url = "https://api.openai.com/v1"
    token_limit_parameter = "max_completion_tokens"
