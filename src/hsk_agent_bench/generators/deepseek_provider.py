"""DeepSeek Chat Completions adapter; its documented API has no seed field."""
from .openai_compatible import OpenAICompatibleGenerator


class DeepSeekGenerator(OpenAICompatibleGenerator):
    provider = "deepseek"
    default_base_url = "https://api.deepseek.com"
    supports_seed = False
