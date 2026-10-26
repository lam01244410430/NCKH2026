"""DashScope OpenAI-compatible endpoint; override base_url for other regions."""
from .openai_compatible import OpenAICompatibleGenerator


class QwenGenerator(OpenAICompatibleGenerator):
    provider = "qwen"
    default_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
