"""Chinese reading-passage template."""
from ..schemas import ChatMessage, HSKLevel, NonEmpty, PromptRecord, Record
from .common import task_messages


class ReadingOutput(Record):
    title: NonEmpty
    passage: NonEmpty
    declared_hsk_level: HSKLevel
    used_required_vocabulary: list[str]
    used_required_grammar: list[str]


def render_reading_prompt(record: PromptRecord) -> list[ChatMessage]:
    return task_messages(record, "reading", (
        "写一篇围绕主题的中文短文。正文汉字数以target_length_chars为目标；"
        "使用required_vocab中的词和required_grammar中的语法，超纲词数不得超过max_out_of_level_words。"
        "如实填写实际使用的指定词汇和语法，declared_hsk_level填写目标等级。"
    ), ReadingOutput.model_json_schema())
