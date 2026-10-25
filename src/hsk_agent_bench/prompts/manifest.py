"""Deterministic grid expansion; no filesystem writes or API calls.

Defaults are development fixtures, not official HSK mappings or study-ready
materials. Replace placeholder grammar with licensed, expert-validated data
and freeze lengths and template version before collecting study outputs.
"""
from urllib.parse import quote

from ..config import ExperimentConfig
from ..schemas import ExamPayload, GrammarPayload, PromptRecord, ReadingPayload, TaskType
from .common import hash_messages
from .exam_generation import render_exam_prompt
from .grammar_explanation import render_grammar_prompt
from .reading import render_reading_prompt


def build_manifest(config: ExperimentConfig) -> list[PromptRecord]:
    """Expand tasks, levels, topics in configuration order, with stable cell IDs.

    The shipped configuration yields 144 records; subsets remain usable for
    dry runs. Vocabulary/grammar requirements stay empty until sources exist.
    """
    renderers = {
        TaskType.READING: render_reading_prompt,
        TaskType.GRAMMAR_EXPLANATION: render_grammar_prompt,
        TaskType.EXAM_GENERATION: render_exam_prompt,
    }
    records = []
    for task in config.tasks:
        for level in config.levels:
            for topic in config.topics:
                if task == TaskType.READING:
                    payload = ReadingPayload(target_length_chars=300)
                elif task == TaskType.GRAMMAR_EXPLANATION:
                    payload = GrammarPayload(
                        grammar_point="待替换的语法点（开发占位，未经等级验证）", context=topic,
                    )
                else:
                    payload = ExamPayload(source_length=300, num_items=1)
                record = PromptRecord(
                    prompt_id=f"{task}-hsk{level}-{quote(topic, safe='')}",
                    task=task, hsk_level=level, topic=topic, payload=payload,
                    prompt_version="dev-1",
                )
                record.prompt_hash = hash_messages(renderers[task](record))
                records.append(record)
    return records
