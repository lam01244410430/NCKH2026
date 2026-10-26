"""Shared neutral instructions and canonical message hashing."""
import hashlib
import json

from ..schemas import ChatMessage, PromptRecord

SYSTEM_MESSAGE = (
    "你是中文学习材料编写助手。遵守指定的HSK等级和任务约束，内容自然、独立完整且事实合理。"
    "只输出符合给定JSON Schema的严格JSON，不输出Markdown、额外说明或推理过程。"
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_messages(messages: list[ChatMessage]) -> str:
    """SHA-256 of canonical UTF-8 JSON, including roles and message order."""
    return hashlib.sha256(canonical_json([m.model_dump() for m in messages]).encode("utf-8")).hexdigest()


def task_messages(record: PromptRecord, task: str, instructions: str, schema: dict) -> list[ChatMessage]:
    if record.task != task:
        raise ValueError(f"expected task {task}, got {record.task}")
    return [ChatMessage(role="system", content=SYSTEM_MESSAGE), ChatMessage(
        role="user", content=(
            f"提示编号：{record.prompt_id}\n模板版本：{record.prompt_version}\n任务：{record.task}\n"
            f"目标等级：HSK {record.hsk_level}\n主题：{record.topic}\n"
            f"任务约束：{canonical_json(record.payload.model_dump())}\n{instructions}\n"
            f"输出JSON Schema：{canonical_json(schema)}"
        ),
    )]
