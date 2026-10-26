"""Strict task-output parsing and deterministic learner-facing projection."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from ..prompts.exam_generation import ExamOutput, validate_exam_payload
from ..prompts.grammar_explanation import GrammarOutput
from ..prompts.reading import ReadingOutput
from ..schemas import TaskType


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_material_output(task: TaskType | str, raw_text: str, *,
                          hsk_level: int | None = None) -> BaseModel:
    """Parse exactly one strict JSON object and apply task-specific validation."""
    try:
        data = json.loads(raw_text, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError(f"non-JSON constant: {value}")))
        task = TaskType(task)
        if task == TaskType.READING:
            parsed = ReadingOutput.model_validate(data)
            if hsk_level is not None and parsed.declared_hsk_level != hsk_level:
                raise ValueError("declared_hsk_level must match the requested HSK level")
        elif task == TaskType.GRAMMAR_EXPLANATION:
            parsed = GrammarOutput.model_validate(data)
        else:
            parsed = ExamOutput.model_validate(data)
            validate_exam_payload(parsed)
        return parsed
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"material output must be strict JSON matching {task}: {exc}") from exc


def _project(parsed: BaseModel) -> str:
    if isinstance(parsed, ReadingOutput):
        return f"{parsed.title}\n{parsed.passage}"
    if isinstance(parsed, GrammarOutput):
        lines = [parsed.explanation, f"形式：{parsed.form}"]
        lines.extend(f"使用限制：{item}" for item in parsed.usage_constraints)
        lines.extend(f"例句：{item}" for item in parsed.examples)
        lines.extend((f"错误例句：{parsed.incorrect_example.incorrect}",
                      f"改正：{parsed.incorrect_example.correction}",
                      f"练习：{parsed.practice.question}", f"答案：{parsed.practice.answer}"))
        return "\n".join(lines)
    if isinstance(parsed, ExamOutput):
        lines = [parsed.source_passage]
        for index, item in enumerate(parsed.items, 1):
            lines.append(f"第{index}题：{item.question}")
            lines.extend(f"{label}. {option}" for label, option in zip("ABCD", item.options))
            lines.extend((f"答案：{item.answer}", f"依据：{item.rationale}"))
        return "\n".join(lines)
    raise TypeError("unsupported material output")


def project_material_text(task: TaskType | str, raw_text: str, *,
                          hsk_level: int | None = None) -> str:
    """Return validated learner-facing text without scoring JSON syntax or keys."""
    return _project(parse_material_output(task, raw_text, hsk_level=hsk_level))


def material_metadata(task: TaskType | str, raw_text: str, projected: str) -> dict[str, str]:
    parsed = parse_material_output(task, raw_text)
    return {
        "schema": type(parsed).__name__,
        "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "projected_text_sha256": hashlib.sha256(projected.encode("utf-8")).hexdigest(),
        "projection_version": "1",
    }
