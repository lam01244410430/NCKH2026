import hashlib
import json

import pytest

from hsk_agent_bench.schemas import PromptRecord
from hsk_agent_bench.prompts.common import hash_messages
from hsk_agent_bench.prompts.reading import render_reading_prompt
from hsk_agent_bench.prompts.grammar_explanation import render_grammar_prompt
from hsk_agent_bench.prompts.exam_generation import render_exam_prompt


@pytest.mark.parametrize("task,payload,render,fields", [
    ("reading", {"target_length_chars": 350, "required_vocab": ["学习"],
                 "required_grammar": ["因为……所以……"], "max_out_of_level_words": 2},
     render_reading_prompt, ["title", "passage", "declared_hsk_level", "used_required_vocabulary", "used_required_grammar"]),
    ("grammar_explanation", {"grammar_point": "把字句", "learner_l1": "English",
                             "context": "整理房间", "example_count": 4},
     render_grammar_prompt, ["explanation", "form", "usage_constraints", "examples", "incorrect_example", "practice"]),
    ("exam_generation", {"item_type": "multiple_choice", "source_length": 380, "num_items": 2},
     render_exam_prompt, ["source_passage", "items", "question", "options", "answer", "rationale", "target_vocabulary", "target_grammar"]),
])
def test_prompts_preserve_constraints_and_have_reproducible_neutral_content(task, payload, render, fields):
    record = PromptRecord(prompt_id=f"{task}-4-health", task=task, hsk_level=4,
                          topic="health", payload={"task": task, **payload})
    messages = render(record)
    assert [message.role for message in messages] == ["system", "user"]
    assert messages == render(record)
    content = "\n".join(message.content for message in messages)
    assert "HSK 4" in content and "health" in content and record.prompt_id in content
    for value in payload.values():
        for part in value if isinstance(value, list) else [value]:
            assert str(part) in content
    assert all(field in content for field in fields)
    assert "JSON" in content and "推理" in content
    assert all(name not in content.lower() for name in ("openai", "gemini", "qwen", "deepseek", "gpt", "google"))
    canonical = json.dumps([m.model_dump() for m in messages], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hash_messages(messages) == hashlib.sha256(canonical).hexdigest()
    assert hash_messages(messages) == hash_messages(render(record))
    changed = record.model_copy(update={"topic": "education"})
    assert hash_messages(render(changed)) != hash_messages(messages)


def test_renderer_rejects_wrong_task():
    record = PromptRecord(prompt_id="reading-3", task="reading", hsk_level=3,
                          topic="travel", payload={"task": "reading", "target_length_chars": 300})
    with pytest.raises(ValueError, match="task"):
        render_exam_prompt(record)
