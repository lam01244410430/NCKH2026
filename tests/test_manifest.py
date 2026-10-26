from collections import Counter
from pathlib import Path

import pytest

from hsk_agent_bench.config import load_experiment_config
from hsk_agent_bench.prompts.common import hash_messages
from hsk_agent_bench.prompts.exam_generation import ExamOutput, validate_exam_payload
from hsk_agent_bench.prompts.manifest import build_manifest
from hsk_agent_bench.prompts.reading import render_reading_prompt
from hsk_agent_bench.prompts.grammar_explanation import render_grammar_prompt
from hsk_agent_bench.prompts.exam_generation import render_exam_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_balanced_deterministic_and_hashes_actual_messages():
    config = load_experiment_config(ROOT / "configs/experiment.yaml")
    records = build_manifest(config)
    assert records == build_manifest(config)
    assert len(records) == len({r.prompt_id for r in records}) == len({r.prompt_hash for r in records}) == 144
    assert Counter(r.task for r in records) == {"reading": 48, "grammar_explanation": 48, "exam_generation": 48}
    assert Counter(r.hsk_level for r in records) == {3: 36, 4: 36, 5: 36, 6: 36}
    assert [(r.task, r.hsk_level, r.topic) for r in records] == [
        (task, level, topic) for task in config.tasks for level in config.levels for topic in config.topics
    ]
    renderers = {"reading": render_reading_prompt, "grammar_explanation": render_grammar_prompt,
                 "exam_generation": render_exam_prompt}
    for record in records:
        assert record.prompt_hash == hash_messages(renderers[record.task](record))
    assert all(not r.payload.required_vocab and not r.payload.required_grammar
               for r in records if r.task == "reading")


def test_manifest_respects_configured_subset_and_preserves_cell_ids():
    config = load_experiment_config(ROOT / "configs/experiment.yaml")
    all_records = build_manifest(config)
    config.tasks = ["reading"]
    config.levels = [4]
    config.topics = ["health"]
    subset = build_manifest(config)
    assert subset == [r for r in all_records if r.task == "reading" and r.hsk_level == 4 and r.topic == "health"]


def exam_data():
    return {"source_passage": "小明今天去学校。", "items": [
        {"question": "小明去哪儿？", "options": ["学校", "医院", "商店", "公园"],
         "answer": "学校", "rationale": "材料说去学校。", "target_vocabulary": [], "target_grammar": []}
    ]}


def test_valid_exam_has_unique_answer():
    assert validate_exam_payload(ExamOutput.model_validate(exam_data())) is None


@pytest.mark.parametrize("problem", ["duplicate", "missing", "multiple", "unmatched", "wrong_count"])
def test_invalid_exam_is_rejected(problem):
    data = exam_data()
    item = data["items"][0]
    if problem == "duplicate":
        item["options"][2] = "医院"
    elif problem == "multiple":
        item["options"][2] = " 学校 "
    elif problem == "missing":
        del item["answer"]
    elif problem == "unmatched":
        item["answer"] = "北京"
    else:
        item["options"].pop()
    with pytest.raises(ValueError):
        validate_exam_payload(ExamOutput.model_validate(data))
