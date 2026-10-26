import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from typer.testing import CliRunner

from hsk_agent_bench.cli import app
from hsk_agent_bench.evaluators.materials import project_material_text
from hsk_agent_bench.evaluators.lexical import HSKLexicon
from hsk_agent_bench.evaluators.pipeline import EvaluationConfig, evaluate_shard
from hsk_agent_bench.io import write_jsonl_shard_atomic
from hsk_agent_bench.schemas import ChatMessage, GenerationParameters, GenerationRecord, MetricRecord, TokenUsage


def generated(task, text, *, generation_id="g1", level=3):
    return GenerationRecord(
        generation_id=generation_id, prompt_id=f"p-{task}", task=task, hsk_level=level,
        family="GPT", origin="global", provider="openai", model_id="frozen-test-model",
        prompt_hash="0" * 64, messages=[ChatMessage(role="user", content="generate")],
        parameters=GenerationParameters(temperature=0, top_p=1, max_output_tokens=100),
        run_id="primary", replicate=0, text=json.dumps(text, ensure_ascii=False),
        raw_response={"provider_payload": text}, requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        latency_ms=0, usage=TokenUsage(), retry_count=0, seed_applied=False,
    )


READING = {
    "title": "学习", "passage": "我学习。", "declared_hsk_level": 3,
    "used_required_vocabulary": ["学习"], "used_required_grammar": [],
}
GRAMMAR = {
    "explanation": "因为表示原因。", "form": "因为A，所以B",
    "usage_constraints": ["说明原因"], "examples": ["因为下雨，所以我在家。"],
    "incorrect_example": {"incorrect": "因为下雨，但是我在家。", "correction": "因为下雨，所以我在家。"},
    "practice": {"question": "填空：___下雨，所以我在家。", "answer": "因为"},
}
EXAM = {
    "source_passage": "我去学校。", "items": [{
        "question": "我去哪儿？", "options": ["学校", "医院", "商店", "公园"],
        "answer": "学校", "rationale": "原文说明去学校。",
        "target_vocabulary": ["学校"], "target_grammar": [],
    }],
}


@pytest.mark.parametrize(("task", "payload", "expected"), [
    ("reading", READING, "学习\n我学习。"),
    ("grammar_explanation", GRAMMAR, "因为表示原因。"),
    ("exam_generation", EXAM, "我去哪儿？"),
])
def test_task_outputs_are_strictly_validated_and_projected(task, payload, expected):
    projected = project_material_text(task, json.dumps(payload, ensure_ascii=False), hsk_level=3)
    assert expected in projected
    assert '"source_passage"' not in projected
    assert '"explanation"' not in projected


def test_malformed_or_ambiguous_task_output_is_rejected_without_mutating_raw_record():
    payload = json.loads(json.dumps(EXAM, ensure_ascii=False))
    payload["items"][0]["options"][2] = "医院"
    record = generated("exam_generation", payload)
    before = record.model_dump_json()
    with pytest.raises(ValueError, match="duplicated options"):
        project_material_text(record.task, record.text, hsk_level=record.hsk_level)
    assert record.model_dump_json() == before
    with pytest.raises(ValueError, match="strict JSON"):
        project_material_text("reading", "```json\n{}\n```", hsk_level=3)
    wrong_level = dict(READING, declared_hsk_level=4)
    with pytest.raises(ValueError, match="declared_hsk_level"):
        project_material_text("reading", json.dumps(wrong_level, ensure_ascii=False), hsk_level=3)


def test_evaluation_scores_projected_material_not_raw_json_keys(tmp_path):
    record = generated("reading", READING)
    raw, output = tmp_path / "raw.jsonl", tmp_path / "metrics.jsonl"
    write_jsonl_shard_atomic([record], raw)
    evaluate_shard(raw, output, EvaluationConfig(lexicon=HSKLexicon({"学习": 2, "我": 1})))
    metric = MetricRecord.model_validate_json(output.read_text(encoding="utf-8").strip())
    assert metric.metrics["unknown_token_rate"] == 0
    assert metric.metrics["mean_dependency_distance"] is None
    assert "parser_error" in metric.metadata["syntax"]
    assert metric.metadata["material"]["schema"] == "ReadingOutput"
    assert record.raw_response == {"provider_payload": READING}


def test_rating_preparation_projects_all_tasks_and_rejects_malformed_outputs(tmp_path):
    records = [generated("reading", READING, generation_id="g-reading"),
               generated("grammar_explanation", GRAMMAR, generation_id="g-grammar"),
               generated("exam_generation", EXAM, generation_id="g-exam")]
    source = tmp_path / "records.csv"
    pd.DataFrame([record.model_dump(mode="json") for record in records]).to_csv(source, index=False)
    output, key = tmp_path / "rating.csv", tmp_path / "key.csv"
    result = CliRunner().invoke(app, [
        "prepare-ratings", "--input", str(source), "--output", str(output),
        "--key-output", str(key), "--device", "configs/devices/pc.yaml", "--fraction", "1",
    ])
    assert result.exit_code == 0, result.output
    sheet = pd.read_csv(output)
    assert len(sheet) == 3
    assert all(not text.lstrip().startswith("{") for text in sheet.text)
    assert any("我去哪儿？" in text for text in sheet.text)

    bad = records + [generated("exam_generation", {**EXAM, "items": []}, generation_id="bad")]
    pd.DataFrame([record.model_dump(mode="json") for record in bad]).to_csv(tmp_path / "bad.csv", index=False)
    failed = CliRunner().invoke(app, [
        "prepare-ratings", "--input", str(tmp_path / "bad.csv"), "--output", str(tmp_path / "bad-rating.csv"),
        "--key-output", str(tmp_path / "bad-key.csv"), "--device", "configs/devices/pc.yaml", "--fraction", "1",
    ])
    assert failed.exit_code != 0
    assert not (tmp_path / "bad-rating.csv").exists()
