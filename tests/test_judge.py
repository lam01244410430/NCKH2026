import json

import pandas as pd
import pytest

from hsk_agent_bench.config import ModelConfig
from hsk_agent_bench.evaluators.judge import (
    JudgeResponseError, JudgeRubric, JudgeValidationThreshold, NormalizedOutput,
    parse_judge_response, render_judge_request, validate_judge_against_human,
)
from hsk_agent_bench.prompts.common import hash_messages
from hsk_agent_bench.schemas import GenerationRecord, RatingRecord


def source_record():
    material = {"title": "学习中文", "passage": "小明今天去学校学习中文。", "declared_hsk_level": 4,
                "used_required_vocabulary": [], "used_required_grammar": []}
    return GenerationRecord(
        generation_id="source-g", prompt_id="source-p", task="reading", hsk_level=4,
        family="SECRET-SOURCE", provider="qwen", origin="china", model_id="secret-source-model",
        prompt_hash="a" * 64, messages=[dict(role="user", content="secret source config")],
        parameters=dict(temperature=0, top_p=1, max_output_tokens=500),
        text=json.dumps(material, ensure_ascii=False), raw_response={"private": "secret"},
        requested_at="2026-10-26T00:00:00Z", latency_ms=1, seed_applied=False,
    )


def rubric():
    return JudgeRubric(judge_model=ModelConfig(
        family="judge-family", provider="openai", origin="global", model_id="judge-frozen",
        api_key_env="JUDGE_API_KEY"))


def response(text):
    request = render_judge_request(NormalizedOutput.from_generation(source_record()), rubric())
    return GenerationRecord(**request.model_dump(), text=text,
                            raw_response={"id": "raw-judge", "answer": text},
                            requested_at="2026-10-26T00:00:00Z", latency_ms=2, seed_applied=False)


def valid_payload():
    dimensions = ["naturalness", "pedagogical_quality", "cultural_appropriateness", "reading_coherence"]
    return {"scores": dict.fromkeys(dimensions, 4), "evidence": dict.fromkeys(dimensions, "学习中文")}


def test_request_uses_generation_contract_but_source_identity_is_not_in_messages():
    output = NormalizedOutput.from_generation(source_record())
    request = render_judge_request(output, rubric())
    rendered = " ".join(message.content for message in request.messages)
    assert request.model_id == "judge-frozen"
    assert request.prompt_hash == hash_messages(request.messages)
    assert "secret" not in rendered.lower()
    assert "SECRET-SOURCE" not in rendered
    assert "qwen" not in rendered
    assert "source-g" not in rendered
    assert "source-p" not in rendered
    payload = json.loads(request.messages[1].content)
    assert payload["task"] == "reading"
    assert payload["hsk_level"] == 4
    assert payload["output"] == "学习中文\n小明今天去学校学习中文。"
    assert set(payload["rubric"]["naturalness"]) == {"1", "2", "3", "4", "5"}
    assert "reading_coherence" in payload["rubric"]
    assert "explanation_accuracy" not in payload["rubric"]


def test_distinct_judge_runs_cannot_collide_in_generation_storage():
    output = NormalizedOutput.from_generation(source_record())
    config = rubric()
    original = render_judge_request(output, config)
    assert original.generation_id == render_judge_request(output, config).generation_id
    config.judge_model.model_id = "another-judge"
    assert render_judge_request(output, config).generation_id != original.generation_id
    config = rubric()
    config.rubric_version = "2"
    assert render_judge_request(output, config).generation_id != original.generation_id
    config = rubric()
    config.parameters.seed = 55
    assert render_judge_request(output, config).generation_id != original.generation_id


def test_response_for_another_item_is_rejected_with_raw_record():
    raw = response(json.dumps(valid_payload()))
    output = NormalizedOutput.from_generation(source_record())
    output.generation_id = "another-item"
    with pytest.raises(JudgeResponseError, match="match"):
        parse_judge_response(raw, output, rubric())


@pytest.mark.parametrize("blank", ["", "   \n\t"])
def test_blank_output_can_be_scored_with_one_consistent_marker(blank):
    output = NormalizedOutput.from_generation(source_record())
    output.text = blank
    request = render_judge_request(output, rubric())
    payload = valid_payload()
    payload["evidence"] = dict.fromkeys(payload["scores"], "[blank output]")
    raw = GenerationRecord(**request.model_dump(), text=json.dumps(payload), raw_response={"answer": payload},
                           requested_at="2026-10-26T00:00:00Z", latency_ms=2, seed_applied=False)
    assert parse_judge_response(raw, output, rubric()).rating.evidence["naturalness"] == "[blank output]"
    assert json.loads(request.messages[1].content)["output"] == blank


@pytest.mark.parametrize("task,dimension", [("grammar_explanation", "explanation_accuracy"),
                                            ("exam_generation", "item_validity")])
def test_judge_rubric_contains_task_specific_anchors(task, dimension):
    output = NormalizedOutput(generation_id="g", prompt_id="p", task=task, hsk_level=3, text="中文")
    payload = json.loads(render_judge_request(output, rubric()).messages[1].content)
    assert set(payload["rubric"][dimension]) == {"1", "2", "3", "4", "5"}


def test_parsing_preserves_raw_response_and_emits_shared_rating_record():
    raw = response(json.dumps(valid_payload(), ensure_ascii=False))
    parsed = parse_judge_response(raw, NormalizedOutput.from_generation(source_record()), rubric())
    assert isinstance(parsed.rating, RatingRecord)
    assert parsed.rating.generation_id == "source-g"
    assert parsed.rating.prompt_id == "source-p"
    assert parsed.rating.rater_type == "llm"
    assert parsed.rating.scores["naturalness"] == 4
    assert parsed.rating.evidence["naturalness"] == "学习中文"
    assert parsed.response.raw_response == raw.raw_response


@pytest.mark.parametrize("text", [
    '```json\n{}\n```', '{} trailing', '[]',
    '{"scores":{},"scores":{},"evidence":{}}',
    '{"scores":{"naturalness":NaN},"evidence":{}}',
])
def test_non_strict_json_is_rejected_without_losing_raw_response(text):
    raw = response(text)
    with pytest.raises(JudgeResponseError) as error:
        parse_judge_response(raw, NormalizedOutput.from_generation(source_record()), rubric())
    assert error.value.response.raw_response == raw.raw_response


@pytest.mark.parametrize("mutation", ["extra", "missing", "boolean", "string", "range", "evidence", "long"])
def test_judge_rejects_invalid_ratings_and_evidence(mutation):
    payload = valid_payload()
    if mutation == "extra":
        payload["explanation"] = "not allowed"
    elif mutation == "missing":
        del payload["scores"]["reading_coherence"]
    elif mutation in {"boolean", "string", "range"}:
        payload["scores"]["naturalness"] = {"boolean": True, "string": "4", "range": 6}[mutation]
    elif mutation == "evidence":
        payload["evidence"]["naturalness"] = "invented span"
    else:
        payload["evidence"]["naturalness"] = "中" * 241
    with pytest.raises(JudgeResponseError):
        parse_judge_response(response(json.dumps(payload)), NormalizedOutput.from_generation(source_record()), rubric())


def rating_frame(values, ids=None, dimension="naturalness"):
    ids = ids if ids is not None else [f"g{i}" for i in range(len(values))]
    return pd.DataFrame({"generation_id": ids, "scores": [{dimension: score} for score in values]})


def test_validation_metrics_and_configurable_preregistered_threshold():
    human = rating_frame([1, 2, 3, 4, 5])
    judge = rating_frame([1, 3, 2, 5, 4])
    threshold = JudgeValidationThreshold(minimum_pairs=5, spearman=.7, weighted_kappa=.7,
                                        exact_agreement=.2, adjacent_agreement=1)
    report = validate_judge_against_human(judge, human, threshold=threshold)
    metric = report.by_dimension["naturalness"]
    assert metric.spearman == pytest.approx(.8)
    assert metric.weighted_kappa == pytest.approx(.8)
    assert metric.exact_agreement == .2
    assert metric.adjacent_agreement == 1
    assert report.validated is True
    assert validate_judge_against_human(judge, human).validated is False
    assert validate_judge_against_human(judge, human, threshold=threshold.model_copy(
        update={"spearman": .9})).validated is False


def test_validation_pairs_by_id_and_dimension_and_reports_missing():
    judge = rating_frame([5, 3, None, 1], ["e", "c", "b", "a"])
    human = rating_frame([1, 2, 3, 4, 5], list("abcde"))
    report = validate_judge_against_human(judge, human)
    metric = report.by_dimension["naturalness"]
    assert metric.n_pairs == 3
    assert metric.n_missing == 2
    assert metric.spearman == pytest.approx(1)


def test_validation_does_not_pool_dimensions_or_accept_undefined_agreement():
    judge = rating_frame([1, 2, 3, 4, 5])
    human = rating_frame([1, 2, 3, 4, 5])
    judge["scores"] = [{**score, "cultural_appropriateness": 3} for score in judge.scores]
    human["scores"] = [{**score, "cultural_appropriateness": 3} for score in human.scores]
    report = validate_judge_against_human(judge, human, threshold=JudgeValidationThreshold(minimum_pairs=2))
    assert report.by_dimension["naturalness"].spearman == pytest.approx(1)
    assert report.by_dimension["cultural_appropriateness"].spearman is None
    assert report.validated is False


def test_ambiguous_duplicate_ratings_require_explicit_aggregation():
    with pytest.raises(ValueError, match="duplicate"):
        validate_judge_against_human(rating_frame([1, 2], ["g", "g"]), rating_frame([1]))


def test_long_rating_tables_report_absent_dimensions_and_empty_pairs():
    human = pd.DataFrame({"generation_id": ["a", "b"], "dimension": ["naturalness", "naturalness"],
                          "score": [1, 5]})
    judge = pd.DataFrame({"generation_id": ["a", "b"], "dimension": ["item_validity", "item_validity"],
                          "score": [1, 5]})
    report = validate_judge_against_human(judge, human, threshold=JudgeValidationThreshold(minimum_pairs=2))
    assert report.validated is False
    assert report.by_dimension["naturalness"].n_pairs == 0
    assert report.by_dimension["naturalness"].n_missing == 2
    assert report.by_dimension["item_validity"].n_pairs == 0


def test_empty_rating_tables_are_not_validated():
    empty = pd.DataFrame(columns=["generation_id", "scores"])
    report = validate_judge_against_human(empty, empty, threshold=JudgeValidationThreshold())
    assert report.by_dimension == {}
    assert report.validated is False
