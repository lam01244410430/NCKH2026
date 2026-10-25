import pytest
from pydantic import ValidationError
from hsk_agent_bench.schemas import GenerationRequest, GenerationRecord, MetricRecord, RatingRecord


def request_data():
    return dict(generation_id="g1", prompt_id="p1", task="reading", hsk_level=3,
                model_id="frozen-model", provider="openai", family="GPT", origin="global",
                prompt_hash="a" * 64, messages=[dict(role="user", content="你好")],
                parameters=dict(temperature=0, top_p=1, max_output_tokens=100, seed=20261025))


def test_generation_request_preserves_sampling_and_identity():
    request = GenerationRequest.model_validate(request_data())
    assert request.parameters.seed == 20261025
    assert GenerationRequest.model_validate_json(request.model_dump_json()) == request


def test_generation_record_preserves_audit_metadata():
    data = request_data()
    data.update(text="你好", raw_response={"answer": "你好"}, request_id="r1",
                requested_at="2026-10-25T12:00:00Z", latency_ms=12,
                usage=dict(input_tokens=10, output_tokens=2), finish_reason="stop", seed_applied=False)
    record = GenerationRecord.model_validate(data)
    assert record.raw_response == {"answer": "你好"}
    assert record.usage.output_tokens == 2
    assert record.seed_applied is False


def test_metric_record_keeps_parser_provenance():
    record = MetricRecord(generation_id="g1", prompt_id="p1", metrics={"mdd": 1.5},
                          evaluator="syntax", evaluator_version="v1", device="cpu")
    assert record.metrics["mdd"] == 1.5


@pytest.mark.parametrize("score", [0, 6, 1.5, True])
def test_ratings_reject_nonordinal_scores(score):
    with pytest.raises(ValidationError):
        RatingRecord(generation_id="g1", prompt_id="p1", rater_id="anonymous",
                     rater_type="human", scores={"naturalness": score})
