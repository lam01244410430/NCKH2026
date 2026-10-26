import json
from pathlib import Path

import pytest

from hsk_agent_bench.evaluators.syntax import (
    LazyParser, ParsedDocument, mean_dependency_distance, mean_sentence_length,
    score_syntax,
)
from hsk_agent_bench.evaluators.tokenization import tokenize_chinese


def document():
    return ParsedDocument.model_validate(json.loads(
        (Path(__file__).parent / "fixtures/dependencies.json").read_text(encoding="utf-8")))


def test_sentence_length_splits_terminal_punctuation_and_ignores_empty_sentences():
    assert mean_sentence_length("我，你！他？我。；", list) == 4 / 3
    assert mean_sentence_length("我。\n你！", tokenize_chinese) == 1
    assert mean_sentence_length("。！？", tokenize_chinese) == 0


def test_dependency_distance_uses_all_nonroot_arcs():
    assert mean_dependency_distance(document()) == 1.25
    assert mean_dependency_distance(ParsedDocument(sentences=[])) == 0


def test_invalid_dependency_heads_are_rejected():
    with pytest.raises(ValueError):
        ParsedDocument.model_validate({"sentences": [{"tokens": [
            {"index": 1, "text": "我", "head": 2, "relation": "obj"}]}]})


class Backend:
    def parse(self, text):
        return document()


def test_lazy_cpu_fallback_and_metric_provenance():
    calls = []
    def load(device):
        calls.append(device)
        assert device == "cpu"
        return Backend()

    parser = LazyParser(name="fixture", version="1.2", loader=load,
                        device="cuda", gpu_available=lambda: False)
    assert calls == []
    result = score_syntax("我。", list, parser, generation_id="g", prompt_id="p")
    assert result.metrics == {"mean_sentence_length": 1, "mean_dependency_distance": 1.25}
    assert result.device == "cpu"
    assert result.metadata["parser_name"] == "fixture"
    assert result.metadata["parser_version"] == "1.2"
    assert result.metadata["parser_device"] == "cpu"
    assert result == score_syntax("我。", list, parser, generation_id="g", prompt_id="p")
    assert calls == ["cpu"]


def test_gpu_load_failure_falls_back_to_cpu():
    def load(device):
        if device == "cuda":
            raise RuntimeError("GPU unavailable")
        return Backend()

    parser = LazyParser(name="fixture", version="1", loader=load,
                        device="cuda", gpu_available=lambda: True)
    assert mean_dependency_distance(parser.parse("我")) == 1.25
    assert parser.device == "cpu"


def test_parser_failure_preserves_missing_metric_and_error():
    def load(device):
        raise RuntimeError("model not installed")

    parser = LazyParser(name="fixture", version="1", loader=load)
    result = score_syntax("我。", list, parser, generation_id="g", prompt_id="p")
    assert result.metrics["mean_dependency_distance"] is None
    assert result.metrics["mean_sentence_length"] == 1
    assert "model not installed" in result.metadata["parser_error"]


def test_gpu_parse_failure_retries_cpu_and_retries_failed_cpu_load_safely():
    class BrokenGPU:
        def parse(self, text):
            raise RuntimeError("GPU lost")

    cpu_loads = []
    def load(device):
        if device == "cuda":
            return BrokenGPU()
        cpu_loads.append(device)
        if len(cpu_loads) == 1:
            raise RuntimeError("temporary CPU load failure")
        return Backend()

    parser = LazyParser(name="fixture", version="1", loader=load,
                        device="cuda", gpu_available=lambda: True)
    first = score_syntax("我。", list, parser, generation_id="g", prompt_id="p")
    assert first.metrics["mean_dependency_distance"] is None
    second = score_syntax("我。", list, parser, generation_id="g", prompt_id="p")
    assert second.metrics["mean_dependency_distance"] == 1.25
    assert second.device == "cpu"
