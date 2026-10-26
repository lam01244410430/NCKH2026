import pytest

from hsk_agent_bench.evaluators.grammar import (
    GrammarCatalog, GrammarFlag, candidate_grammar_flags,
    confirmed_violation_rate, score_grammar_alignment,
)


def test_catalog_counts_occurrences_and_levels_separately():
    catalog = GrammarCatalog([
        {"grammar_id": "because", "level": 3, "pattern": "因为"},
        {"grammar_id": "although", "level": 5, "pattern": "虽然"},
    ])
    result = score_grammar_alignment("因为下雨，因为很冷，虽然我想去。", 3, catalog)
    assert result.in_level_matches == 2
    assert result.target_level_matches == 2
    assert result.above_level_matches == 1
    assert [(m.start, m.end, m.text) for m in result.matches] == [
        (0, 2, "因为"), (5, 7, "因为"), (10, 12, "虽然")]
    assert result.metadata["catalog_sha256"]


class Detector:
    def detect(self, text):
        return [GrammarFlag(flag_id="a", start=0, end=1, reason="dependency anomaly",
                            source="dependency")]


def test_dependency_candidates_require_human_confirmation_and_export_spans():
    flags = candidate_grammar_flags("我学习。", Detector())
    assert flags[0].model_dump() == {
        "flag_id": "a", "start": 0, "end": 1, "reason": "dependency anomaly",
        "source": "dependency", "status": "candidate"}
    assert confirmed_violation_rate(flags, {}, 4) == 0
    assert confirmed_violation_rate(flags, {"a": False}, 4) == 0
    assert confirmed_violation_rate(flags, {"a": True}, 4) == 25
    assert confirmed_violation_rate(flags + flags, {"a": True}, 4) == 25


def test_rate_uses_chinese_token_count_per_hundred():
    flags = candidate_grammar_flags("我学习。", Detector())
    assert confirmed_violation_rate(flags, {"a": True}, 200) == 0.5
    for count in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            confirmed_violation_rate(flags, {}, count)
    with pytest.raises(ValueError):
        confirmed_violation_rate(flags, {"a": 1}, 4)


def test_detector_invalid_spans_and_conflicting_ids_are_rejected():
    with pytest.raises(ValueError):
        candidate_grammar_flags("", Detector())
    flags = candidate_grammar_flags("我学习。", Detector())
    other = flags[0].model_copy(update={"end": 2})
    with pytest.raises(ValueError):
        confirmed_violation_rate(flags + [other], {"a": True}, 4)


@pytest.mark.parametrize("entries", [
    [{"grammar_id": "x", "level": 3, "pattern": "["}],
    [{"grammar_id": "x", "level": 3, "pattern": "a*"}],
    [{"grammar_id": "x", "level": 0, "pattern": "我"}],
    [{"grammar_id": "x", "level": 3, "pattern": "我"}] * 2,
])
def test_catalog_rejects_invalid_rules(entries):
    with pytest.raises(ValueError):
        GrammarCatalog(entries)
