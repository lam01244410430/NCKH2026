import json
from pathlib import Path

import pytest

from hsk_agent_bench.evaluators.grammar import (
    GrammarCatalog, GrammarFlag, GrammarRule, candidate_grammar_flags,
    confirmed_violation_rate, score_grammar_alignment,
)


def test_catalog_snapshots_rules_and_rejects_mutation():
    rule = GrammarRule(grammar_id="because", level=3, pattern="因为")
    catalog = GrammarCatalog([rule])
    digest = catalog.sha256
    assert catalog.rules[0] is not rule
    with pytest.raises(ValueError):
        catalog.rules[0].level = 5
    with pytest.raises(ValueError):
        rule.pattern = "虽然"
    with pytest.raises(AttributeError):
        catalog.rules = ()
    result = score_grammar_alignment("因为", 3, catalog)
    assert result.target_level_matches == 1
    assert result.metadata["catalog_sha256"] == digest


def test_canonical_resource_projection_preserves_provenance_and_snapshot():
    entries = json.loads((Path(__file__).parents[1] /
                         "data/hsk_grammar_points/schema.example.json").read_text(encoding="utf-8"))
    catalog = GrammarCatalog.from_canonical(entries)
    rule = catalog.rules[0]
    assert rule.level == 3
    assert rule.provenance.source == "Replace with a verified licensed source citation"
    assert rule.provenance.standard == "illustrative-only-not-an-official-mapping"
    assert rule.provenance.license == "Replace with redistribution terms"
    assert rule.provenance.examples == ("此处填写经过验证的例句。",)
    digest = catalog.sha256
    entries[0]["examples"].append("changed")
    entries[0]["min_hsk_level"] = 6
    assert score_grammar_alignment("待替换的语法点", 3, catalog).target_level_matches == 1
    assert catalog.sha256 == digest
    assert GrammarCatalog.from_canonical(entries).sha256 != digest


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
