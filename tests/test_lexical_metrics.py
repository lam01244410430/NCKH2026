import json
from pathlib import Path

import pytest
import jieba

from hsk_agent_bench.evaluators.lexical import HSKLexicon, score_lexical_alignment
from hsk_agent_bench.evaluators.tokenization import tokenize_chinese


@pytest.fixture(autouse=True)
def controlled_dictionary(tmp_path, monkeypatch):
    # Exercise real jieba using an independently frozen tiny dictionary.
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("我 10 r\n学生 10 n\n学习 10 v\n科学 10 n\n哲学 10 n\n一 10 m\n", encoding="utf-8")
    real_tokenizer = jieba.Tokenizer
    monkeypatch.setattr(jieba, "Tokenizer", lambda: real_tokenizer(str(dictionary)))


@pytest.fixture
def lexicon():
    return HSKLexicon(json.loads((Path(__file__).parent / "fixtures/hsk_vocab_tiny.json").read_text(encoding="utf-8")))


def test_tokenization_retains_punctuation_and_splits_chinese(lexicon):
    assert tokenize_chinese("我学生，学习！12", lexicon) == ["我", "学生", "，", "学习", "！", "12"]
    assert tokenize_chinese("我，12 abc") == ["我", "，", "12", "abc"]


def test_hand_calculated_metrics_and_auditable_trace(lexicon):
    result = score_lexical_alignment("我我学生，学习科学哲学 12 XYZ 一。", 3, lexicon)
    assert result.v_in_token == 5 / 9
    assert result.v_in_type == 4 / 8
    assert result.v_oov_token == 2 / 9
    assert result.v_oov_severe == 1 / 9
    assert result.unknown_token_rate == 2 / 9
    assert result.mattr_50 == 8 / 9
    assert result.tokens == ["我", "我", "学生", "，", "学习", "科学", "哲学", "12", "XYZ", "一", "。"]
    assert result.scorable_tokens == ["我", "我", "学生", "学习", "科学", "哲学", "12", "XYZ", "一"]
    assert result.metadata["tokenizer_version"]
    assert result.metadata["tokenizer"]
    assert result.metadata["lexicon_sha256"]


@pytest.mark.parametrize("count,expected", [(1, 1), (49, 1 / 49), (50, 1 / 50), (51, 0.03)])
def test_mattr_short_and_sliding_windows(lexicon, count, expected):
    text = "我 " * count if count <= 50 else "我 " * 50 + "学生"
    assert score_lexical_alignment(text, 3, lexicon).mattr_50 == expected


@pytest.mark.parametrize("text", ["", " \n\t", "，。！？—…"])
def test_empty_scorable_text_fails(lexicon, text):
    with pytest.raises(ValueError, match="scorable"):
        score_lexical_alignment(text, 3, lexicon)


@pytest.mark.parametrize("level", [True, 0, 10, 3.5])
def test_invalid_target_levels_fail(lexicon, level):
    with pytest.raises(ValueError):
        score_lexical_alignment("我", level, lexicon)


@pytest.mark.parametrize("mapping", [{"": 1}, {"我": True}, {"我": 0}, {"我": 10}, {" 我": 1}])
def test_invalid_canonical_lexicon_fails(mapping):
    with pytest.raises(ValueError):
        HSKLexicon(mapping)


def test_longest_match_unknowns_and_reproducibility():
    lexicon = HSKLexicon({"学": 1, "学生": 4})
    result = score_lexical_alignment("学生龘+", 3, lexicon)
    assert result.tokens == ["学生", "龘", "+"]
    assert result.v_oov_token == 1 / 3
    assert result.unknown_token_rate == 2 / 3
    assert result.v_oov_severe == 0
    assert result.metadata == score_lexical_alignment("学生龘+", 3, HSKLexicon({"学生": 4, "学": 1})).metadata
