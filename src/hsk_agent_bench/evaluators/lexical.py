"""HSK lexical fractions over all non-punctuation tokens, including unknowns."""

from collections.abc import Iterator, Mapping
import hashlib
import json
import unicodedata

import jieba
from pydantic import Field

from ..schemas import Record
from .tokenization import tokenize_chinese


class HSKLexicon(Mapping[str, int]):
    """Immutable canonical word -> minimum level mapping (HSK levels 1--9).

    Input must already be NFC-normalized and trimmed, matching the canonical
    data format. Provenance/standard selection belongs to the source dataset.
    """

    def __init__(self, words: Mapping[str, int]):
        self._words = dict(words)
        for word, level in self._words.items():
            if (not isinstance(word, str) or not word or word.strip() != word
                    or unicodedata.normalize("NFC", word) != word):
                raise ValueError("lexicon words must be nonempty, trimmed NFC strings")
            if type(level) is not int or not 1 <= level <= 9:
                raise ValueError("minimum HSK levels must be integers from 1 to 9")

    def __getitem__(self, word: str) -> int:
        return self._words[word]

    def __iter__(self) -> Iterator[str]:
        return iter(self._words)

    def __len__(self) -> int:
        return len(self._words)


class LexicalMetrics(Record):
    v_in_token: float = Field(ge=0, le=1)
    v_in_type: float = Field(ge=0, le=1)
    v_oov_token: float = Field(ge=0, le=1)
    v_oov_severe: float = Field(ge=0, le=1)
    unknown_token_rate: float = Field(ge=0, le=1)
    mattr_50: float = Field(ge=0, le=1)
    tokens: list[str]
    scorable_tokens: list[str]
    metadata: dict[str, str | int | bool]


def score_lexical_alignment(text: str, target_level: int, lexicon: HSKLexicon) -> LexicalMetrics:
    """Score normalized text; short texts use a single full-length MATTR window.

    Numerals, Latin words and symbols are scorable and unknown unless listed.
    Only Unicode punctuation and whitespace are excluded. Severe spillover is
    minimum level >= target + 2; unknowns never count as above-level vocabulary.
    """
    if type(target_level) is not int or not 1 <= target_level <= 9:
        raise ValueError("target_level must be an integer from 1 to 9")
    tokens = tokenize_chinese(text, lexicon)
    scorable = [token for token in tokens if not all(
        char.isspace() or unicodedata.category(char).startswith("P") for char in token
    )]
    if not scorable:
        raise ValueError("text must contain at least one scorable token")
    levels = [lexicon.get(token) for token in scorable]
    types = set(scorable)
    count = len(scorable)
    window = min(50, count)
    ratios = [len(set(scorable[start:start + window])) / window
              for start in range(count - window + 1)]
    digest = hashlib.sha256(json.dumps(dict(lexicon), ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return LexicalMetrics(
        v_in_token=sum(level is not None and level <= target_level for level in levels) / count,
        v_in_type=sum(token in lexicon and lexicon[token] <= target_level for token in types) / len(types),
        v_oov_token=sum(level is not None and level > target_level for level in levels) / count,
        v_oov_severe=sum(level is not None and level >= target_level + 2 for level in levels) / count,
        unknown_token_rate=sum(level is None for level in levels) / count,
        mattr_50=sum(ratios) / len(ratios), tokens=tokens, scorable_tokens=scorable,
        metadata={"tokenizer": "jieba", "tokenizer_version": jieba.__version__,
                  "hmm": False, "lexicon_word_frequency": 1_000_000,
                  "lexicon_sha256": digest, "unicode_version": unicodedata.unidata_version,
                  "punctuation_policy": "exclude Unicode P categories; retain symbols and numerals",
                  "mattr_short_text_policy": "single full-length window", "evaluator_version": "1"},
    )
