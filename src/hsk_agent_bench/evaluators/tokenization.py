"""Local jieba segmentation, independent of process-global dictionary changes."""

from collections.abc import Mapping

import jieba


def tokenize_chinese(text: str, lexicon: Mapping[str, int] | None = None) -> list[str]:
    """Segment text with HMM disabled; preserve punctuation, discard whitespace.

    Each call owns its dictionary. Canonical entries receive the same fixed
    frequency in sorted order, making segmentation independent of mapping order.
    Unlisted words retain jieba's base-dictionary segmentation.
    """
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for word in sorted(lexicon or {}):
        tokenizer.add_word(word, freq=1_000_000)
    return [token for token in tokenizer.cut(text, HMM=False) if not token.isspace()]
