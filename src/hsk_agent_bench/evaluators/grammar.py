"""Catalog alignment and human-confirmed grammar violations are separate measures.

Detector backends (including optional MuCGEC adapters) implement GrammarDetector;
none are imported or installed by the core package. Flags are candidate evidence,
never error labels. Human decisions are supplied separately, keyed by flag_id.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from ..schemas import Record


class CanonicalGrammarResource(Record):
    """Typed snapshot of data/hsk_grammar_points/schema.example.json entries."""

    model_config = ConfigDict(frozen=True)
    grammar_id: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    min_hsk_level: int = Field(strict=True, ge=1, le=9)
    description: str
    examples: tuple[str, ...]
    standard: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    notes: str | None = None


class GrammarRule(Record):
    model_config = ConfigDict(frozen=True)
    grammar_id: str = Field(min_length=1)
    level: int = Field(strict=True, ge=1, le=9)
    pattern: str = Field(min_length=1)
    provenance: CanonicalGrammarResource | None = None


@dataclass(frozen=True, init=False)
class GrammarCatalog:
    """Versionable regex catalog; matches indicate presence, not grammaticality."""

    rules: tuple[GrammarRule, ...]
    patterns: tuple[re.Pattern[str], ...]
    sha256: str

    def __init__(self, entries: Sequence[GrammarRule | Mapping]):
        # Revalidation snapshots even existing model instances and nested provenance.
        rules = tuple(GrammarRule.model_validate(
            entry.model_dump() if isinstance(entry, GrammarRule) else entry
        ) for entry in entries)
        object.__setattr__(self, "rules", rules)
        if len({rule.grammar_id for rule in self.rules}) != len(self.rules):
            raise ValueError("grammar IDs must be unique")
        try:
            patterns = tuple(re.compile(rule.pattern) for rule in self.rules)
        except re.error as exc:
            raise ValueError("invalid grammar regex") from exc
        object.__setattr__(self, "patterns", patterns)
        if any(pattern.search("") is not None for pattern in self.patterns):
            raise ValueError("grammar patterns must not match empty strings")
        digest = hashlib.sha256(json.dumps(
            [rule.model_dump() for rule in self.rules], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_canonical(cls, entries: Sequence[CanonicalGrammarResource | Mapping]) -> "GrammarCatalog":
        """Project canonical min_hsk_level to level; retain all source fields.

        Canonical pattern strings are interpreted as evaluator regexes. Resources
        must therefore be curated for regex matching before study use; this loader
        does not infer correctness or official HSK status from source metadata.
        """
        resources = (CanonicalGrammarResource.model_validate(
            entry.model_dump() if isinstance(entry, CanonicalGrammarResource) else entry
        ) for entry in entries)
        return cls([GrammarRule(grammar_id=resource.grammar_id,
                                level=resource.min_hsk_level, pattern=resource.pattern,
                                provenance=resource) for resource in resources])


class GrammarMatch(Record):
    grammar_id: str
    level: int
    start: int
    end: int
    text: str


class GrammarAlignmentMetrics(Record):
    in_level_matches: int = Field(ge=0)
    target_level_matches: int = Field(ge=0)
    above_level_matches: int = Field(ge=0)
    matches: list[GrammarMatch]
    metadata: dict[str, str]


def score_grammar_alignment(text: str, target_level: int,
                            catalog: GrammarCatalog) -> GrammarAlignmentMetrics:
    """Count non-overlapping occurrences per rule; preserve spans for audit."""
    if type(target_level) is not int or not 1 <= target_level <= 9:
        raise ValueError("target_level must be an integer from 1 to 9")
    matches = []
    for rule, pattern in zip(catalog.rules, catalog.patterns):
        for match in pattern.finditer(text):
            if match.start() == match.end():
                raise ValueError("grammar patterns must not produce empty matches")
            matches.append(GrammarMatch(grammar_id=rule.grammar_id, level=rule.level,
                                        start=match.start(), end=match.end(), text=match.group()))
    matches.sort(key=lambda match: (match.start, match.end, match.grammar_id))
    return GrammarAlignmentMetrics(
        in_level_matches=sum(match.level <= target_level for match in matches),
        target_level_matches=sum(match.level == target_level for match in matches),
        above_level_matches=sum(match.level > target_level for match in matches),
        matches=matches, metadata={"catalog_sha256": catalog.sha256, "evaluator_version": "1",
                                  "match_policy": "non-overlapping per rule; presence only"})


class GrammarFlag(Record):
    flag_id: str = Field(min_length=1)
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, gt=0)
    reason: str = Field(min_length=1)
    source: str = Field(min_length=1)
    status: Literal["candidate"] = "candidate"

    @model_validator(mode="after")
    def nonempty_span(self):
        if self.end <= self.start:
            raise ValueError("flag span must be nonempty")
        return self


class GrammarDetector(Protocol):
    def detect(self, text: str) -> Sequence[GrammarFlag]: ...


def _unique_flags(flags: Sequence[GrammarFlag]) -> list[GrammarFlag]:
    unique = {}
    for flag in flags:
        flag = GrammarFlag.model_validate(flag)
        if flag.flag_id in unique and flag != unique[flag.flag_id]:
            raise ValueError("conflicting flag IDs")
        unique[flag.flag_id] = flag
    return list(unique.values())


def candidate_grammar_flags(text: str, detector: GrammarDetector) -> list[GrammarFlag]:
    """Export candidate spans/reasons for blinded human review (Python offsets)."""
    flags = _unique_flags(detector.detect(text))
    if any(flag.end > len(text) for flag in flags):
        raise ValueError("candidate flag span exceeds text length")
    return flags


def confirmed_violation_rate(flags: Sequence[GrammarFlag], labels: Mapping[str, bool],
                             token_count: int) -> float:
    """Human-confirmed unique violations per 100 Chinese tokens.

    Caller supplies the count from Chinese tokenization, excluding punctuation,
    Latin tokens and numerals. Missing/false labels contribute zero. Labels must
    originate in human review; automatic detector output is never confirmation.
    """
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("token_count must be a positive Chinese token count")
    if any(type(label) is not bool for label in labels.values()):
        raise ValueError("human labels must be booleans")
    return 100 * sum(labels.get(flag.flag_id, False) for flag in _unique_flags(flags)) / token_count
