"""Sentence and dependency metrics with injected, lazy local parser backends."""

from collections.abc import Callable, Sequence
import re
from typing import Protocol
import unicodedata

from pydantic import Field, model_validator

from ..schemas import MetricRecord, Record


class Tokenizer(Protocol):
    def __call__(self, text: str) -> Sequence[str]: ...


class DependencyToken(Record):
    index: int = Field(strict=True, gt=0)
    text: str = Field(min_length=1)
    head: int = Field(strict=True, ge=0)
    relation: str = Field(min_length=1)


class ParsedSentence(Record):
    tokens: list[DependencyToken]

    @model_validator(mode="after")
    def valid_heads(self):
        indices = [token.index for token in self.tokens]
        if indices != list(range(1, len(self.tokens) + 1)):
            raise ValueError("dependency indices must be contiguous and one-based")
        for token in self.tokens:
            if token.head > len(self.tokens) or token.head == token.index:
                raise ValueError("dependency head must reference another token or zero")
        return self


class ParsedDocument(Record):
    sentences: list[ParsedSentence]


class ParserBackend(Protocol):
    def parse(self, text: str) -> ParsedDocument: ...


class Parser(ParserBackend, Protocol):
    name: str
    version: str
    device: str


class LazyParser:
    """Load a configured local backend on first parse, retrying GPU failures on CPU.

    The loader owns model configuration and must not download models. GPU use is
    opt-in via an availability probe; without one, CPU is deterministic by default.
    Backends normalize parser-specific output into ParsedDocument (root head=0).
    """

    def __init__(self, *, name: str, version: str,
                 loader: Callable[[str], ParserBackend], device: str = "cpu",
                 gpu_available: Callable[[], bool] | None = None):
        if not name or not version or device not in {"cpu", "cuda"}:
            raise ValueError("parser name/version are required; device must be cpu or cuda")
        self.name, self.version = name, version
        self._requested_device = device
        self.device = "cpu"
        self._loader = loader
        self._gpu_available = gpu_available or (lambda: False)
        self._backend: ParserBackend | None = None

    def parse(self, text: str) -> ParsedDocument:
        if self._backend is None:
            try:
                use_gpu = self._requested_device == "cuda" and self._gpu_available()
            except Exception:
                use_gpu = False
            self.device = "cuda" if use_gpu else "cpu"
        try:
            if self._backend is None:
                self._backend = self._loader(self.device)
            return ParsedDocument.model_validate(self._backend.parse(text))
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self._requested_device = "cpu"
            self._backend = None
            self._backend = self._loader("cpu")
            return ParsedDocument.model_validate(self._backend.parse(text))


def mean_sentence_length(text: str, tokenizer: Tokenizer) -> float:
    """Mean non-punctuation token count over nonempty terminal-delimited sentences.

    Chinese/ASCII full stops, questions, exclamations and newlines terminate a
    sentence. Commas and semicolons do not. Empty/punctuation-only input is zero.
    """
    lengths = []
    for sentence in re.split(r"[。！？.!?\r\n]+", text):
        count = sum(not all(char.isspace() or unicodedata.category(char).startswith("P")
                            for char in token) for token in tokenizer(sentence))
        if count:
            lengths.append(count)
    return sum(lengths) / len(lengths) if lengths else 0.0


def mean_dependency_distance(parsed: ParsedDocument) -> float:
    """Arc-weighted mean |dependent-head|; exclude root arcs (head=0)."""
    distances = [abs(token.index - token.head) for sentence in parsed.sentences
                 for token in sentence.tokens if token.head != 0]
    return sum(distances) / len(distances) if distances else 0.0


def score_syntax(text: str, tokenizer: Tokenizer, parser: Parser, *,
                 generation_id: str, prompt_id: str) -> MetricRecord:
    """Preserve parser failures as missing MDD with provenance and diagnostic data."""
    metrics = {"mean_sentence_length": mean_sentence_length(text, tokenizer),
               "mean_dependency_distance": None}
    error = None
    try:
        metrics["mean_dependency_distance"] = mean_dependency_distance(parser.parse(text))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    metadata = {"parser_name": parser.name, "parser_version": parser.version,
                "parser_device": parser.device, "root_arc_policy": "exclude head=0"}
    if error is not None:
        metadata["parser_error"] = error
    return MetricRecord(generation_id=generation_id, prompt_id=prompt_id,
                        evaluator="syntax", evaluator_version="1", device=parser.device,
                        metrics=metrics, metadata=metadata)
