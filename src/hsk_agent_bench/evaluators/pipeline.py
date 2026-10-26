"""Immutable evaluation of verified generation shards."""

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import ConfigDict, Field
import yaml

from ..io import verify_shard, write_jsonl_shard_atomic
from ..schemas import GenerationRecord, MetricRecord, Record
from .grammar import GrammarCatalog, score_grammar_alignment
from .lexical import HSKLexicon, score_lexical_alignment
from .materials import material_metadata, project_material_text
from .syntax import LazyParser, ParsedDocument, Parser, score_syntax
from .tokenization import tokenize_chinese


class EvaluationSummary(Record):
    input_path: Path
    output_path: Path
    checksum_path: Path
    records_read: int = Field(ge=0)
    records_written: int = Field(ge=0)


class CanonicalVocabularyResource(Record):
    """Validated provenance-bearing row from the canonical vocabulary JSON."""

    model_config = ConfigDict(frozen=True)
    word: str = Field(min_length=1)
    min_hsk_level: int = Field(strict=True, ge=1, le=9)
    standard: str = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    notes: str | None = None


class ParserPluginConfig(Record):
    """Local parser factory metadata; only a CPU backend is accepted here."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    factory: str = Field(min_length=3)
    device: Literal["cpu"] = "cpu"


class _UnavailableParser:
    name = "not-configured"
    version = "0"
    device = "cpu"

    def parse(self, text: str) -> ParsedDocument:
        raise RuntimeError("dependency parser is not configured")


@dataclass(frozen=True)
class EvaluationConfig:
    lexicon: HSKLexicon
    grammar_catalog: GrammarCatalog | None = None
    parser: Parser | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("evaluation device must be cpu or cuda")


def load_lexicon(path: Path) -> HSKLexicon:
    """Load canonical vocabulary rows, retaining the lowest listed level."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("vocabulary source must be a JSON array")
    resources = [CanonicalVocabularyResource.model_validate(row) for row in rows]
    words: dict[str, int] = {}
    for resource in resources:
        words[resource.word] = min(words.get(resource.word, resource.min_hsk_level),
                                   resource.min_hsk_level)
    return HSKLexicon(words)


def load_parser_config(path: Path) -> LazyParser:
    """Create a lazy local CPU parser from a ``module:factory`` YAML config.

    The module and factory are not imported or called until the first parse.
    The factory receives the literal ``"cpu"`` and must return a local backend
    with ``parse(text)``; downloading models is outside this interface.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = ParserPluginConfig.model_validate(payload)
    pattern = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    if re.fullmatch(pattern, config.factory) is None:
        raise ValueError("parser factory must be a module:callable import path")
    module_name, attribute_path = config.factory.split(":", 1)

    def loader(device: str):
        target = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            target = getattr(target, attribute)
        if not callable(target):
            raise TypeError("configured parser factory is not callable")
        return target(device)

    return LazyParser(name=config.name, version=config.version, loader=loader, device="cpu")


def evaluate_shard(input_path: Path, output_path: Path,
                   config: EvaluationConfig) -> EvaluationSummary:
    """Score every record in a verified shard and publish one immutable metric shard."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not verify_shard(input_path):
        raise ValueError(f"input shard checksum integrity failure: {input_path}")
    records = [GenerationRecord.model_validate_json(line) for line in input_path.read_text(
        encoding="utf-8").splitlines()]
    parser = config.parser or _UnavailableParser()
    metrics = []
    for record in records:
        text = project_material_text(record.task, record.text, hsk_level=record.hsk_level)
        lexical = score_lexical_alignment(text, record.hsk_level, config.lexicon)
        syntax = score_syntax(
            text, lambda value: tokenize_chinese(value, config.lexicon), parser,
            generation_id=record.generation_id, prompt_id=record.prompt_id,
        )
        values = {
            key: value for key, value in lexical.model_dump().items()
            if key not in {"tokens", "scorable_tokens", "metadata"}
        }
        values.update(syntax.metrics)
        metadata = {"material": material_metadata(record.task, record.text, text),
                    "lexical": lexical.metadata, "syntax": syntax.metadata}
        if config.grammar_catalog is not None:
            grammar = score_grammar_alignment(text, record.hsk_level, config.grammar_catalog)
            values.update({
                "in_level_matches": float(grammar.in_level_matches),
                "target_level_matches": float(grammar.target_level_matches),
                "above_level_matches": float(grammar.above_level_matches),
            })
            metadata["grammar"] = grammar.metadata
        metrics.append(MetricRecord(
            generation_id=record.generation_id, prompt_id=record.prompt_id,
            evaluator="evaluation_pipeline", evaluator_version="1", device=parser.device,
            metrics=values, metadata=metadata,
        ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checksum = write_jsonl_shard_atomic(metrics, output_path)
    return EvaluationSummary(input_path=input_path, output_path=output_path,
                             checksum_path=checksum, records_read=len(records),
                             records_written=len(metrics))
