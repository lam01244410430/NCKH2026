"""Immutable evaluation of verified generation shards."""

from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import Field

from ..io import verify_shard, write_jsonl_shard_atomic
from ..schemas import GenerationRecord, MetricRecord, Record
from .grammar import GrammarCatalog, score_grammar_alignment
from .lexical import HSKLexicon, score_lexical_alignment
from .syntax import ParsedDocument, Parser, score_syntax
from .tokenization import tokenize_chinese


class EvaluationSummary(Record):
    input_path: Path
    output_path: Path
    checksum_path: Path
    records_read: int = Field(ge=0)
    records_written: int = Field(ge=0)


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
    words: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or "word" not in row or "min_hsk_level" not in row:
            raise ValueError("each vocabulary row requires word and min_hsk_level")
        word, level = row["word"], row["min_hsk_level"]
        words[word] = min(words.get(word, level), level)
    return HSKLexicon(words)


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
        lexical = score_lexical_alignment(record.text, record.hsk_level, config.lexicon)
        syntax = score_syntax(
            record.text, lambda text: tokenize_chinese(text, config.lexicon), parser,
            generation_id=record.generation_id, prompt_id=record.prompt_id,
        )
        values = {
            key: value for key, value in lexical.model_dump().items()
            if key not in {"tokens", "scorable_tokens", "metadata"}
        }
        values.update(syntax.metrics)
        metadata = {"lexical": lexical.metadata, "syntax": syntax.metadata}
        if config.grammar_catalog is not None:
            grammar = score_grammar_alignment(record.text, record.hsk_level, config.grammar_catalog)
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
