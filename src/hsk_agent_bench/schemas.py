"""Shared validated records. All HSK prompt levels refer to the core 3--6 study."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskType(StrEnum):
    READING = "reading"
    GRAMMAR_EXPLANATION = "grammar_explanation"
    EXAM_GENERATION = "exam_generation"


class ModelOrigin(StrEnum):
    CHINA = "china"
    GLOBAL = "global"


HSKLevel = Annotated[int, Field(strict=True, ge=3, le=6)]
NonEmpty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Provider = Literal["openai", "gemini", "qwen", "deepseek"]


class ReadingPayload(Record):
    task: Literal["reading"] = "reading"
    target_length_chars: int = Field(gt=0)
    required_vocab: list[str] = Field(default_factory=list)
    required_grammar: list[str] = Field(default_factory=list)
    max_out_of_level_words: int = Field(default=0, ge=0)


class GrammarPayload(Record):
    task: Literal["grammar_explanation"] = "grammar_explanation"
    grammar_point: NonEmpty
    learner_l1: NonEmpty = "Chinese"
    context: str = ""
    example_count: int = Field(default=3, gt=0)


class ExamPayload(Record):
    task: Literal["exam_generation"] = "exam_generation"
    item_type: NonEmpty = "multiple_choice"
    source_length: int = Field(default=300, gt=0)
    num_items: int = Field(default=1, gt=0)


TaskPayload = Annotated[ReadingPayload | GrammarPayload | ExamPayload, Field(discriminator="task")]


class PromptRecord(Record):
    prompt_id: NonEmpty
    task: TaskType
    hsk_level: HSKLevel
    topic: NonEmpty
    payload: TaskPayload
    prompt_version: NonEmpty = "1"
    prompt_hash: Digest | None = None

    @model_validator(mode="after")
    def matching_task(self):
        if self.task != self.payload.task:
            raise ValueError("task must match the task-specific payload")
        return self


class ChatMessage(Record):
    role: Literal["system", "user", "assistant"]
    content: NonEmpty


class GenerationParameters(Record):
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(ge=0, le=1)
    max_output_tokens: int = Field(gt=0)
    seed: int | None = None


class GenerationRequest(Record):
    generation_id: NonEmpty
    prompt_id: NonEmpty
    task: TaskType
    hsk_level: HSKLevel
    family: NonEmpty
    origin: ModelOrigin
    provider: Provider
    model_id: NonEmpty
    prompt_hash: Digest
    messages: list[ChatMessage] = Field(min_length=1)
    parameters: GenerationParameters
    run_id: NonEmpty = "default"
    replicate: int = Field(default=0, ge=0)


class TokenUsage(Record):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationRecord(GenerationRequest):
    text: str
    raw_response: dict[str, Any]
    requested_at: datetime
    request_id: str | None = None
    model_version_date: str | None = None
    latency_ms: float = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    retry_count: int = Field(default=0, ge=0)
    seed_applied: bool


class MetricRecord(Record):
    generation_id: NonEmpty
    prompt_id: NonEmpty
    evaluator: NonEmpty
    evaluator_version: NonEmpty
    device: NonEmpty = "cpu"
    metrics: dict[str, float | None]
    metadata: dict[str, Any] = Field(default_factory=dict)


OrdinalScore = Annotated[int, Field(strict=True, ge=1, le=5)]


class RatingRecord(Record):
    generation_id: NonEmpty
    prompt_id: NonEmpty
    rater_id: NonEmpty
    rater_type: Literal["human", "llm"]
    scores: dict[str, OrdinalScore] = Field(min_length=1)
    rubric_version: NonEmpty = "1"
    evidence: dict[str, str] = Field(default_factory=dict)
    rated_at: datetime | None = None
