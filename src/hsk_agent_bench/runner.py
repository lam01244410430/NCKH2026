"""Resumable async generation with verified, immutable, single-writer shards.

Only `<run_id>/*.jsonl` are final generation shards. `.checkpoints`, `.state`,
and `.failures` are internal journals, ownership metadata, and error reports.
Never ingest these internal directories as generation data.

Each seed is crossed with each repeat: the persisted replicate number is
`seed_index * run.replicates + repeat_index`. A complete final shard is never
reopened. Partial work resumes from per-generation atomic checkpoint pairs.
Stale locks and incomplete checksum pairs fail closed for manual inspection.
"""
import asyncio
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from .config import ModelConfig, RunConfig
from .generators import make_generator
from .generators.base import ProviderError
from .io import verify_shard, write_jsonl_shard_atomic
from .prompts.common import canonical_json, hash_messages
from .prompts.exam_generation import render_exam_prompt
from .prompts.grammar_explanation import render_grammar_prompt
from .prompts.reading import render_reading_prompt
from .schemas import GenerationParameters, GenerationRecord, GenerationRequest, PromptRecord, Record, TaskType


class RunSummary(Record):
    run_id: str
    output_dir: Path
    shards: list[Path] = Field(default_factory=list)
    failure_shards: list[Path] = Field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    retried: int = 0


class FailureRecord(Record):
    request: GenerationRequest
    message: str
    error_type: str
    status_code: int | None = None
    retry_count: int
    terminal: bool
    failed_at: datetime
    raw_response: dict | None = None


class _Identity(Record):
    digest: str


class _Owner(Record):
    namespace: str
    identity: str


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _read(path: Path, record_type: type[Record]) -> list:
    if not verify_shard(path):
        raise ValueError(f"shard checksum integrity failure: {path}")
    return [record_type.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _present(path: Path) -> bool:
    return path.exists() or Path(str(path) + ".sha256").exists() or Path(str(path) + ".lock").exists()


def _ensure_metadata(path: Path, record: Record) -> None:
    if _present(path):
        if _read(path, type(record)) != [record]:
            raise ValueError(f"shard owner namespace or run configuration identity mismatch: {path}")
    else:
        write_jsonl_shard_atomic([record], path)


@contextmanager
def _claim(path: Path):
    handle = path.open("x", encoding="utf-8")
    try:
        yield
    finally:
        handle.close()
        path.unlink()


def _validate_record(record: GenerationRecord, request: GenerationRequest) -> None:
    # A checkpoint must describe exactly the requested experiment, even if a
    # malicious or mistaken writer recomputed a checksum for the wrong record.
    actual = {key: getattr(record, key) for key in GenerationRequest.model_fields}
    if GenerationRequest(**actual) != request:
        raise ValueError("generation checkpoint request identity mismatch")


async def run_manifest(
    manifest: Sequence[PromptRecord], models: Sequence[ModelConfig], run: RunConfig, output_dir: Path, *,
    run_id: str | None = None, shard_namespace: str = "laptop",
    provider_concurrency: Mapping[str, int] | None = None, max_retries: int = 3,
    transient_statuses: set[int] | frozenset[int] = frozenset({429, 500, 502, 503, 504}),
    backoff_seconds: float = 1.0,
) -> RunSummary:
    """Generate frozen requests, retry configured HTTP statuses, and resume.

    Limits are shared by all models using a provider and default to one call.
    `max_retries` counts additional attempts, not the initial request. Provider
    SDK/HTTP transports have no independent retry policy. Terminal failures are
    durable and remain terminal on resume; start a distinct run to recollect.

    Pass the device profile's shard_namespace explicitly from orchestration.
    Namespaces own shards, not separate copies: using `pc` after `laptop` on
    the same run/output directory is an error even after the first run closes.
    """
    name_pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    if not re.fullmatch(name_pattern, shard_namespace):
        raise ValueError("invalid shard_namespace")
    if run_id is not None and not re.fullmatch(name_pattern, run_id):
        raise ValueError("invalid run_id")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if not math.isfinite(backoff_seconds) or backoff_seconds < 0:
        raise ValueError("backoff_seconds must be finite and nonnegative")
    if any(isinstance(code, bool) or not isinstance(code, int) or not 100 <= code <= 599 for code in transient_statuses):
        raise ValueError("transient_statuses must contain HTTP status integers")
    limits = dict(provider_concurrency or {})
    if any(provider not in {"openai", "gemini", "qwen", "deepseek"} or isinstance(limit, bool)
           or not isinstance(limit, int) or limit < 1 for provider, limit in limits.items()):
        raise ValueError("provider_concurrency requires supported providers and positive integers")
    prompts = [record.model_copy(deep=True) for record in manifest]
    enabled = [model.model_copy(deep=True) for model in models if model.enabled]
    run = run.model_copy(deep=True)
    if len({record.prompt_id for record in prompts}) != len(prompts):
        raise ValueError("manifest prompt IDs must be unique")
    if len({model.family for model in enabled}) != len(enabled):
        raise ValueError("model families must be unique")
    renderers = {TaskType.READING: render_reading_prompt, TaskType.GRAMMAR_EXPLANATION: render_grammar_prompt,
                 TaskType.EXAM_GENERATION: render_exam_prompt}
    rendered = {}
    for prompt in prompts:
        messages = renderers[prompt.task](prompt)
        if prompt.prompt_hash != hash_messages(messages):
            raise ValueError(f"manifest prompt hash mismatch: {prompt.prompt_id}")
        if prompt.task not in run.max_output_tokens:
            raise ValueError(f"missing output token budget: {prompt.task}")
        rendered[prompt.prompt_id] = messages
    identity = _digest({"manifest": [record.model_dump(mode="json") for record in prompts],
                        "models": [model.model_dump(mode="json") for model in enabled],
                        "run": run.model_dump(mode="json"), "max_retries": max_retries,
                        "transient_statuses": sorted(transient_statuses), "backoff_seconds": backoff_seconds})
    run_id = run_id or f"run-{identity[:24]}"
    root = Path(output_dir) / run_id
    summary = RunSummary(run_id=run_id, output_dir=root)
    if not prompts or not enabled:
        return summary
    # Rank without process-dependent random state; preserve original prompt order
    # inside final shards, and use the identical subset for every model/repeat.
    selected_ids = {prompt.prompt_id for prompt in sorted(prompts, key=lambda prompt: _digest(
        [run.seeds[0], prompt.prompt_id, prompt.prompt_hash]))[:math.ceil(len(prompts) * run.fraction)]}
    selected = [prompt for prompt in prompts if prompt.prompt_id in selected_ids]
    for directory in (root, root / ".state", root / ".checkpoints", root / ".failures"):
        directory.mkdir(parents=True, exist_ok=True)
    _ensure_metadata(root / ".state" / "identity.jsonl", _Identity(digest=identity))
    semaphores = {provider: asyncio.Semaphore(limits.get(provider, 1)) for provider in {model.provider for model in enabled}}
    jobs = []

    async def generate_one(request, generator, checkpoint_dir, successes, failures, attempts):
        generation_id = request.generation_id
        previous = attempts.get(generation_id, [])
        first_attempt = previous[-1].retry_count + 1 if previous else 0
        if previous:
            # A crash during the previous wait cannot bypass the retry policy.
            await asyncio.sleep(backoff_seconds * (2 ** previous[-1].retry_count))
        for attempt in range(first_attempt, max_retries + 1):
            async with semaphores[request.provider]:
                if attempt:
                    summary.retried += 1
                try:
                    record = await generator.generate(request.model_copy(deep=True))
                    _validate_record(record, request)
                except Exception as error:
                    status = error.status_code if isinstance(error, ProviderError) else None
                    terminal = attempt >= max_retries or status not in transient_statuses
                    failure = FailureRecord(request=request, message=str(error), error_type=type(error).__name__,
                                            status_code=status, retry_count=attempt, terminal=terminal,
                                            failed_at=datetime.now(timezone.utc),
                                            raw_response=error.raw_response if isinstance(error, ProviderError) else None)
                    write_jsonl_shard_atomic([failure], checkpoint_dir / f"{generation_id}.attempt-{attempt:04d}.jsonl")
                    if terminal:
                        failures[generation_id] = failure
                        return
                else:
                    record.retry_count = attempt
                    write_jsonl_shard_atomic([record], checkpoint_dir / f"{generation_id}.success.jsonl")
                    successes[generation_id] = record
                    return
            await asyncio.sleep(backoff_seconds * (2 ** attempt))

    async def finish_shard(target, checkpoint_dir, requests, model, successes, failures, attempts):
        generator = make_generator(model)
        tasks = [asyncio.create_task(generate_one(request, generator, checkpoint_dir, successes, failures, attempts))
                 for request in requests if request.generation_id not in successes and request.generation_id not in failures]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        ordered_successes = [successes[request.generation_id] for request in requests if request.generation_id in successes]
        ordered_failures = [failures[request.generation_id] for request in requests if request.generation_id in failures]
        failure_target = root / ".failures" / target.name
        # Failures publish first so the generation shard remains the final marker.
        if ordered_failures:
            if _present(failure_target):
                if _read(failure_target, FailureRecord) != ordered_failures:
                    raise ValueError("failure shard identity mismatch")
            else:
                write_jsonl_shard_atomic(ordered_failures, failure_target)
            summary.failure_shards.append(failure_target)
        write_jsonl_shard_atomic(ordered_successes, target)
        summary.shards.append(target)
        summary.succeeded += len(ordered_successes)
        summary.failed += len(ordered_failures)

    with ExitStack() as stack:
        for model in enabled:
            for seed_index, seed in enumerate(run.seeds):
                for repeat_index in range(run.replicates):
                    replicate = seed_index * run.replicates + repeat_index
                    stem = f"{model.provider}-{_digest(model.model_dump(mode='json'))[:16]}-replicate-{replicate:04d}"
                    target = root / f"{stem}.jsonl"
                    stack.enter_context(_claim(root / ".state" / f"{stem}.active"))
                    _ensure_metadata(root / ".state" / f"{stem}.owner.jsonl", _Owner(namespace=shard_namespace, identity=identity))
                    requests = []
                    for prompt in selected:
                        generation_id = _digest([run_id, model.model_dump(mode="json"), prompt.prompt_id, prompt.prompt_hash, replicate])
                        requests.append(GenerationRequest(
                            generation_id=generation_id, prompt_id=prompt.prompt_id, task=prompt.task, hsk_level=prompt.hsk_level,
                            family=model.family, origin=model.origin, provider=model.provider, model_id=model.model_id,
                            prompt_hash=prompt.prompt_hash, messages=rendered[prompt.prompt_id], run_id=run_id, replicate=replicate,
                            parameters=GenerationParameters(temperature=run.temperature, top_p=run.top_p,
                                                            max_output_tokens=run.max_output_tokens[prompt.task], seed=seed)))
                    expected = {request.generation_id: request for request in requests}
                    failure_target = root / ".failures" / target.name
                    errors = _read(failure_target, FailureRecord) if _present(failure_target) else []
                    if _present(target):
                        records = _read(target, GenerationRecord)
                        completed = [record.generation_id for record in records] + [error.request.generation_id for error in errors]
                        if len(completed) != len(set(completed)) or set(completed) != set(expected):
                            raise ValueError("final shard request identity mismatch")
                        for record in records:
                            _validate_record(record, expected[record.generation_id])
                        for error in errors:
                            if not error.terminal or error.request != expected[error.request.generation_id]:
                                raise ValueError("failure shard request identity mismatch")
                        summary.shards.append(target)
                        if errors:
                            summary.failure_shards.append(failure_target)
                        summary.succeeded += len(records)
                        summary.failed += len(errors)
                        summary.skipped += len(records)
                        continue
                    checkpoint_dir = root / ".checkpoints" / stem
                    checkpoint_dir.mkdir(exist_ok=True)
                    successes, failures, attempts = {}, {}, {}
                    # Verify every entry including orphan sidecars/locks before
                    # scheduling a paid call. Incomplete pairs are never erased.
                    paths = {Path(str(path).removesuffix(".sha256").removesuffix(".lock")) for path in checkpoint_dir.iterdir()}
                    for path in sorted(paths):
                        if path.name.endswith(".success.jsonl"):
                            records = _read(path, GenerationRecord)
                            if len(records) != 1 or records[0].generation_id not in expected:
                                raise ValueError("generation checkpoint identity mismatch")
                            record = records[0]
                            _validate_record(record, expected[record.generation_id])
                            if path.name != f"{record.generation_id}.success.jsonl":
                                raise ValueError("checkpoint filename identity mismatch")
                            successes[record.generation_id] = record
                        else:
                            records = _read(path, FailureRecord)
                            if len(records) != 1 or records[0].request.generation_id not in expected:
                                raise ValueError("failure checkpoint identity mismatch")
                            error = records[0]
                            generation_id = error.request.generation_id
                            if error.request != expected[generation_id] or path.name != f"{generation_id}.attempt-{error.retry_count:04d}.jsonl":
                                raise ValueError("failure checkpoint request identity mismatch")
                            attempts.setdefault(generation_id, []).append(error)
                    for generation_id, history in attempts.items():
                        history.sort(key=lambda error: error.retry_count)
                        if [error.retry_count for error in history] != list(range(len(history))):
                            raise ValueError("failure checkpoint attempt sequence mismatch")
                        if history[-1].terminal:
                            if generation_id in successes:
                                raise ValueError("conflicting terminal checkpoints")
                            failures[generation_id] = history[-1]
                    summary.skipped += len(successes)
                    jobs.append((target, checkpoint_dir, requests, model, successes, failures, attempts))
        tasks = [asyncio.create_task(finish_shard(*job)) for job in jobs]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    summary.shards.sort()
    summary.failure_shards.sort()
    return summary
