"""Real adapters, durable shards, and runner scheduling with fake network calls."""
import asyncio
import copy
import json
from collections import Counter
from pathlib import Path

import pytest

import hsk_agent_bench.runner as runner
from hsk_agent_bench.config import RunConfig, load_experiment_config
from hsk_agent_bench.generators import make_generator
from hsk_agent_bench.generators.base import ProviderError
from hsk_agent_bench.io import verify_shard
from hsk_agent_bench.prompts.common import hash_messages
from hsk_agent_bench.prompts.reading import render_reading_prompt
from hsk_agent_bench.prompts.manifest import build_manifest
from hsk_agent_bench.schemas import PromptRecord, ReadingPayload
from test_providers import fixture, model


def manifest(count=3):
    records = []
    for index in range(count):
        record = PromptRecord(prompt_id=f"p{index}", task="reading", hsk_level=3, topic=f"topic{index}",
                              payload=ReadingPayload(target_length_chars=100))
        record.prompt_hash = hash_messages(render_reading_prompt(record))
        records.append(record)
    return records


def settings(**changes):
    values = dict(temperature=.4, top_p=.9, seeds=[7], replicates=1,
                  max_output_tokens={"reading": 100})
    values.update(changes)
    return RunConfig(**values)


def install_transport(monkeypatch, transport):
    monkeypatch.setattr(runner, "make_generator", lambda config: make_generator(config, transport=transport))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_retries_only_transient_failures_preserves_payload_and_logs_terminal_errors(tmp_path, monkeypatch):
    calls = []
    attempts = Counter()
    async def transport(payload):
        calls.append(copy.deepcopy(payload))
        content = payload["messages"][-1]["content"]
        prompt_id = next(value for value in ("p0", "p1", "p2") if f"提示编号：{value}\n" in content)
        attempts[prompt_id] += 1
        if prompt_id == "p1" and attempts[prompt_id] == 1:
            raise ProviderError("rate limited", status_code=429)
        if prompt_id == "p2":
            return {"bad": "schema"}
        return fixture("openai")
    delays = []
    async def sleep(delay):
        delays.append(delay)
    install_transport(monkeypatch, transport)
    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    result = asyncio.run(runner.run_manifest(manifest(), [model("openai")], settings(), tmp_path,
                                             max_retries=2, backoff_seconds=.5))
    assert result.succeeded == 2 and result.failed == 1 and result.retried == 1
    assert attempts == {"p0": 1, "p1": 2, "p2": 1}
    assert delays == [.5]
    retry_calls = [payload for payload in calls if "提示编号：p1\n" in payload["messages"][-1]["content"]]
    assert retry_calls[0] == retry_calls[1]
    records = lines(result.shards[0])
    assert [record["prompt_id"] for record in records] == ["p0", "p1"]
    assert [record["retry_count"] for record in records] == [0, 1]
    errors = lines(result.failure_shards[0])
    assert errors[0]["request"]["prompt_id"] == "p2" and errors[0]["terminal"] is True
    assert "response" in errors[0]["message"]
    assert all(verify_shard(path) for path in result.shards + result.failure_shards)


def test_interruption_preserves_checkpoint_and_resume_skips_successful_ids(tmp_path, monkeypatch):
    calls = []
    async def interrupted(payload):
        calls.append(copy.deepcopy(payload))
        if len(calls) == 2:
            raise asyncio.CancelledError()
        return fixture("openai")
    install_transport(monkeypatch, interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_manifest(manifest(2), [model("openai")], settings(), tmp_path))
    checkpoints = list(tmp_path.rglob("*.success.jsonl"))
    assert len(checkpoints) == 1 and verify_shard(checkpoints[0])
    preserved = checkpoints[0].read_bytes()
    async def resumed(payload):
        calls.append(copy.deepcopy(payload))
        return fixture("openai")
    install_transport(monkeypatch, resumed)
    result = asyncio.run(runner.run_manifest(manifest(2), [model("openai")], settings(), tmp_path))
    assert result.succeeded == 2 and result.skipped == 1
    assert len(calls) == 3
    assert checkpoints[0].read_bytes() == preserved
    records = lines(result.shards[0])
    assert len({record["generation_id"] for record in records}) == 2
    assert {record["prompt_id"] for record in records} == {"p0", "p1"}
    original = result.shards[0].read_bytes()
    repeat = asyncio.run(runner.run_manifest(manifest(2), [model("openai")], settings(), tmp_path))
    assert repeat.skipped == 2 and len(calls) == 3
    assert repeat.shards[0].read_bytes() == original


def test_namespace_and_active_writer_cannot_claim_another_writer_shard(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path, shard_namespace="laptop"))
    original = result.shards[0].read_bytes()
    with pytest.raises(ValueError, match="owner|namespace"):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path, shard_namespace="pc"))
    assert result.shards[0].read_bytes() == original
    async def simultaneous():
        started, release = asyncio.Event(), asyncio.Event()
        async def waiting(payload):
            started.set()
            await release.wait()
            return fixture("openai")
        install_transport(monkeypatch, waiting)
        task = asyncio.create_task(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path / "other"))
        await started.wait()
        try:
            with pytest.raises(FileExistsError):
                await runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path / "other")
        finally:
            release.set()
            await task
    asyncio.run(simultaneous())


def test_bounds_concurrency_across_models_using_same_provider(tmp_path, monkeypatch):
    active, peak = 0, 0
    async def transport(payload):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001)
        active -= 1
        return fixture("openai")
    install_transport(monkeypatch, transport)
    result = asyncio.run(runner.run_manifest(manifest(3), [model("openai"), model("openai", family="other")],
                                             settings(), tmp_path, provider_concurrency={"openai": 2}))
    assert result.succeeded == 6 and peak == 2
    assert len(result.shards) == 2


def test_seed_cross_replicate_grid_has_unique_ids_and_immutable_shards(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("qwen")
    install_transport(monkeypatch, transport)
    run = settings(seeds=[7, 9], replicates=2)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("qwen")], run, tmp_path))
    records = [row for path in result.shards for row in lines(path)]
    assert len(result.shards) == 4 and len({row["generation_id"] for row in records}) == 4
    assert sorted((row["replicate"], row["parameters"]["seed"]) for row in records) == [(0, 7), (1, 7), (2, 9), (3, 9)]
    assert all(row["run_id"] == result.run_id for row in records)
    assert all(verify_shard(path) for path in result.shards)


def test_resume_rejects_tampering_or_orphan_checksum_before_network(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path))
    result.shards[0].write_text("tampered\n", encoding="utf-8")
    async def forbidden(payload):
        pytest.fail("corrupt shard must fail closed before network")
    install_transport(monkeypatch, forbidden)
    with pytest.raises(ValueError, match="checksum|integrity"):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path))


def test_explicit_run_id_refuses_changed_config_and_invalid_paths(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path, run_id="pilot"))
    with pytest.raises(ValueError, match="identity|configuration"):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(temperature=.8), tmp_path, run_id="pilot"))
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path, run_id="../escape"))


def test_only_configured_transient_statuses_retry_with_exponential_backoff(tmp_path, monkeypatch):
    count = 0
    async def transport(payload):
        nonlocal count
        count += 1
        raise ProviderError("busy", status_code=503)
    delays = []
    async def sleep(delay):
        delays.append(delay)
    install_transport(monkeypatch, transport)
    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path,
                                             transient_statuses={503}, max_retries=2, backoff_seconds=.25))
    assert count == 3 and delays == [.25, .5] and result.failed == 1 and result.retried == 2
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path / "no-retry",
                                             transient_statuses={429}, max_retries=2))
    assert count == 4 and result.failed == 1 and result.retried == 0


def test_fraction_is_same_across_models_and_disabled_models_make_no_calls(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    result = asyncio.run(runner.run_manifest(manifest(8), [model("openai"), model("openai", family="other"),
                                                         model("openai", family="disabled", enabled=False)],
                                             settings(fraction=.25), tmp_path))
    assert result.succeeded == 4 and len(result.shards) == 2
    assert {row["prompt_id"] for row in lines(result.shards[0])} == {row["prompt_id"] for row in lines(result.shards[1])}


def test_fraction_is_allocated_within_each_task_level_stratum_and_order_independent(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    root = Path(__file__).resolve().parents[1]
    prompts = build_manifest(load_experiment_config(root / "configs/experiment.yaml"))
    run = settings(fraction=.25, max_output_tokens={
        "reading": 100, "grammar_explanation": 100, "exam_generation": 100,
    })
    first = asyncio.run(runner.run_manifest(prompts, [model("openai")], run, tmp_path / "first"))
    second = asyncio.run(runner.run_manifest(list(reversed(prompts)), [model("openai")], run,
                                             tmp_path / "second"))
    first_rows, second_rows = lines(first.shards[0]), lines(second.shards[0])
    counts = Counter((row["task"], row["hsk_level"]) for row in first_rows)
    assert len(counts) == 12
    assert set(counts.values()) == {3}
    assert {row["prompt_id"] for row in first_rows} == {row["prompt_id"] for row in second_rows}


def test_resume_preserves_retry_budget_and_backoff_after_cancelled_wait(tmp_path, monkeypatch):
    count = 0
    async def transport(payload):
        nonlocal count
        count += 1
        if count == 1:
            raise ProviderError("rate limited", status_code=429)
        return fixture("openai")
    async def interrupted_sleep(delay):
        raise asyncio.CancelledError()
    install_transport(monkeypatch, transport)
    monkeypatch.setattr(runner.asyncio, "sleep", interrupted_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path,
                                       max_retries=1, backoff_seconds=.5))
    delays = []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr(runner.asyncio, "sleep", sleep)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path,
                                             max_retries=1, backoff_seconds=.5))
    assert count == 2 and result.retried == 1
    assert delays == [.5]
    assert lines(result.shards[0])[0]["retry_count"] == 1


def test_orphan_final_checksum_is_preserved_and_never_recollected(tmp_path, monkeypatch):
    async def transport(payload):
        return fixture("openai")
    install_transport(monkeypatch, transport)
    result = asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path))
    result.shards[0].unlink()
    checksum = result.shards[0].with_suffix(".jsonl.sha256")
    preserved = checksum.read_bytes()
    async def forbidden(payload):
        pytest.fail("an orphan final checksum must not cause recollection")
    install_transport(monkeypatch, forbidden)
    with pytest.raises(ValueError, match="checksum|integrity"):
        asyncio.run(runner.run_manifest(manifest(1), [model("openai")], settings(), tmp_path))
    assert checksum.read_bytes() == preserved and not result.shards[0].exists()


def test_orphan_failure_checksum_fails_before_resuming_paid_work(tmp_path, monkeypatch):
    count = 0
    async def interrupted(payload):
        nonlocal count
        count += 1
        if count == 2:
            raise asyncio.CancelledError()
        return fixture("openai")
    install_transport(monkeypatch, interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_manifest(manifest(2), [model("openai")], settings(), tmp_path))
    checkpoint = next(tmp_path.rglob("*.success.jsonl"))
    root = checkpoint.parent.parent.parent
    orphan = root / ".failures" / f"{checkpoint.parent.name}.jsonl.sha256"
    orphan.write_text("0" * 64 + "\n", encoding="ascii")
    async def forbidden(payload):
        pytest.fail("orphan failure shard must be rejected before paid work")
    install_transport(monkeypatch, forbidden)
    with pytest.raises(ValueError, match="checksum|integrity"):
        asyncio.run(runner.run_manifest(manifest(2), [model("openai")], settings(), tmp_path))
    assert orphan.exists()
