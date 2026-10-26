import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hsk_agent_bench.generators as generators
import hsk_agent_bench.cli as cli
from hsk_agent_bench.cli import app


ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
COMMANDS = (
    "validate-data", "build-prompts", "generate", "evaluate",
    "prepare-ratings", "analyze", "plot", "dry-run",
)


@pytest.mark.parametrize("command", COMMANDS)
def test_each_pipeline_stage_has_help(command):
    result = RUNNER.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_generation_stage_rejects_pc_profile_before_reading_inputs(tmp_path):
    result = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"),
        "--manifest", str(tmp_path / "missing.jsonl"),
        "--output-dir", str(tmp_path / "raw"),
    ])
    assert result.exit_code != 0
    assert "device profile 'pc' does not allow stage 'generate'" in result.output


def test_validate_data_reports_each_missing_source(tmp_path):
    result = RUNNER.invoke(app, [
        "validate-data", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--vocab", str(tmp_path / "vocab.json"),
        "--grammar", str(tmp_path / "grammar.json"),
        "--summary", str(tmp_path / "summary.json"),
    ])
    assert result.exit_code != 0
    assert "missing vocabulary source" in result.output
    assert "missing grammar source" in result.output


def test_generate_dry_run_renders_requests_without_constructing_clients(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    build = RUNNER.invoke(app, [
        "build-prompts", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--output", str(manifest),
    ])
    assert build.exit_code == 0, build.output

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not instantiate a provider client")

    monkeypatch.setattr(generators, "make_generator", forbidden)
    preview = tmp_path / "requests.jsonl"
    result = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "raw"),
        "--preview-output", str(preview), "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    requests = [json.loads(line) for line in preview.read_text(encoding="utf-8").splitlines()]
    assert len(requests) == 144 * 4 * 2
    assert {row["provider"] for row in requests} == {"openai", "gemini", "qwen", "deepseek"}
    assert all(row["messages"] and row["prompt_hash"] for row in requests)
    assert not (tmp_path / "raw").exists()
    summary = json.loads(preview.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "generate-preview"
    assert summary["requests"] == len(requests)


def test_live_generation_rejects_unfrozen_model_ids_before_network(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    build = RUNNER.invoke(app, [
        "build-prompts", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--output", str(manifest),
    ])
    assert build.exit_code == 0, build.output

    async def forbidden(*args, **kwargs):
        pytest.fail("placeholder model IDs must fail before generation starts")

    monkeypatch.setattr(cli, "run_manifest", forbidden)
    result = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "raw"),
    ])
    assert result.exit_code != 0
    assert "freeze exact model ids" in result.output.lower()
    assert not (tmp_path / "raw").exists()
