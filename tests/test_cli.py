import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import hsk_agent_bench.generators as generators
import hsk_agent_bench.cli as cli
from hsk_agent_bench.cli import app
from hsk_agent_bench.config import load_experiment_config
from hsk_agent_bench.io import write_jsonl_shard_atomic
from hsk_agent_bench.runner import RunSummary
from hsk_agent_bench.schemas import (
    ChatMessage, GenerationParameters, GenerationRecord, MetricRecord, TokenUsage,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
COMMANDS = (
    "validate-data", "build-prompts", "generate", "evaluate",
    "prepare-ratings", "analyze", "plot", "dry-run",
)


def analysis_frame():
    frame = pd.read_csv(ROOT / "tests/fixtures/synthetic_metrics.csv")
    frame["model_id"] = frame.model_id.map({
        "GPT": "FREEZE_OPENAI_MODEL_ID", "Gemini": "FREEZE_GEMINI_MODEL_ID",
        "Qwen": "FREEZE_QWEN_MODEL_ID", "DeepSeek": "FREEZE_DEEPSEEK_MODEL_ID",
    })
    frame["run_id"] = "primary"
    return frame


@pytest.mark.parametrize("command", COMMANDS)
def test_each_pipeline_stage_has_help(command):
    result = RUNNER.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_generation_stage_rejects_pc_profile_before_reading_inputs(tmp_path):
    result = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"),
        "--protocol", "primary",
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


def test_validate_data_rejects_vocabulary_without_provenance(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text('[{"word":"我","min_hsk_level":1}]', encoding="utf-8")
    summary = tmp_path / "summary.json"
    result = RUNNER.invoke(app, [
        "validate-data", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"),
        "--vocab", str(vocab),
        "--grammar", str(ROOT / "data/hsk_grammar_points/schema.example.json"),
        "--summary", str(summary),
    ])
    assert result.exit_code != 0
    assert "invalid vocabulary source" in result.output
    assert "standard" in result.output and "source" in result.output and "license" in result.output
    assert not summary.exists()


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
        "--protocol", "primary",
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
        "--protocol", "primary",
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "raw"),
    ])
    assert result.exit_code != 0
    assert "freeze exact model ids" in result.output.lower()
    assert not (tmp_path / "raw").exists()


def test_generate_protocol_selects_stochastic_settings_for_preview_and_live(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    build = RUNNER.invoke(app, [
        "build-prompts", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"), "--output", str(manifest),
    ])
    assert build.exit_code == 0, build.output
    preview = tmp_path / "stochastic.jsonl"
    result = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"), "--protocol", "stochastic",
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "raw"),
        "--preview-output", str(preview), "--dry-run", "--run-id", "stochastic-pilot",
    ])
    assert result.exit_code == 0, result.output
    requests = [json.loads(line) for line in preview.read_text(encoding="utf-8").splitlines()]
    assert len(requests) == 36 * 4 * 3
    assert {row["parameters"]["temperature"] for row in requests} == {.7}
    assert {row["parameters"]["top_p"] for row in requests} == {.9}
    assert {row["replicate"] for row in requests} == {0, 1, 2}

    experiment = load_experiment_config(ROOT / "configs/experiment.yaml")
    experiment.models = [model.model_copy(update={"model_id": f"frozen-{model.provider}"})
                         for model in experiment.models]
    captured = {}
    async def capture(manifest, models, run, output_dir, **kwargs):
        captured.update(manifest=manifest, run=run, kwargs=kwargs)
        return RunSummary(run_id=kwargs["run_id"], output_dir=Path(output_dir) / kwargs["run_id"])
    monkeypatch.setattr(cli, "load_experiment_config", lambda path: experiment.model_copy(deep=True))
    monkeypatch.setattr(cli, "run_manifest", capture)
    live = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"), "--protocol", "stochastic",
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "live"),
        "--run-id", "stochastic-pilot",
    ])
    assert live.exit_code == 0, live.output
    assert captured["run"].model_dump() == experiment.stochastic.model_dump()
    assert len(captured["manifest"]) == 144

    invalid = RUNNER.invoke(app, [
        "generate", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/laptop.yaml"), "--protocol", "invented",
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "invalid"),
    ])
    assert invalid.exit_code != 0


def test_analyze_adjusts_multiple_pvalues_within_declared_nonlexical_family(tmp_path):
    source, output = tmp_path / "analysis.csv", tmp_path / "statistics.csv"
    analysis_frame().to_csv(source, index=False)
    result = RUNNER.invoke(app, [
        "analyze", "--input", str(source), "--output", str(output),
        "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"),
        "--run-id", "primary", "--replicate", "0",
        "--metric", "v_in_token", "--metric", "v_oov_token", "--family", "quality",
    ])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(output)
    assert table.metric.tolist() == ["v_in_token", "v_oov_token"]
    assert table.family.tolist() == ["quality", "quality"]
    assert table.pvalue.tolist() == pytest.approx([.0078125, .0078125])
    assert table.p_adjusted.tolist() == pytest.approx([.015625, .015625])


def test_evaluate_uses_configured_lazy_cpu_parser_for_mdd(tmp_path, monkeypatch):
    module = tmp_path / "fixture_parser.py"
    module.write_text(
        "from hsk_agent_bench.evaluators.syntax import DependencyToken, ParsedDocument, ParsedSentence\n"
        "class Backend:\n"
        "    def parse(self, text):\n"
        "        return ParsedDocument(sentences=[ParsedSentence(tokens=[\n"
        "            DependencyToken(index=1, text='我', head=2, relation='nsubj'),\n"
        "            DependencyToken(index=2, text='学习', head=0, relation='root')])])\n"
        "def load(device):\n"
        "    assert device == 'cpu'\n"
        "    return Backend()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    parser = tmp_path / "parser.yaml"
    parser.write_text("name: fixture\nversion: '1'\nfactory: fixture_parser:load\ndevice: cpu\n", encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps([
        {"word": "我", "min_hsk_level": 1, "standard": "test", "source": "test-fixture",
         "license": "test-only", "notes": None},
        {"word": "学习", "min_hsk_level": 2, "standard": "test", "source": "test-fixture",
         "license": "test-only", "notes": None},
    ], ensure_ascii=False), encoding="utf-8")
    request = dict(
        generation_id="g1", prompt_id="p1", task="reading", hsk_level=3, family="GPT",
        origin="global", provider="openai", model_id="frozen-test-model", prompt_hash="0" * 64,
        messages=[ChatMessage(role="user", content="write")],
        parameters=GenerationParameters(temperature=0, top_p=1, max_output_tokens=20),
        run_id="primary", replicate=0,
    )
    raw = tmp_path / "raw.jsonl"
    write_jsonl_shard_atomic([GenerationRecord(
        **request, text=json.dumps({"title": "学习", "passage": "我学习。", "declared_hsk_level": 3,
                                   "used_required_vocabulary": [], "used_required_grammar": []},
                                  ensure_ascii=False), raw_response={"fixture": True},
        requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc), latency_ms=0,
        usage=TokenUsage(), retry_count=0, seed_applied=False,
    )], raw)
    output = tmp_path / "metrics.jsonl"
    result = RUNNER.invoke(app, [
        "evaluate", "--input", str(raw), "--output", str(output), "--vocab", str(vocab),
        "--parser-config", str(parser), "--device", str(ROOT / "configs/devices/pc.yaml"),
    ])
    assert result.exit_code == 0, result.output
    metric = MetricRecord.model_validate_json(output.read_text(encoding="utf-8").strip())
    assert metric.metrics["mean_dependency_distance"] == 1
    assert metric.metadata["syntax"]["parser_name"] == "fixture"
    assert metric.metadata["syntax"]["parser_device"] == "cpu"


def test_plot_preflights_all_outputs_before_rendering(tmp_path):
    source = tmp_path / "analysis.csv"
    analysis_frame().to_csv(source, index=False)
    output = tmp_path / "figures"
    output.mkdir()
    summary = output / "run-summary.json"
    summary.write_text("preserve-summary", encoding="utf-8")
    sentinel = output / "overall_heatmap.svg"
    sentinel.write_bytes(b"preserve-figure")
    result = RUNNER.invoke(app, [
        "plot", "--input", str(source), "--output-dir", str(output),
        "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"),
        "--run-id", "primary", "--replicate", "0",
    ])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert summary.read_text(encoding="utf-8") == "preserve-summary"
    assert sentinel.read_bytes() == b"preserve-figure"
    assert not (output / "overall_heatmap.png").exists()


def test_plot_explicit_stability_mode_keeps_comparable_repeats(tmp_path):
    primary = analysis_frame()
    repeated = primary.assign(generation_id=primary.generation_id + "-repeat", replicate=1,
                              v_in_token=primary.v_in_token - .03)
    source = tmp_path / "repeated.csv"
    pd.concat([primary, repeated], ignore_index=True).to_csv(source, index=False)
    output = tmp_path / "figures"
    result = RUNNER.invoke(app, [
        "plot", "--input", str(source), "--output-dir", str(output),
        "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"), "--run-id", "primary",
        "--include-repeats-for-stability",
    ])
    assert result.exit_code == 0, result.output
    stability = (output / "model_stability.svg").read_text(encoding="utf-8")
    assert "Within-prompt sample variance" in stability
    summary = json.loads((output / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["include_repeats_for_stability"] is True
