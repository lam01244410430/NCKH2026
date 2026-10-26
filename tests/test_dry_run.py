import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from hsk_agent_bench.cli import app
from hsk_agent_bench.io import verify_shard


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_crosses_artifact_boundary_without_network_or_overwrite(tmp_path, monkeypatch):
    import hsk_agent_bench.generators as generators

    def forbidden(*args, **kwargs):
        raise AssertionError("the end-to-end dry run attempted network client construction")

    monkeypatch.setattr(generators, "make_generator", forbidden)
    output = tmp_path / "dry-run"
    args = [
        "dry-run", "--config", str(ROOT / "configs/experiment.yaml"),
        "--device", str(ROOT / "configs/devices/pc.yaml"),
        "--output-dir", str(output),
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    manifest = Path(summary["verified_manifest_shard"])
    assert manifest.exists() and verify_shard(manifest)
    raw = Path(summary["verified_fixture_shard"])
    assert raw.exists() and verify_shard(raw)
    assert summary["providers"] == ["openai", "gemini", "qwen", "deepseek"]
    metrics = pd.read_csv(summary["metric_table"])
    assert set(metrics.model_id) == {
        "FREEZE_OPENAI_MODEL_ID", "FREEZE_GEMINI_MODEL_ID",
        "FREEZE_QWEN_MODEL_ID", "FREEZE_DEEPSEEK_MODEL_ID",
    }
    assert {"v_in_token", "mean_sentence_length", "mean_dependency_distance"} <= set(metrics)
    assert {"text", "material_text"} <= set(metrics)
    assert all(isinstance(json.loads(raw), dict) for raw in metrics.text)
    assert all(not projected.lstrip().startswith("{") for projected in metrics.material_text)
    prepared, prepared_key = tmp_path / "prepared-from-analysis.csv", tmp_path / "prepared-key.csv"
    prepared_result = CliRunner().invoke(app, [
        "prepare-ratings", "--input", str(summary["metric_table"]),
        "--output", str(prepared), "--key-output", str(prepared_key),
        "--device", str(ROOT / "configs/devices/pc.yaml"), "--fraction", "1",
    ])
    assert prepared_result.exit_code == 0, prepared_result.output
    prepared_sheet = pd.read_csv(prepared)
    prepared_mapping = pd.read_csv(prepared_key)
    assert set(prepared_sheet.text) == set(metrics.material_text)
    assert all(isinstance(json.loads(raw), dict) for raw in prepared_mapping.text)
    assert set(prepared_mapping.material_text) == set(prepared_sheet.text)
    statistics = pd.read_csv(summary["statistical_table"])
    assert {"pvalue", "p_adjusted", "rank_biserial", "ci_low", "ci_high"} <= set(statistics)
    assert Path(summary["rating_sheet"]).exists()
    coordinator = pd.read_csv(summary["coordinator_key"])
    assert all(isinstance(json.loads(raw), dict) for raw in coordinator.text)
    assert set(coordinator.material_text) == set(pd.read_csv(summary["rating_sheet"]).text)
    rating_columns = set(pd.read_csv(summary["rating_sheet"]).columns)
    assert not ({"provider", "model_id", "origin"} & rating_columns)
    assert all(Path(path).exists() for path in summary["figures"])

    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    second = CliRunner().invoke(app, args)
    assert second.exit_code != 0
    assert "already exists" in second.output
    assert before == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
