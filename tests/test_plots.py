from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image
import pytest

from hsk_agent_bench.analysis.plots import build_all_figures


@pytest.fixture
def plot_frame():
    frame = pd.read_csv(Path(__file__).parent / "fixtures/synthetic_metrics.csv")
    repeats = frame.assign(generation_id=frame.generation_id + "-repeat", replicate=1,
                           v_in_token=frame.v_in_token - .04)
    frame = pd.concat([frame, repeats], ignore_index=True)
    frame["human_naturalness"] = [1, 2, 3, 4] * 16
    frame["llm_naturalness"] = [2, 2, 4, 4] * 16
    frame["human_pedagogical_quality"] = [2, 3, 4, 5] * 16
    frame["llm_pedagogical_quality"] = [2, 4, 4, 5] * 16
    # Caller copies enabled model IDs in frozen registry order.
    frame.attrs["model_order"] = ["GPT", "Gemini", "Qwen", "DeepSeek"]
    return frame


def svg_text(path):
    root = ET.parse(path).getroot()
    return " ".join(root.itertext())


def test_all_six_figures_name_models_origins_and_export_at_300dpi(plot_frame, tmp_path):
    paths = build_all_figures(plot_frame, tmp_path)
    expected = {"overall_heatmap", "task_bars", "hsk_level_lines", "spillover_distribution",
                "model_stability", "judge_human_agreement"}
    assert {path.stem for path in paths} == expected
    assert len(paths) == 12
    assert all(path.stat().st_size > 1000 for path in paths)
    for path in paths:
        if path.suffix == ".svg":
            text = svg_text(path)
            for name in ("GPT", "Gemini", "Qwen", "DeepSeek", "china", "global"):
                assert name in text
        else:
            with Image.open(path) as image:
                assert image.info["dpi"] == pytest.approx((300, 300), abs=.01)
    heatmap = svg_text(tmp_path / "overall_heatmap.svg")
    assert heatmap.index("GPT (global)") < heatmap.index("Gemini (global)") < heatmap.index("Qwen (china)")
    assert "v_in_token" in heatmap
    assert "v_oov_token" in svg_text(tmp_path / "spillover_distribution.svg")
    stability = svg_text(tmp_path / "model_stability.svg")
    assert "Within-prompt" in stability
    agreement = svg_text(tmp_path / "judge_human_agreement.svg")
    assert "naturalness" in agreement
    assert "pedagogical_quality" in agreement


def test_figure_output_is_deterministic_under_input_shuffling(plot_frame, tmp_path):
    first = build_all_figures(plot_frame, tmp_path / "a")
    second = build_all_figures(plot_frame.sample(frac=1, random_state=3), tmp_path / "b")
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


def test_missing_repeats_or_ratings_are_labelled_unavailable(plot_frame, tmp_path):
    frame = plot_frame.query("replicate == 0").drop(columns=[
        c for c in plot_frame if c.startswith(("human_", "llm_"))])
    build_all_figures(frame, tmp_path)
    assert "unavailable" in svg_text(tmp_path / "model_stability.svg").lower()
    assert "unavailable" in svg_text(tmp_path / "judge_human_agreement.svg").lower()


@pytest.mark.parametrize("order", [None, ["GPT"], ["GPT", "GPT", "Qwen", "DeepSeek", "Gemini"]])
def test_configuration_must_explicitly_order_all_models(plot_frame, tmp_path, order):
    plot_frame.attrs.clear()
    if order is not None:
        plot_frame.attrs["model_order"] = order
    with pytest.raises(ValueError, match="model_order"):
        build_all_figures(plot_frame, tmp_path)
