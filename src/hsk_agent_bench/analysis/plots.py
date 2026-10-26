"""Deterministic descriptive figures in editable SVG and 300-dpi PNG.

Set frame.attrs['model_order'] to enabled model IDs in frozen configuration
order, or pass model_order explicitly. Ratings use the human_<dimension>
and llm_<dimension> columns produced by join_analysis_tables. Missing
repeats/ratings are labelled unavailable rather than imputed. No plot or
caption infers significance from error-bar overlap.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .aggregate import aggregate_group_means, prompt_model_means
from ..evaluators.human_ratings import score_intrarater


METRICS = (
    "v_in_token", "v_in_type", "v_oov_token", "v_oov_severe", "unknown_token_rate",
    "mattr_50", "mean_sentence_length", "mean_dependency_distance",
    "in_level_matches", "target_level_matches", "above_level_matches",
    "confirmed_violation_rate",
)
ORIGIN_COLORS = {"china": "#0072B2", "global": "#D55E00"}


def _unavailable(ax, message: str, labels: list[str]) -> None:
    ax.set_axis_off()
    ax.text(.5, .6, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.text(.5, .35, " | ".join(labels), ha="center", va="center", transform=ax.transAxes, fontsize=8)


def _save(fig, name: str, output: Path) -> list[Path]:
    paths = [output / f"{name}.svg", output / f"{name}.png"]
    try:
        fig.savefig(paths[0], bbox_inches="tight", metadata={"Date": None})
        fig.savefig(paths[1], dpi=300, bbox_inches="tight", metadata={"Software": "HSK-AgentBench"})
    finally:
        plt.close(fig)
    return paths


def build_all_figures(frame: pd.DataFrame, output_dir: Path, *,
                      model_order: Sequence[str] | None = None) -> list[Path]:
    """Create all six diagnostic figures, with individual identities visible.

    v_in_token is the primary displayed alignment outcome when present.
    The heatmap displays raw means as annotations and within-metric z scores
    as colors, so metrics on different scales do not share raw color units.
    Stability is the mean sample variance across repeated prompt/model cells.
    Agreement panels remain dimension-specific, matching Task 6 semantics.
    """
    order = list(model_order if model_order is not None else frame.attrs.get("model_order", []))
    if ("model_id" not in frame or not order or len(order) != len(set(order))
            or set(order) != set(frame.model_id)):
        raise ValueError("model_order must explicitly list each observed model exactly once")
    available = [name for name in METRICS if name in frame]
    if not available:
        raise ValueError("no supported evaluator metrics available")
    metric = "v_in_token" if "v_in_token" in available else available[0]
    cells = prompt_model_means(frame, metric)
    for column in ("task", "hsk_level"):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"missing {column} metadata")
        if (frame.groupby("prompt_id")[column].nunique() != 1).any():
            raise ValueError(f"inconsistent {column} within prompt_id")
    source = frame.sort_values(["prompt_id", "model_id"] + [c for c in ("run_id", "replicate") if c in frame]).copy()
    origins = cells.drop_duplicates("model_id").set_index("model_id").origin.to_dict()
    labels = [f"{model} ({origins[model]})" for model in order]
    # Consistent origin color plus distinct markers/hatches preserve model identity.
    colors = [ORIGIN_COLORS[origins[model]] for model in order]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "hsk-agent-bench",
        "figure.dpi": 100, "savefig.facecolor": "white",
    }
    with plt.rc_context(style), sns.axes_style("whitegrid"):
        means = {}
        for name in available:
            result = aggregate_group_means(source, name)
            by_model = result.query("scope == 'model'").set_index("model_id")["mean"]
            by_origin = result.query("scope == 'origin'").set_index("origin")["mean"]
            means[name] = [by_model[m] for m in order] + [by_origin.get(o, np.nan) for o in ("china", "global")]
        table = pd.DataFrame(means, index=labels + ["china origin mean", "global origin mean"])
        scaled = ((table - table.mean()) / table.std(ddof=0).replace(0, 1)).fillna(0)
        fig, ax = plt.subplots(figsize=(max(7.5, .95 * len(available)), 4.8))
        sns.heatmap(scaled, annot=table, fmt=".3g", cmap="vlag", center=0, ax=ax,
                    cbar_kws={"label": "Within-metric standardized mean"})
        ax.set_title("Overall model and origin means (annotations: raw units)")
        ax.tick_params(axis="x", rotation=40)
        plt.setp(ax.get_xticklabels(), ha="right")
        fig.tight_layout()
        paths.extend(_save(fig, "overall_heatmap", output))

        metadata = source[["prompt_id", "task", "hsk_level"]].drop_duplicates("prompt_id")
        cells = cells.merge(metadata, on="prompt_id", validate="many_to_one")
        tasks = sorted(cells.task.unique())
        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = .8 / len(order)
        for i, model in enumerate(order):
            values = cells.loc[cells.model_id == model].groupby("task")[metric].mean().reindex(tasks)
            ax.bar(np.arange(len(tasks)) - .4 + width * (i + .5), values, width,
                   label=labels[i], color=colors[i], hatch=("", "//", "..", "xx")[i % 4])
        ax.set(xticks=np.arange(len(tasks)), xticklabels=tasks, ylabel=metric,
               title="Task-level means by individual model")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        paths.extend(_save(fig, "task_bars", output))

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for i, model in enumerate(order):
            values = cells.loc[cells.model_id == model].groupby("hsk_level")[metric].mean().sort_index()
            ax.plot(values.index, values, label=labels[i], color=colors[i], marker=markers[i % len(markers)],
                    linestyle="-" if i % 2 == 0 else "--")
        ax.set(xlabel="HSK level", ylabel=metric, title="HSK difficulty profile",
               xticks=sorted(cells.hsk_level.unique()))
        ax.legend(fontsize=8)
        fig.tight_layout()
        paths.extend(_save(fig, "hsk_level_lines", output))

        fig, ax = plt.subplots(figsize=(9, 4.8))
        spill = "v_oov_token"
        if spill in source:
            spill_cells = prompt_model_means(source, spill)
            distributions = [spill_cells.loc[spill_cells.model_id == model, spill].to_numpy() for model in order]
            boxes = ax.boxplot(distributions, tick_labels=labels, patch_artist=True, showfliers=True)
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(.5)
            ax.set(ylabel=spill, title="Spillover distribution across prompt/model means")
            ax.tick_params(axis="x", rotation=15)
        else:
            _unavailable(ax, "Spillover unavailable: v_oov_token was not supplied", labels)
        fig.tight_layout()
        paths.extend(_save(fig, "spillover_distribution", output))

        fig, ax = plt.subplots(figsize=(9, 4.8))
        repeats = source.groupby(["prompt_id", "model_id"])[metric].agg(["count", "var"])
        variances = repeats.loc[repeats["count"] >= 2, "var"].groupby("model_id").mean()
        if not variances.empty:
            ax.bar(labels, variances.reindex(order), color=colors)
            ax.set(ylabel=f"Within-prompt sample variance: {metric}",
                   title="Model stability: mean variance of repeated outputs")
            ax.tick_params(axis="x", rotation=15)
        else:
            _unavailable(ax, "Stability unavailable: no repeated prompt/model cells", labels)
        fig.tight_layout()
        paths.extend(_save(fig, "model_stability", output))

        dimensions = sorted(name.removeprefix("human_") for name in source
                            if name.startswith("human_") and f"llm_{name.removeprefix('human_')}" in source)
        if dimensions:
            fig, axes = plt.subplots(1, len(dimensions), figsize=(6 * len(dimensions), 5), squeeze=False)
            for ax, dimension in zip(axes[0], dimensions):
                human, judge = f"human_{dimension}", f"llm_{dimension}"
                paired = source.dropna(subset=[human, judge])
                if paired.empty:
                    _unavailable(ax, f"Agreement unavailable: {dimension}", labels)
                    continue
                for i, model in enumerate(order):
                    subset = paired.loc[paired.model_id == model]
                    ax.scatter(subset[human], subset[judge], label=labels[i], color=colors[i],
                               marker=markers[i % len(markers)], s=45, alpha=.65)
                if "generation_id" not in paired or paired.generation_id.duplicated().any():
                    raise ValueError("agreement requires unique generation_id rows")
                indexed = paired.set_index("generation_id")
                report = score_intrarater(indexed[human], indexed[judge])
                rho = "undefined" if report.spearman is None else f"{report.spearman:.2f}"
                kappa = "undefined" if report.weighted_kappa is None else f"{report.weighted_kappa:.2f}"
                ax.plot([1, 5], [1, 5], color=".6", linestyle=":", zorder=0)
                ax.set(xlabel="Human score", ylabel="Judge score", xticks=range(1, 6), yticks=range(1, 6),
                       xlim=(.8, 5.2), ylim=(.8, 5.2),
                       title=f"{dimension}\nn={report.n_pairs}; Spearman={rho}; weighted kappa={kappa}")
                ax.legend(fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            _unavailable(ax, "Judge-human agreement unavailable: no matched rating dimensions", labels)
        fig.tight_layout()
        paths.extend(_save(fig, "judge_human_agreement", output))
    return paths
