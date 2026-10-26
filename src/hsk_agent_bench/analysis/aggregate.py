"""Strict ID joins and equal-model descriptive aggregation.

Callers must select the intended run/condition before analysis. Repeated
generations within a prompt/model cell are averaged first, then prompts
within a model, then individual models within an origin. Origin summaries
describe the evaluated models, not a population of model developers.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("analysis table cannot be empty")
    if frame[columns].isna().any().any():
        raise ValueError("required analysis fields cannot be missing")


def _metric_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    _require(frame, ["prompt_id", "origin", metric])
    if not frame.origin.isin(["china", "global"]).all():
        raise ValueError("origin must be china or global")
    if not pd.api.types.is_numeric_dtype(frame[metric]) or not np.isfinite(frame[metric]).all():
        raise ValueError(f"metric {metric} must contain finite numeric values")
    return frame.copy()


def prompt_model_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Validate raw cell identity and reduce repeated outputs within cells."""
    source = _metric_frame(frame, metric)
    _require(source, ["model_id"])
    if (source.groupby("model_id").origin.nunique() != 1).any():
        raise ValueError("each model must have exactly one origin")
    if "generation_id" in source:
        _require(source, ["generation_id"])
        if source.generation_id.duplicated().any():
            raise ValueError("duplicate generation_id")
    keys = ["prompt_id", "model_id"] + [k for k in ("run_id", "replicate") if k in source]
    _require(source, keys)
    if source.duplicated(keys).any():
        raise ValueError("duplicate prompt/model/run/replicate cell; select or identify repeats explicitly")
    return source.groupby(["prompt_id", "model_id", "origin"], sort=True, as_index=False)[metric].mean()


def aggregate_group_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return model rows first and equal-weight origin rows with count metadata.

    Columns: scope, origin, model_id (null for origins), metric, mean,
    n_prompts and n_models. Missing metrics are rejected; select a declared
    scored subset explicitly rather than silently dropping observations.
    """
    cells = prompt_model_means(frame, metric)
    rows = []
    for (origin, model), cell in cells.groupby(["origin", "model_id"], sort=True):
        rows.append(dict(scope="model", origin=origin, model_id=model, metric=metric,
                         mean=float(cell[metric].mean()), n_prompts=len(cell), n_models=1))
    models = pd.DataFrame(rows)
    for origin, group in models.groupby("origin", sort=True):
        rows.append(dict(scope="origin", origin=origin, model_id=None, metric=metric,
                         mean=float(group["mean"].mean()),
                         n_prompts=int(cells.loc[cells.origin == origin, "prompt_id"].nunique()),
                         n_models=len(group)))
    return pd.DataFrame(rows)


def paired_origin_values(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return China/global columns aligned exclusively by prompt_id.

    Accept either raw model outputs or one pre-aggregated row per
    origin/prompt. Raw inputs require every selected model at every prompt;
    model dropout cannot silently change the planned origin contrast.
    """
    if "model_id" in frame:
        cells = prompt_model_means(frame, metric)
        roster = cells.groupby("origin").model_id.nunique()
        counts = cells.groupby(["prompt_id", "origin"]).model_id.nunique()
        if any(count != roster[origin] for (_, origin), count in counts.items()):
            raise ValueError("unmatched model roster across prompts")
        source = cells.groupby(["prompt_id", "origin"], sort=True, as_index=False)[metric].mean()
    else:
        source = _metric_frame(frame, metric)
    if source.duplicated(["prompt_id", "origin"]).any():
        raise ValueError("duplicate origin/prompt rows")
    paired = source.pivot(index="prompt_id", columns="origin", values=metric)
    if set(paired.columns) != {"china", "global"} or paired.isna().any().any():
        raise ValueError("unmatched China/global prompt_id pairs")
    return paired.sort_index()[["china", "global"]]


def join_analysis_tables(generations: pd.DataFrame, metrics: pd.DataFrame,
                         ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join MetricRecord/RatingRecord dumps on both durable identifiers.

    Metric dictionaries retain evaluator names verbatim. Rating dimensions
    become human_<dimension>/llm_<dimension>; multiple ratings of the same
    type/item/dimension are rejected, requiring explicit rater selection
    before joining. Unrated generations remain present with missing scores.
    """
    ids = ["generation_id", "prompt_id"]
    _require(generations, ids)
    if generations.generation_id.duplicated().any():
        raise ValueError("duplicate generation_id")
    joined = generations.copy().set_index(ids)
    known = set(joined.index)
    values: dict[str, dict[tuple, float]] = {}
    for table, kind in ((metrics, "metrics"), (ratings, "scores")):
        if table is None or table.empty:
            continue
        _require(table, ids + [kind] + (["rater_type"] if kind == "scores" else []))
        for row in table.to_dict("records"):
            key = (row["generation_id"], row["prompt_id"])
            if key not in known:
                raise ValueError("unmatched generation_id/prompt_id in analysis join")
            payload = row[kind]
            if not isinstance(payload, Mapping):
                raise ValueError(f"{kind} must be dictionaries")
            if kind == "scores" and row["rater_type"] not in ("human", "llm"):
                raise ValueError("rater_type must be human or llm")
            for name, value in payload.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("metric/dimension names must be nonempty strings")
                column = name if kind == "metrics" else f"{row['rater_type']}_{name}"
                if column in generations.columns:
                    raise ValueError(f"metric/rating column collision: {column}")
                bucket = values.setdefault(column, {})
                if key in bucket:
                    raise ValueError(f"duplicate metric/rating for {column}; select raters explicitly")
                bucket[key] = value
    for column, data in values.items():
        joined[column] = pd.Series(data)
    return joined.reset_index()
