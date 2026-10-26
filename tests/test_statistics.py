from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsk_agent_bench.analysis.aggregate import aggregate_group_means, join_analysis_tables
from hsk_agent_bench.analysis.hypothesis_tests import (
    fit_mixed_effects, holm_adjust, paired_origin_test, paired_wilcoxon,
)


@pytest.fixture
def metrics():
    return pd.read_csv(Path(__file__).parent / "fixtures/synthetic_metrics.csv")


def group_rows(differences):
    return pd.DataFrame([
        {"prompt_id": f"p{i}", "origin": origin, "v_in_token": .4 + (d if origin == "china" else 0)}
        for i, d in enumerate(differences) for origin in ("china", "global")
    ])


def test_models_are_equal_weighted_after_repeats_and_prompts():
    frame = pd.DataFrame([
        {"generation_id": f"g{i}", "prompt_id": p, "model_id": m, "origin": "china",
         "replicate": r, "v_in_token": value}
        for i, (p, m, r, value) in enumerate([
            ("p1", "qwen", 0, 0), ("p1", "qwen", 1, 0), ("p2", "qwen", 0, 1),
            ("p1", "deepseek", 0, 1), ("p2", "deepseek", 0, 1)])
    ])
    result = aggregate_group_means(frame, "v_in_token")
    assert result.query("scope == 'model' and model_id == 'qwen'").iloc[0]["mean"] == .5
    assert result.query("scope == 'origin'").iloc[0]["mean"] == .75
    assert set(result.scope) == {"model", "origin"}


def test_known_shift_and_prompt_alignment_are_order_invariant(metrics):
    first = paired_origin_test(metrics, "v_in_token", 1000, 7)
    shuffled = paired_origin_test(metrics.sample(frac=1, random_state=19), "v_in_token", 1000, 7)
    assert first == shuffled
    assert first.n_pairs == 8
    assert first.median_difference == pytest.approx(.2)
    assert first.rank_biserial == 1
    assert first.ci_low == pytest.approx(.2)
    assert first.ci_high == pytest.approx(.2)
    assert first.pvalue == pytest.approx(2 / 256)


def test_monte_carlo_is_seeded_and_two_sided():
    frame = group_rows([.1] * 25)
    result = paired_origin_test(frame, "v_in_token", 499, 20)
    assert result == paired_origin_test(frame.sample(frac=1, random_state=5), "v_in_token", 499, 20)
    reversed_frame = frame.assign(v_in_token=-frame.v_in_token)
    assert paired_origin_test(reversed_frame, "v_in_token", 499, 20).pvalue == result.pvalue
    assert result.pvalue == pytest.approx(1 / 500)


@pytest.mark.parametrize("bad", ["duplicate", "unmatched", "nan", "origin", "empty"])
def test_invalid_group_pairs_are_rejected(bad):
    frame = group_rows([.1, .2])
    if bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif bad == "unmatched":
        frame = frame.iloc[1:]
    elif bad == "nan":
        frame.loc[0, "v_in_token"] = np.nan
    elif bad == "origin":
        frame.loc[0, "origin"] = "other"
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        paired_origin_test(frame, "v_in_token", 100, 0)


def test_duplicate_generations_and_missing_models_are_rejected(metrics):
    with pytest.raises(ValueError, match="duplicate"):
        paired_origin_test(pd.concat([metrics, metrics.iloc[:1]]), "v_in_token", 100, 0)
    with pytest.raises(ValueError, match="model"):
        paired_origin_test(metrics.iloc[1:], "v_in_token", 100, 0)


def test_rank_biserial_uses_signed_ranks_and_zero_differences():
    result = paired_wilcoxon(group_rows([.1, -.2, .3, 0]), "v_in_token")
    assert result.rank_biserial == pytest.approx(1 / 3)
    assert result.median_difference == pytest.approx(.05)
    zero = paired_wilcoxon(group_rows([0, 0]), "v_in_token")
    assert zero.pvalue == 1
    assert zero.rank_biserial == 0
    assert (zero.ci_low, zero.ci_high) == (0, 0)


def test_tiny_nonzero_differences_are_not_silently_rounded_to_zero():
    frame = pd.DataFrame([
        {"prompt_id": f"p{i}", "origin": origin, "v_in_token": 1e-16 if origin == "china" else 0.}
        for i in range(4) for origin in ("china", "global")
    ])
    result = paired_origin_test(frame, "v_in_token", 16, 7)
    assert result.median_difference == 1e-16
    assert result.rank_biserial == 1
    assert result.pvalue == .125


def test_holm_corrects_only_the_declared_family():
    result = paired_wilcoxon(group_rows([.1, .2, .3]), "v_in_token")
    results = [replace(result, pvalue=p) for p in [.01, .04, .03]]
    adjusted = holm_adjust(results, family="lexical")
    assert [r.p_adjusted for r in adjusted] == pytest.approx([.03, .06, .06])
    assert all(r.family == "lexical" for r in adjusted)
    assert all(r.p_adjusted is None for r in results)
    with pytest.raises(ValueError, match="famil"):
        holm_adjust([replace(result, family="quality")], "lexical")


def test_join_preserves_metric_names_and_task6_rating_scores(metrics):
    generations = metrics.drop(columns=["v_in_token", "v_oov_token"])
    scores = metrics[["generation_id", "prompt_id", "v_in_token", "v_oov_token"]]
    metric_records = scores[["generation_id", "prompt_id"]].copy()
    metric_records["metrics"] = scores[["v_in_token", "v_oov_token"]].to_dict("records")
    ratings = pd.DataFrame([
        {"generation_id": metrics.iloc[0].generation_id, "prompt_id": "p1", "rater_id": "expert",
         "rater_type": "human", "scores": {"naturalness": 4}},
        {"generation_id": metrics.iloc[0].generation_id, "prompt_id": "p1", "rater_id": "judge",
         "rater_type": "llm", "scores": {"naturalness": 3}},
    ])
    joined = join_analysis_tables(generations, metric_records, ratings)
    assert joined.iloc[0].v_in_token == .8
    assert joined.iloc[0].human_naturalness == 4
    assert joined.iloc[0].llm_naturalness == 3
    assert pd.isna(joined.iloc[1].human_naturalness)
    with pytest.raises(ValueError, match="prompt"):
        join_analysis_tables(generations, metric_records.assign(prompt_id="wrong"))
    with pytest.raises(ValueError, match="duplicate"):
        join_analysis_tables(generations, pd.concat([metric_records, metric_records.iloc[:1]]))


def test_mixed_effects_success_and_explicit_failure_diagnostics():
    rng = np.random.default_rng(22)
    frame = pd.DataFrame([
        {"prompt_id": f"p{i}", "origin": origin, "score": i / 10 + (1 if origin == "china" else 0) + rng.normal(0, .1)}
        for i in range(20) for origin in ("china", "global")
    ])
    fitted = fit_mixed_effects(frame, "score ~ C(origin)", "prompt_id")
    assert fitted.success
    assert fitted.n_obs == 40
    assert fitted.estimates["C(origin)[T.global]"] == pytest.approx(-1, abs=.15)
    failed = fit_mixed_effects(frame, "missing ~ C(origin)", "prompt_id")
    assert not failed.success
    assert failed.diagnostics
    assert failed.method == "mixed_effects"
    deficient = fit_mixed_effects(frame.assign(copy=frame.score), "score ~ copy + score", "prompt_id")
    assert not deficient.success
    assert deficient.diagnostics
