import pandas as pd
import pytest

from hsk_agent_bench.evaluators.human_ratings import (
    blind_rating_sheet, sample_for_human_rating, score_intrarater,
)


def records():
    return pd.DataFrame([
        dict(generation_id=f"g-{model}-{task}-{level}-{rep}",
             prompt_id=f"p-{task}-{level}-{rep}", model_id=model,
             provider="secret-provider", origin="secret-origin", family="secret-family",
             parameters={"seed": 44}, config={"private": True}, raw_response={"model": model},
             task=task, hsk_level=level, text=f"学习材料 {rep}")
        for model in ("model-a", "model-b")
        for task in ("reading", "grammar_explanation", "exam_generation")
        for level in (3, 4, 5, 6) for rep in range(8)
    ])


def test_quarter_sample_preserves_each_joint_stratum_and_is_deterministic():
    source = records()
    sample = sample_for_human_rating(source, fraction=.25, seed=12)
    assert len(sample) == 48
    assert sample.groupby(["model_id", "task", "hsk_level"]).size().tolist() == [2] * 24
    pd.testing.assert_frame_equal(sample, sample_for_human_rating(source, .25, 12))
    pd.testing.assert_frame_equal(sample, sample_for_human_rating(source.iloc[::-1], .25, 12))
    assert set(sample.generation_id) != set(sample_for_human_rating(source, .25, 13).generation_id)
    assert len(source) == 192


def test_sparse_sampling_has_exact_size_and_balances_equal_cells():
    source = records().groupby(["model_id", "task", "hsk_level"]).head(3)
    sample = sample_for_human_rating(source, .25, 9)
    counts = sample.groupby(["model_id", "task", "hsk_level"]).size()
    assert len(sample) == 18
    assert counts.max() == 1


def test_blinded_sheet_whitelists_fields_and_keeps_separate_joinable_key():
    source = records()
    sheet, key = blind_rating_sheet(source, seed=44)
    assert set(sheet.columns) == {"rating_id", "task", "hsk_level", "text"}
    assert sheet.rating_id.is_unique
    restored = sheet.merge(key.mapping, on="rating_id", validate="one_to_one")
    assert set(restored.generation_id) == set(source.generation_id)
    assert all(a != b for a, b in zip(restored.prompt_id, restored.prompt_id.iloc[1:]))
    pd.testing.assert_frame_equal(sheet, blind_rating_sheet(source, 44)[0])
    assert not sheet.equals(blind_rating_sheet(source, 45)[0])


def test_impossible_nonadjacent_schedule_is_rejected():
    source = records().iloc[:4].copy()
    source["prompt_id"] = ["same", "same", "same", "other"]
    with pytest.raises(ValueError, match="adjacent"):
        blind_rating_sheet(source, 12)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan")])
def test_invalid_sample_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="fraction"):
        sample_for_human_rating(records(), fraction, 1)


def test_duplicate_generation_ids_cannot_produce_ambiguous_key():
    source = records().iloc[:2].copy()
    source["generation_id"] = "duplicate"
    with pytest.raises(ValueError, match="unique"):
        blind_rating_sheet(source, 3)


def test_intrarater_aligns_ids_and_reports_missing_pairs():
    report = score_intrarater(pd.Series([1, 2, 3, 4, 5, None], index=list("abcdef")),
                             pd.Series([5, 4, 3, 2, 1, 2], index=list("edcbag")))
    assert report.n_pairs == 5
    assert report.n_missing == 2
    assert report.spearman == pytest.approx(1)
    assert report.weighted_kappa == pytest.approx(1)
    assert report.exact_agreement == 1
    assert report.adjacent_agreement == 1


def test_intrarater_ordinal_metrics_have_hand_checked_values():
    report = score_intrarater(pd.Series([1, 2, 3, 4, 5]), pd.Series([1, 3, 2, 5, 4]))
    assert report.spearman == pytest.approx(.8)
    assert report.weighted_kappa == pytest.approx(.8)
    assert report.exact_agreement == .2
    assert report.adjacent_agreement == 1


def test_reversed_ratings_capture_nonadjacent_disagreement():
    report = score_intrarater(pd.Series([1, 2, 3, 4, 5]), pd.Series([5, 4, 3, 2, 1]))
    assert report.spearman == pytest.approx(-1)
    assert report.weighted_kappa == pytest.approx(-1)
    assert report.exact_agreement == .2
    assert report.adjacent_agreement == .2


def test_tied_ordinal_ratings_use_average_rank_spearman():
    report = score_intrarater(pd.Series([1, 1, 2, 3]), pd.Series([1, 2, 2, 3]))
    assert report.spearman == pytest.approx(5 / 6)
    assert report.weighted_kappa == pytest.approx(.8)


def test_undefined_reliability_is_missing_not_perfect():
    constant = score_intrarater(pd.Series([3, 3]), pd.Series([3, 3]))
    assert constant.spearman is None
    assert constant.weighted_kappa is None
    assert constant.exact_agreement == 1
    empty = score_intrarater(pd.Series([None]), pd.Series([1]))
    assert empty.n_pairs == 0
    assert empty.spearman is None
    assert empty.exact_agreement is None


@pytest.mark.parametrize("bad", [0, 6, 1.2, True, "3"])
def test_intrarater_rejects_invalid_scores(bad):
    with pytest.raises(ValueError, match="1.*5"):
        score_intrarater(pd.Series([bad]), pd.Series([3]))
