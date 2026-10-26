"""Deterministic stratified sampling, identity blinding and ordinal agreement.

Inputs are DataFrames of GenerationRecord.model_dump() rows (or projections
containing the required columns). The returned key is restricted coordinator
data: never give it to the evaluator while scoring or re-rating.
"""

from dataclasses import dataclass
import heapq
import math
from numbers import Real
import random

import pandas as pd
from scipy.stats import spearmanr

from ..schemas import Record


@dataclass(frozen=True)
class BlindingKey:
    mapping: pd.DataFrame


class ReliabilityReport(Record):
    n_pairs: int
    n_missing: int
    spearman: float | None
    weighted_kappa: float | None
    exact_agreement: float | None
    adjacent_agreement: float | None


def _require_records(records: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns) - set(records.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if records[columns].isna().any().any():
        raise ValueError("required record fields cannot be missing")
    if not records.generation_id.is_unique:
        raise ValueError("generation_id must be unique")
    return records.sort_values("generation_id").reset_index(drop=True)


def sample_for_human_rating(records: pd.DataFrame, fraction: float = .25,
                            seed: int = 20261025) -> pd.DataFrame:
    """Proportional joint model/task/level sampling with a fixed total size.

Use largest-remainder allocation: each cell receives floor(n * fraction),
then seeded ties distribute the remaining slots. The total rounds half up
with a minimum of one for nonempty inputs. Small cells may receive zero when
the sampling budget is smaller than the number of strata. Input order has
no effect; different-size strata retain their original proportions.
"""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    source = _require_records(records, ["generation_id", "model_id", "task", "hsk_level"])
    if source.empty:
        return source
    rng = random.Random(seed)
    cells = list(source.groupby(["model_id", "task", "hsk_level"], sort=True))
    allocations = [math.floor(len(cell) * fraction) for _, cell in cells]
    target = max(1, math.floor(len(source) * fraction + .5))
    priorities = sorted(range(len(cells)), key=lambda i: (
        -(len(cells[i][1]) * fraction - allocations[i]), rng.random()))
    for i in priorities[:target - sum(allocations)]:
        allocations[i] += 1
    selected = []
    for (_, cell), size in zip(cells, allocations):
        selected.extend(rng.sample(cell.index.tolist(), size))
    return source.loc[selected].sort_values("generation_id").reset_index(drop=True)


def blind_rating_sheet(records: pd.DataFrame, seed: int = 20261025, *,
                       display_text_column: str = "text"
                       ) -> tuple[pd.DataFrame, BlindingKey]:
    """Whitelist rating content and schedule without adjacent prompt siblings.

Randomized heap ties and within-prompt shuffles are reproducible. An
impossible schedule raises instead of silently breaking the blinding rule.
The key retains original record metadata and must be stored separately.
Output text is preserved verbatim; self-identification inside generated
content requires coordinator review before release to raters.
"""
    required = ["generation_id", "prompt_id", "task", "hsk_level", "text"]
    if display_text_column != "text":
        required.append(display_text_column)
    source = _require_records(records, required)
    rng = random.Random(seed)
    groups = []
    for _, cell in source.groupby("prompt_id", sort=True):
        indices = cell.index.tolist()
        rng.shuffle(indices)
        groups.append(indices)
    if groups and max(map(len, groups)) > (len(source) + 1) // 2:
        raise ValueError("cannot avoid adjacent outputs from the same prompt")
    heap = [(-len(indices), rng.random(), i) for i, indices in enumerate(groups)]
    heapq.heapify(heap)
    order, previous = [], None
    while heap:
        remaining, _, i = heapq.heappop(heap)
        order.append(groups[i].pop())
        if previous is not None:
            heapq.heappush(heap, previous)
        previous = (remaining + 1, rng.random(), i) if remaining < -1 else None
    rating_ids = [f"r-{rng.getrandbits(128):032x}" for _ in order]
    ordered = source.loc[order].reset_index(drop=True)
    sheet = pd.DataFrame({"rating_id": rating_ids, "task": ordered.task.tolist(),
                          "hsk_level": ordered.hsk_level.tolist(),
                          "text": ordered[display_text_column].tolist()})
    mapping = ordered.copy(deep=True)
    mapping.insert(0, "rating_id", rating_ids)
    return sheet, BlindingKey(mapping=mapping)


def _validate_scores(scores: pd.Series) -> None:
    for score in scores.dropna():
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not math.isfinite(score) or int(score) != score or not 1 <= score <= 5):
            raise ValueError("scores must be integers from 1 to 5 or missing")


def score_intrarater(first: pd.Series, second: pd.Series) -> ReliabilityReport:
    """Align item IDs, drop incomplete pairs, and use quadratic weighted kappa.

    Report undefined correlation/kappa as None (including constant equal
    ratings), never as evidence of perfect reliability. Missing counts use
    the union of item IDs; duplicate indices are ambiguous and rejected.
    """
    if not first.index.is_unique or not second.index.is_unique:
        raise ValueError("rating indices must be unique")
    _validate_scores(first)
    _validate_scores(second)
    aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1)
    pairs = aligned.dropna()
    n = len(pairs)
    if not n:
        return ReliabilityReport(n_pairs=0, n_missing=len(aligned), spearman=None,
                                 weighted_kappa=None, exact_agreement=None, adjacent_agreement=None)
    a, b = pairs["first"].astype(float), pairs["second"].astype(float)
    correlation = float(spearmanr(a, b).statistic) if a.nunique() > 1 and b.nunique() > 1 else None
    observed = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    ac, bc = a.value_counts(), b.value_counts()
    expected = sum((x - y) ** 2 * nx * ny for x, nx in ac.items() for y, ny in bc.items()) / n ** 2
    return ReliabilityReport(n_pairs=n, n_missing=len(aligned) - n, spearman=correlation,
                             weighted_kappa=1 - observed / expected if expected else None,
                             exact_agreement=float((a == b).mean()),
                             adjacent_agreement=float(((a - b).abs() <= 1).mean()))
