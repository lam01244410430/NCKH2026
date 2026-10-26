"""Prompt-blocked two-sided inference; positive effects mean China - global.

The bootstrap resamples whole prompt differences, preserving the pairing.
Wilcoxon is an explicit robustness/fallback function, never automatically
substituted for a failed model. Holm families must be declared by callers.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import product
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from .aggregate import paired_origin_values


@dataclass(frozen=True)
class TestResult:
    metric: str
    method: str
    statistic: float
    pvalue: float
    n_pairs: int
    rank_biserial: float
    median_difference: float
    ci_low: float
    ci_high: float
    seed: int
    n_permutations: int = 0
    exact: bool = False
    family: str | None = None
    p_adjusted: float | None = None
    bootstrap_samples: int = 2000


@dataclass(frozen=True)
class ModelReport:
    success: bool
    formula: str
    group: str
    method: str = "mixed_effects"
    n_obs: int = 0
    converged: bool = False
    estimates: dict[str, float] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


def _differences(frame: pd.DataFrame, metric: str) -> np.ndarray:
    paired = paired_origin_values(frame, metric)
    return (paired.china - paired["global"]).to_numpy(dtype=float)


def _effects(differences: np.ndarray, seed: int) -> dict:
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero))
    rank_biserial = float(np.sum(np.sign(nonzero) * ranks) / ranks.sum()) if len(nonzero) else 0.
    rng = np.random.default_rng(seed)
    # Chunking bounds memory for large prompt grids.
    medians = np.concatenate([
        np.median(rng.choice(differences, size=(min(200, 2000 - start), len(differences))), axis=1)
        for start in range(0, 2000, 200)
    ])
    low, high = np.quantile(medians, [.025, .975])
    return dict(n_pairs=len(differences), rank_biserial=rank_biserial,
                median_difference=float(np.median(differences)), ci_low=float(low), ci_high=float(high))


def paired_origin_test(frame: pd.DataFrame, metric: str, n_permutations: int = 10000,
                       seed: int = 20261025) -> TestResult:
    """Two-sided sign-flip test of the mean paired origin difference.

    Each prompt contributes one China/global group mean. Enumerate all
    2**n sign assignments when within the requested budget; otherwise use
    a seeded Monte Carlo sample and the conservative (b+1)/(B+1) correction.
    Blocks are prompt IDs, irrespective of input row order or topic order.
    """
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    differences = _differences(frame, metric)
    n = len(differences)
    observed = float(differences.mean())
    exact = n < 63 and 2 ** n <= n_permutations
    total = 2 ** n if exact else n_permutations
    extreme = 0
    rng = np.random.default_rng(seed)
    assignments = product((-1, 1), repeat=n) if exact else None
    for start in range(0, total, 256):
        size = min(256, total - start)
        if exact:
            signs = np.array([next(assignments) for _ in range(size)])
        else:
            signs = rng.choice([-1, 1], size=(size, n))
        null = (signs * differences).mean(axis=1)
        absolute = np.abs(null)
        extreme += int(np.count_nonzero((absolute >= abs(observed))
                                        | np.isclose(absolute, abs(observed), rtol=1e-12, atol=0)))
    pvalue = extreme / total if exact else (extreme + 1) / (total + 1)
    return TestResult(metric=metric, method="paired_sign_flip", statistic=observed,
                      pvalue=pvalue, seed=seed, n_permutations=total, exact=exact,
                      **_effects(differences, seed))


def paired_wilcoxon(frame: pd.DataFrame, metric: str, *, seed: int = 20261025) -> TestResult:
    """Explicit two-sided signed-rank robustness test; discard zero ranks."""
    differences = _differences(frame, metric)
    if np.any(differences):
        test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    else:
        statistic, pvalue = 0., 1.
    return TestResult(metric=metric, method="wilcoxon", statistic=statistic,
                      pvalue=pvalue, seed=seed, **_effects(differences, seed))


def holm_adjust(results: Sequence[TestResult], family: str) -> list[TestResult]:
    """Return corrected copies in input order for one declared test family."""
    if not isinstance(family, str) or not family.strip():
        raise ValueError("a nonempty metric family must be declared")
    if any(r.family not in (None, family) for r in results):
        raise ValueError("cannot mix declared metric families")
    if any(not np.isfinite(r.pvalue) or not 0 <= r.pvalue <= 1 for r in results):
        raise ValueError("p-values must be finite and between zero and one")
    order = sorted(range(len(results)), key=lambda i: results[i].pvalue)
    adjusted = [0.] * len(results)
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(results) - rank) * results[index].pvalue)
        adjusted[index] = min(1., maximum)
    return [replace(r, family=family, p_adjusted=adjusted[i]) for i, r in enumerate(results)]


def fit_mixed_effects(frame: pd.DataFrame, formula: str, group: str) -> ModelReport:
    """Fit the requested Gaussian random-intercept model or return diagnostics.

    No rows are dropped implicitly. Singular designs, nonconvergence and
    invalid covariance estimates fail explicitly. This does not implement
    an ordinal cumulative-link model; ordinal ratings use paired_wilcoxon
    unless the caller has separately justified a Gaussian model.
    """
    diagnostics = []
    try:
        import statsmodels.formula.api as smf

        if group not in frame or frame[group].isna().any() or frame[group].nunique() < 2:
            raise ValueError("random intercept requires at least two nonmissing groups")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, frame, groups=frame[group], missing="raise")
            if not np.isfinite(model.exog).all() or not np.isfinite(model.endog).all():
                raise ValueError("model values must be finite")
            if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
                raise ValueError("rank-deficient fixed-effects design")
            fit = model.fit(method="lbfgs", reml=False)
            diagnostics.extend(str(w.message) for w in caught)
        if not fit.converged:
            raise ValueError("mixed-effects optimizer did not converge")
        if not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all():
            raise ValueError("nonfinite model estimates or covariance")
        if np.linalg.eigvalsh(fit.cov_re).min() <= 1e-10:
            raise ValueError("singular random-effects covariance")
        intervals = {key: tuple(map(float, value)) for key, value in fit.conf_int().iterrows()}
        return ModelReport(success=True, formula=formula, group=group, n_obs=int(fit.nobs),
                           converged=True, estimates=fit.params.to_dict(),
                           standard_errors=fit.bse.to_dict(), confidence_intervals=intervals,
                           diagnostics=tuple(diagnostics))
    except Exception as exc:
        diagnostics.append(f"{type(exc).__name__}: {exc}")
        return ModelReport(success=False, formula=formula, group=group,
                           n_obs=len(frame), diagnostics=tuple(diagnostics))
