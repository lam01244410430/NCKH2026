# HSK-AgentBench data dictionary

All text files use UTF-8 and all timestamps use timezone-aware ISO 8601. JSONL
artifacts are immutable: each data file has a sibling `<name>.sha256` containing
the lowercase SHA-256 digest. IDs are stable join keys; filenames are not keys.

## Canonical sources

### Vocabulary JSON

A JSON array with one object per surface form:

| Field | Type | Meaning |
|---|---|---|
| `word` | non-empty NFC string | Canonical word form, with no surrounding whitespace. |
| `min_hsk_level` | integer 1–9 | Lowest level assigned by the selected standard. |
| `standard` | string | Exact HSK standard/edition. |
| `source` | string | Citation or stable source identifier. |
| `license` | string | Redistribution/use terms. |
| `notes` | string/null | Normalization decisions. |

Duplicate forms must be resolved with documented provenance before evaluation;
the benchmark scores against the lowest retained level in a validated snapshot.

### Grammar JSON

A JSON array containing `grammar_id`, regex `pattern`, `min_hsk_level` (1–9),
`description`, `examples`, `standard`, `source`, `license`, and optional `notes`.
Patterns indicate occurrence only, not grammatical correctness. They must not
match empty strings and require expert curation before study use.

## Prompt manifest (`PromptRecord`)

| Field | Meaning |
|---|---|
| `prompt_id` | Stable task/level/topic identifier. |
| `task` | `reading`, `grammar_explanation`, or `exam_generation`. |
| `hsk_level` | Confirmatory level 3–6. |
| `topic` | Balanced topic family. |
| `payload` | Task-discriminated constraints and requested content. |
| `prompt_version` | Frozen template version. |
| `prompt_hash` | SHA-256 of the exact provider-neutral chat messages. |

The standard configuration has 144 prompts: 3 tasks × 4 levels × 12 topics.

## Generation shard (`GenerationRecord`)

Each row includes `generation_id`, `prompt_id`, task/level, family, origin,
provider, exact `model_id`, prompt hash/messages, sampling parameters, `run_id`,
replicate, generated `text`, raw provider response, request timestamp/ID,
reported model version, latency, token usage, finish reason, retry count, and
whether the provider applied the seed. Secrets are never stored.

`generation_id` is unique. `run_id` plus `replicate` identifies the analysis
condition; repeated conditions must be selected explicitly before inference.

## Metric shard (`MetricRecord`)

`generation_id` and `prompt_id` join metrics to generation rows. `evaluator`,
`evaluator_version`, and `device` record provenance. `metrics` contains:

| Key | Definition |
|---|---|
| `v_in_token` | Fraction of scorable tokens at or below target HSK level. |
| `v_in_type` | Fraction of unique scorable types at or below target level. |
| `v_oov_token` | Fraction of listed tokens above target level. |
| `v_oov_severe` | Fraction at least two levels above target. |
| `unknown_token_rate` | Fraction absent from the canonical lexicon. |
| `mattr_50` | Mean type-token ratio over length-50 windows (full text if shorter). |
| `mean_sentence_length` | Mean Han-token count in nonempty sentences. |
| `mean_dependency_distance` | Mean non-root dependency arc distance; null on parser failure. |
| `in_level_matches` | Grammar-pattern occurrences at/below target (when configured). |
| `target_level_matches` | Occurrences exactly at target. |
| `above_level_matches` | Occurrences above target. |

`metadata` stores tokenizer, lexicon/catalog hashes, parser name/version/device,
segmentation policies, and parser diagnostics. Unknown tokens remain distinct
from known above-level tokens.

## Rating artifacts

The evaluator-facing CSV contains only `rating_id`, `task`, `hsk_level`, and
`text`; it never contains provider, model, origin, generation ID, or prompt ID.
The separately protected coordinator key maps random `rating_id` values back to
the complete generation metadata. Rating records contain rater ID/type, 1–5
dimension scores, rubric version, evidence spans, and optional rating time.

## Analysis tables and outputs

The joined table has one row per `generation_id`/`prompt_id`, generation
metadata, evaluator keys unchanged, and ratings prefixed `human_` or `llm_`.
Callers select one run/replicate and provide enabled frozen model IDs in config
order before aggregation. Statistical CSVs include `pvalue`, Holm-adjusted
`p_adjusted`, `rank_biserial`, `median_difference`, `ci_low`, `ci_high`, paired
sample size, seed, and permutation method/count. Figures are emitted as editable
SVG and 300-dpi PNG.

Output folders are `raw`, `normalized`, `metrics`, `ratings`, `tables`, and
`figures`. Generated contents are ignored by Git; `.gitkeep` records only the
directory contract.
