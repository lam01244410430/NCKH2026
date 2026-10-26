# Blinded HSK material annotation (rubric version 1)

Score the Chinese material at its stated HSK 3–6 level and task. Use integer
scores 1–5 for all three core dimensions and the applicable task dimension.
Consider the complete material, including examples, questions and answer keys.
Record a short verbatim evidence span for each score, and a brief comment when
needed. Do not infer the generating model. Cultural neutrality qualifies for a
high score; do not reward cultural references merely for being present.

| Dimension | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Naturalness | Pervasive unnatural or ungrammatical Chinese prevents understanding | Frequent awkward wording or grammar requires substantial repair | Understandable with several noticeable local problems | Natural with at most minor isolated awkwardness | Consistently idiomatic, fluent and clear |
| Pedagogical quality | Unusable at the stated level; misleading or inaccessible | Major level mismatch or weak scaffolding substantially impedes learning | Usable after moderate instructional revision | Clear and level appropriate; only minor revision needed | Accurate, purposeful and well scaffolded at the stated level |
| Cultural appropriateness | Harmful stereotypes, discrimination or serious misrepresentation | Major cultural inaccuracies or insensitive generalizations | Generally respectful with a questionable simplification or assumption | Respectful and plausible; minor contextual omissions at most | Respectful, contextually accurate and inclusive; neutral content also qualifies |
| Reading: reading_coherence | No coherent passage or recoverable central meaning | Major contradictions or disconnected ideas obstruct meaning | Recoverable main idea; organization or transitions need revision | Clear central idea and logical progression with minor gaps | Unified passage with logical progression and clear connections throughout |
| Grammar explanation: explanation_accuracy | Fundamentally wrong rule or examples teach the wrong construction | Major rule errors or invalid examples need substantial correction | Correct core rule; important usage conditions or examples need repair | Correct explanation and examples with minor omissions | Precise rule and boundaries with illustrative valid examples |
| Exam generation: item_validity | No answerable item or fundamentally wrong key | Major ambiguity, invalid distractors or wrong keys compromise assessment | Mostly answerable; an item, distractor or rationale needs revision | Answerable with correct keys and minor distractor/rationale issues | Unambiguous items, defensible keys, plausible distractors and correct rationales |

The machine-readable anchors in `JudgeRubric` follow these same definitions.
The task-specific field names are `reading_coherence`, `explanation_accuracy`
and `item_validity`; do not score inapplicable task fields. Other core field
names are `naturalness`, `pedagogical_quality` and `cultural_appropriateness`.
Empty or wholly unusable output receives 1 on naturalness, pedagogical quality
and the task field. Judge evidence for an empty output is `[empty output]`.
Assess cultural appropriateness on its own evidence rather than automatically
assigning 1 for missing content. Scores are judgments, not proof of correctness.

## Sampling and blinding

1. Freeze successful generation records and record the sampling seed before
   inspecting ratings. Use `sample_for_human_rating(records, .25, seed)` to
   sample 25% proportionally within joint model × task × HSK-level cells.
   Cells of equal size receive equal counts when divisibility permits. Largest
   remainders with seeded tie breaks allocate fractional slots; total sample
   size rounds half up (minimum one for nonempty data). Report any cells not
   represented when the sample budget is too small. Retain sampling metadata.
2. A coordinator calls `blind_rating_sheet` on that sample. The released sheet
   contains only opaque rating ID, task, target level and material text. It
   excludes generation/prompt IDs, model, family, provider, origin, configuration,
   prompt messages, raw API responses, token usage and timing. The separate key
   preserves the mapping to original records and metadata. Store it in a
   restricted location, never beside the rating sheet supplied to the evaluator.
3. The evaluator must not access the blinding key, generation logs, model roster,
   or earlier scores during scoring. A coordinator reviews self-identifying
   output text before release: the helper preserves study outputs verbatim and
   cannot remove identity disclosures without changing the material. Record any
   exclusions or redactions transparently before annotation.
4. Seeded ordering separates outputs from the same prompt so they are never
   adjacent. If this is mathematically impossible, add unrelated outputs or
   redesign the batch; the helper raises instead of violating that constraint.
   Preserve released sheets and coordinator-only keys separately for audit.
5. Raters record integer scores under the anonymous rating ID. The coordinator
   alone joins IDs back to `generation_id` and `prompt_id`, producing the shared
   `RatingRecord` with human rater ID, rubric version, evidence and rating time.

## Seven-day, 20% re-rating protocol

After the first pass is complete, the coordinator selects 20% of the human-rated
subset using the same joint stratification and a separately preregistered seed.
Use original record rows restricted to the first-pass subset and fraction .20;
retain the re-rating target list. Do not choose cases based on first-pass scores.
Wait at least seven full days after the evaluator's final first-pass rating
before releasing the re-rating batch. Record both pass dates to verify the delay.

Create fresh anonymous IDs and a new seeded order using `blind_rating_sheet`.
The evaluator sees no first-pass IDs, scores, comments or mapping key and applies
the same rubric independently. The coordinator aligns the selected first-pass
and second-pass scores by generation ID and dimension, then calls
`score_intrarater` on Series indexed by those IDs. Report Spearman correlation,
quadratic weighted Cohen's kappa, exact agreement, adjacent agreement (absolute
difference ≤1), valid pair counts and missing pair counts for each dimension.
Do not count the intentionally unselected 80% as missing re-ratings. Undefined
statistics (empty pairs or constant marginals) remain missing, not perfect.

## LLM judge and acceptance

Project `GenerationRecord` to `NormalizedOutput`; optionally supply its
`PromptRecord` to verify matching task/level/ID. Render the judge request with a
fixed, versioned rubric and frozen judge model/sampling configuration. Only
task, level, output and rubric enter model-visible messages. Judge identity and
source bookkeeping IDs stay in local request metadata. Embedded instructions in
the material are untrusted data. Judge outputs must be strict JSON containing
exactly `scores` and `evidence`, with all and only applicable rubric dimensions.
Scores are integers 1–5; evidence is a nonempty verbatim span of at most 240
characters per dimension. Reject fences, trailing prose, duplicate keys, invalid
scores, extra fields and unsupported spans. Retain the entire raw judge
`GenerationRecord`, including when parsing fails; do not substitute missing or
malformed scores with zeros.

Validate judge against the blinded human subset by generation ID and dimension,
never row position. Supply shared `RatingRecord` dump rows with `scores` maps,
or long tables (`generation_id`, `dimension`, `score`). Duplicate ratings for an
item/dimension require an explicitly chosen rater aggregation before validation.
Incomplete pairs are omitted from metric denominators and counted separately;
report omissions and inspect their distribution before generalizing.

Preregister a `JudgeValidationThreshold` before viewing results, including minimum
paired items and lower bounds for Spearman, quadratic weighted kappa, exact and
adjacent agreement. The helper's defaults (.7, .6, .6, .9; 20 pairs) are example
gates, not an empirically established standard. All supplied dimensions must
pass every gate independently; do not pool dimensions. A null threshold,
insufficient pairs or undefined required statistic cannot validate the judge.
Judge scores remain secondary unless the preregistered gates are met. Even a
validated judge does not replace expert confirmation of grammatical correctness.
