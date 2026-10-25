# HSK-AgentBench: Experimental System Design

Date: 2026-10-25  
Status: Proposed for approval  
Paper: *Do China-developed LLM Agents Produce Better HSK-Aligned Learning Materials? A Controlled Comparison with Global Models*

## 1. Intent

Build a reproducible, API-first Python benchmark that compares China-developed and global LLM families under the same agent workflow when producing HSK-aligned Chinese learning materials. The system must generate materials, calculate automatic linguistic metrics, support blinded human and LLM-as-a-Judge ratings, and run paired statistical analyses.

The two-month constraint makes this an evaluation paper, not a new-model or fine-tuning project. A laptop will handle prompt construction and cloud-API generation. The PC (Ryzen 5 5600, 16 GB RAM, RX 6600 8 GB VRAM) will handle Chinese NLP preprocessing, metric calculation, statistics, plots, and optional GPU-accelerated local evaluators. Large-model generation will use cloud APIs; local inference of large generative models is explicitly out of scope.

## 2. Goals, Non-goals, and Anti-goals

### Goals

- G1 Coverage: three tasks (reading passage, grammar explanation, exam-item generation), four core HSK levels (3--6), and balanced topics.
- G2 Diagnostics: separate lexical alignment, syntactic complexity, grammatical correctness, content quality, and cultural appropriateness.
- G3 Reproducibility: versioned prompts and configs, immutable raw responses, request metadata, fixed sampling manifests, and deterministic seeds where providers support them.
- G4 Quality: automated validation for every output, blinded human ratings on a stratified subset, and validation of LLM-as-a-Judge against human ratings.
- Fit within eight weeks and consumer hardware.

### Non-goals

- Training or fine-tuning foundation models.
- Claiming that model origin causally explains all differences; origin is a family-level grouping, while individual model identity remains visible in the analysis.
- Evaluating speech, listening, pronunciation, multimodality, or learner improvement over a semester.
- Running large local models on the RX 6600.
- Covering all HSK 1--9 levels in the first confirmatory experiment. Vocabulary and grammar resources may cover 1--9, but the core benchmark uses HSK 3--6 to keep cells sufficiently populated.

### Anti-goals

- Comparing different agent architectures, prompt quality, retrieval corpora, or token budgets across model groups.
- Treating BLEU/ROUGE, a parser failure, or an LLM judge score as proof of grammatical correctness.
- Reporting only p-values without effect sizes, uncertainty intervals, and multiple-testing correction.
- Quietly substituting model versions after data collection begins.

## 3. Approaches Considered

| Approach | Advantages | Disadvantages | Decision |
|---|---|---|---|
| Minimal folders and documents | Fastest | Not executable; weak reproducibility | Rejected |
| Python benchmark package | Reproducible, testable, API-neutral, suitable for a paper artifact | More initial engineering | **Selected** |
| Notebook-first | Fast exploration and visualization | Harder to test, resume, and audit | Rejected as the primary architecture; notebooks may be optional consumers of exported tables |

## 4. Benchmark Definition

### 4.1 Working name and evaluation gap

Working name: **HSK-AgentBench**.

Gap statement: prior work evaluates HSK alignment, Chinese text readability, Chinese grammatical correction, or a single LLM-assisted teaching intervention, but does not conduct a controlled, prompt-matched comparison of China-developed and global model families inside the same agent workflow across multiple learning-material tasks.

### 4.2 Research questions

- RQ1: Under a controlled agent scaffold, do China-developed and global LLM families differ in automatic HSK alignment across reading, grammar-explanation, and exam-generation tasks?
- RQ2: Do the two groups differ in blinded ratings of naturalness, pedagogical quality, and cultural appropriateness?
- RQ3: How do any group differences change with HSK level, task type, and generation stochasticity?

### 4.3 Units and sampling frame

- Core levels: HSK 3, 4, 5, and 6.
- Tasks: three.
- Topics: 12 balanced topic families covering daily life, education, work, travel, health, technology, environment, culture, public services, commerce, interpersonal communication, and academic life.
- Prompt count: 3 tasks x 4 levels x 12 topics = **144 standardized prompts**.
- Models: initially four model families, two China-developed (Qwen, DeepSeek) and two global (OpenAI GPT, Google Gemini). Exact API model IDs must be frozen in `configs/models.yaml` before the pilot.
- Primary run: one output per prompt-model cell at the provider's most deterministic supported setting = 576 outputs.
- Reliability run: repeat all primary cells once = another 576 outputs.
- Stochasticity analysis: a 25% stratified prompt subset, three fixed seeds at temperature 0.7 = 432 outputs.
- Planned maximum: approximately **1,584 material outputs**, excluding retries and judge calls.

Model origin is not interchangeable with model identity. Results will report both individual models and the planned China/global contrast.

## 5. Repository Architecture

```text
prj/
├── configs/
│   ├── models.example.yaml
│   ├── experiment.yaml
│   └── judges.yaml
├── data/
│   ├── hsk_vocab_levels/
│   ├── hsk_grammar_points/
│   ├── test_prompts/
│   └── manifests/
├── docs/
│   ├── annotation_rubric.md
│   ├── data_dictionary.md
│   └── superpowers/specs/
├── src/
│   ├── generators/
│   │   ├── base.py
│   │   ├── openai_provider.py
│   │   ├── gemini_provider.py
│   │   ├── qwen_provider.py
│   │   └── deepseek_provider.py
│   ├── evaluators/
│   │   ├── lexical.py
│   │   ├── syntax.py
│   │   ├── grammar.py
│   │   ├── judge.py
│   │   └── human_ratings.py
│   ├── analysis/
│   │   ├── aggregate.py
│   │   ├── hypothesis_tests.py
│   │   └── plots.py
│   ├── prompts/
│   │   ├── reading.py
│   │   ├── grammar_explanation.py
│   │   └── exam_generation.py
│   ├── schemas.py
│   ├── runner.py
│   └── cli.py
├── tests/
│   ├── fixtures/
│   ├── test_lexical_metrics.py
│   ├── test_prompt_rendering.py
│   ├── test_response_parsing.py
│   └── test_statistics.py
├── outputs/
│   ├── raw/
│   ├── normalized/
│   ├── metrics/
│   ├── ratings/
│   ├── tables/
│   └── figures/
├── .env.example
├── pyproject.toml
├── README.md
└── references.bib
```

Empty data and output directories will contain `.gitkeep` files. Secrets will never be stored in configuration files or raw response metadata.

## 6. Prompt Protocol

All providers receive semantically identical system and user messages. Provider-specific wrappers may translate message syntax but may not add content. Each prompt contains a stable `prompt_id`, task, HSK level, topic, required length, required grammar points, output schema, and prohibited behavior.

### 6.1 Reading passage template

Inputs: `hsk_level`, `topic`, `target_length_chars`, `required_vocab`, `required_grammar`, `max_out_of_level_words`.

Required output:

```json
{
  "title": "...",
  "passage": "...",
  "declared_hsk_level": 4,
  "used_required_vocabulary": ["..."],
  "used_required_grammar": ["..."]
}
```

The model is instructed to write natural standalone Chinese, preserve factual plausibility, use the requested HSK level, and avoid explaining its reasoning.

### 6.2 Grammar explanation template

Inputs: `hsk_level`, `grammar_point`, `learner_l1`, `context`, `example_count`.

Required output fields: concise explanation, form, usage constraints, correct contextual examples, one contrastive incorrect example with correction, and a short practice item. The explanation language itself must respect the requested HSK level unless the protocol explicitly requests bilingual support.

### 6.3 Exam-item template

Inputs: `hsk_level`, `topic`, `item_type`, `source_length`, `num_items`.

Required output fields: source passage, question, four options, unique answer, rationale, target vocabulary, and target grammar. The rationale is retained for validation but hidden from test takers. Automated checks reject duplicated options, missing keys, and multiple identical answers.

Full Chinese-language prompt wording will live in version-controlled template files and will be hashed into every output record.

## 7. Reproducibility Controls

### 7.1 Primary confirmatory setting

- `temperature`: 0 or the provider's documented minimum.
- `top_p`: 1.0 unless the API disallows combining it with the selected temperature.
- `seed`: 20261025 where supported.
- Fixed maximum output tokens by task.
- No hidden retries that change prompts or sampling parameters.

### 7.2 Stochastic robustness setting

- `temperature`: 0.7.
- `top_p`: 0.9.
- Seeds: 20261025, 20261026, 20261027 where supported.
- Run on a stratified 25% subset only.

A seed does not guarantee identical behavior across providers. Each record will therefore store provider, exact model ID, model-version date when available, request time, request ID, parameters, prompt hash, latency, token usage, finish reason, retry count, and raw response. The analysis treats reproducibility as auditable reconstruction, not bitwise equality across vendors.

### 7.3 Two-machine execution

- Laptop profile: validate/build prompts, call GPT/Gemini/Qwen/DeepSeek APIs, and write immutable raw JSONL shards.
- PC profile: normalize responses, run tokenization/parsing and optional GPU-compatible evaluators, calculate metrics, run statistics, and render figures.
- Both machines use the same locked environment and run manifest, but never write to the same shard concurrently.
- Each completed shard is closed atomically and accompanied by a SHA-256 checksum before synchronization or transfer.
- GPU acceleration is optional. Every evaluator must retain a CPU path because Windows support for the RX 6600 varies by library; the study must not depend on CUDA-only software.

## 8. Evaluation Framework

### 8.1 Automatic metrics

| Dimension | Metric | Definition |
|---|---|---|
| Lexical alignment | `V_in_token` | Tokens whose minimum HSK level is less than or equal to the target level divided by scorable lexical tokens. |
| Lexical alignment | `V_in_type` | Unique in-level word types divided by scorable unique word types. |
| Spillover | `V_oov_token` | Tokens above the target level divided by scorable lexical tokens. Report separate unknown/unlisted-token rate. |
| Severe spillover | `V_oov_severe` | Tokens at least two levels above the target divided by scorable lexical tokens. |
| Lexical diversity | `MATTR_50` | Mean type-token ratio across 50-token windows. |
| Surface readability | Mean sentence length | Chinese lexical tokens per sentence, with punctuation segmentation rules frozen in the data dictionary. |
| Syntactic complexity | `MDD` | Mean dependency distance from a pinned Chinese dependency parser. |
| Grammar alignment | Target grammar coverage | Required HSK grammar points correctly realized divided by requested points. |
| Grammar spillover | Above-level grammar rate | Detected grammar points above the target divided by all detected graded grammar points. |
| Grammatical correctness | Human-validated violation rate | Confirmed grammatical violations per 100 Chinese tokens on the validation subset. |

`MuCGEC` is a benchmark/tool ecosystem, and a dependency parser does not itself detect grammatical errors. Therefore, parser outputs will support syntactic-complexity and rule checks, not serve as a standalone Grammar Error Rate. A pluggable candidate error detector may flag sentences, but its precision/recall must be validated against blinded human annotations before its rate is reported as an error metric.

### 8.2 Subjective metrics

Each output receives 1--5 ordinal ratings for:

- Naturalness/Idiomaticity.
- Pedagogical Quality.
- Cultural Appropriateness.

Task-specific secondary dimensions:

- Reading: coherence and level appropriateness.
- Grammar explanation: correctness, clarity, and example validity.
- Exam generation: answer uniqueness, distractor quality, and question answerability.

The LLM judge receives anonymized, randomly ordered outputs and a fixed rubric. A stratified human-rated subset will be used to measure judge agreement using Spearman correlation, weighted Cohen's kappa, and exact/adjacent agreement. Judge scores remain secondary unless validation reaches a preregistered acceptable threshold.

### 8.3 Single-human-rater safeguards

Because the researcher is the only Chinese evaluator:

- Hide provider, model, model origin, and sampling configuration.
- Randomize output order and separate outputs originating from the same prompt.
- Rate at least 25% of outputs, balanced by model, task, and level.
- Re-rate 20% of that subset after at least seven days.
- Report intra-rater weighted kappa or ICC with bootstrap confidence intervals.
- Explicitly report the absence of inter-rater reliability as a limitation.

## 9. Hypotheses and Statistical Matrix

The confirmatory alpha is 0.05, two-sided. Holm correction is applied within each metric family. Report effect sizes and 95% confidence intervals.

| Hypothesis | Outcome | Unit/blocking | Primary test | Robustness test | Effect size |
|---|---|---|---|---|---|
| H1: origin groups differ in HSK lexical alignment | `V_in`, `V_oov` | Same prompt across models | Prompt-blocked paired permutation test on group means | Wilcoxon signed-rank | Paired rank-biserial correlation and median paired difference |
| H2: origin groups differ in syntactic/grammar alignment | MDD, grammar coverage, violation rate | Same prompt | Mixed-effects model with origin, task, level; prompt random intercept | Wilcoxon by task with Holm correction | Standardized contrast or rank-biserial correlation |
| H3: origin groups differ in human/LLM quality ratings | 1--5 ratings | Same prompt | Cumulative-link model when estimable | Paired Wilcoxon signed-rank | Cliff's delta or rank-biserial correlation |
| H4: origin effect varies by task or HSK level | Each primary outcome | Prompt x task x level | Origin x task and origin x level interactions | Stratified permutation tests | Interaction contrasts with confidence intervals |
| H5: model families differ in output stability | Within-cell variance | Repeated generations | Levene/Brown--Forsythe plus bootstrap variance ratios | Fligner--Killeen | Variance ratio |

Repeated-measures ANOVA will be reported only when residual normality, variance, and sphericity assumptions are defensible. It is not the default because prompts are paired, ratings are ordinal, and models are nested within origin groups.

With only two models per origin group, origin-level generalization is limited. The paper will phrase conclusions as applying to the evaluated model set, not to every model developed in China or elsewhere.

## 10. Data and Construction Pipeline

The benchmark uses controlled template instantiation plus expert validation:

| Stage | Input | Operation | Output | Quality gate |
|---|---|---|---|---|
| Source selection | Licensed HSK 3.0 vocabulary/grammar files | Normalize levels, provenance, spelling, duplicates | Canonical resources | Schema, license, duplicate audit |
| Prompt construction | Taxonomy and canonical resources | Balanced deterministic template expansion | 144 prompts | Cell-count and prompt-hash checks |
| Pilot generation | 12 prompts across all cells | Run all providers | Pilot outputs | JSON validity, truncation, cost and latency audit |
| Full generation | Frozen manifest | Resumable API runner | Raw immutable outputs | Completeness and retry audit |
| Automated scoring | Normalized outputs | Tokenization, parsing, rule metrics | Metric table | Unit tests and range checks |
| Subjective scoring | Blinded stratified subset | LLM judge and human rubric | Rating table | Intra-rater and judge-human agreement |
| Analysis | Frozen tables | Confirmatory and exploratory tests | Tables and figures | Reproducible analysis command |

Prompts and outputs will have separate IDs. Prompt data will be versioned; generated outputs will not be silently overwritten. A run manifest records exclusions and failures.

## 11. Software Design

- Python 3.11 package using a `src/` layout.
- Typed provider interface returning a provider-neutral response schema.
- Async request execution with explicit concurrency and rate limits.
- Retry only on declared transient errors, with exponential backoff and complete logging.
- Pydantic or dataclass schemas for prompts, responses, and metric records.
- JSONL for immutable raw/normalized records; CSV or Parquet for analysis tables.
- CLI stages: `validate-data`, `build-prompts`, `generate`, `evaluate`, `prepare-ratings`, `analyze`, and `plot`.
- Unit tests must not call paid APIs. Provider clients use mocked HTTP responses.
- API dependencies are optional extras so metric development does not require every vendor SDK.

## 12. Eight-Week Execution Plan

| Week | Deliverable |
|---|---|
| 1 | Repository scaffold, schemas, canonical HSK resource format, metric unit tests |
| 2 | Three prompt templates, balanced 144-prompt manifest, provider-neutral generator interface |
| 3 | Four provider adapters, mock tests, pilot run, cost and failure audit |
| 4 | Full primary and reliability generations |
| 5 | Automatic evaluators, parser validation, normalized metric tables |
| 6 | Blinded human subset and LLM-judge validation |
| 7 | Statistical tests, robustness analysis, figures and error taxonomy |
| 8 | Paper tables, case studies, limitations, reproducibility audit and artifact cleanup |

## 13. Risks and Defenses

| Risk | Defense |
|---|---|
| Proprietary APIs change during the study | Freeze exact model IDs and collection window; store metadata and rerun no model silently. |
| Origin is confounded with individual model | Report per-model results, paired prompt contrasts, and cautious group-level wording. |
| Only one human evaluator | Blinding, repeated ratings, intra-rater reliability, validated judge, and explicit limitation. |
| HSK lists differ across sources | Select one canonical licensed source and document normalization; do not merge incompatible standards without provenance. |
| Chinese tokenization changes lexical ratios | Pin tokenizer/version and publish both token- and type-based results plus character-based sensitivity analysis. |
| Parser mistakes are treated as grammar mistakes | Separate syntactic parsing from error detection and human-validate any automatic detector. |
| API cost or rate limits exceed budget | Pilot first, cache all successful calls, use resumable manifests, and limit high-temperature analysis to 25%. |
| RX 6600 lacks mainstream CUDA support | Assign the PC compute role, but keep a CPU fallback and treat DirectML/ONNX/ROCm acceleration as optional after a pilot compatibility check. |
| Laptop and PC corrupt synchronized outputs | Use immutable per-run JSONL shards, atomic close/rename, checksums, and a single-writer rule. |

## 14. Acceptance Criteria

The implementation is complete when:

1. The repository tree and configuration examples exist and contain no secrets.
2. All three prompt types render deterministically from the manifest.
3. Four provider adapters conform to one interface and pass mocked tests.
4. Automatic lexical and syntactic metrics pass hand-calculated fixtures.
5. A dry-run can execute the complete pipeline without paid API calls.
6. A pilot can resume safely after interruption without overwriting successful outputs.
7. Statistical scripts reproduce a small synthetic analysis with corrected p-values and effect sizes.
8. The README documents setup, data provenance, commands, and known limitations.

## 15. Builder Decisions Left Open

The implementer may choose specific lightweight libraries for configuration, HTTP transport, CLI, testing, plotting, and statistical models, provided they support Python 3.11, are actively maintained, and do not require CUDA. The exact commercial model IDs, API credentials, and licensed HSK source files remain user-supplied configuration rather than hard-coded project content.
