# HSK-AgentBench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python benchmark that generates 144 HSK-aligned prompts through GPT, Gemini, Qwen, and DeepSeek, scores their outputs, and performs paired statistical comparisons between China-developed and global model groups.

**Architecture:** The laptop runs prompt construction and cloud-API generation into immutable, checksummed JSONL shards. The PC consumes those shards to normalize text, calculate CPU-first lexical/syntactic metrics, optionally use compatible GPU evaluators, prepare blinded ratings, and produce statistics and figures. Provider adapters share typed schemas and a resumable runner; no paid API is called by tests.

**Tech Stack:** Python 3.11, Pydantic 2, Typer, PyYAML, httpx, OpenAI SDK, Google Gen AI SDK, pandas, NumPy, SciPy, statsmodels, matplotlib, seaborn, jieba, pytest, pytest-asyncio, respx.

**Spec:** `docs/superpowers/specs/2026-10-25-hsk-agent-benchmark-design.md`

## Global Constraints

- Support Python 3.11 on Windows laptop and Windows PC.
- Use GPT and Gemini as the global model families; use Qwen and DeepSeek as the China-developed families.
- Core benchmark contains exactly 144 prompts: 3 tasks x HSK 3--6 x 12 topics.
- Never store API keys in project files; read named environment variables only.
- Unit and integration tests must not call paid APIs.
- Raw generation outputs are immutable JSONL shards and are never silently overwritten.
- Laptop and PC must never write the same shard concurrently; every closed shard gets a SHA-256 sidecar.
- GPU acceleration is optional; all required evaluators retain a CPU path and no required dependency is CUDA-only.
- Primary alpha is 0.05, two-sided; apply Holm correction within metric families and report effect sizes plus 95% confidence intervals.

## File Map

| Area | Files | Responsibility |
|---|---|---|
| Packaging | `pyproject.toml`, `.env.example`, `README.md` | Dependencies, commands, configuration and setup |
| Configuration | `configs/models.example.yaml`, `configs/experiment.yaml`, `configs/judges.yaml`, `configs/devices/*.yaml` | Model registry, experimental constants, judge rubric and machine roles |
| Schemas | `src/hsk_agent_bench/schemas.py`, `config.py`, `io.py` | Typed records, config loading, atomic JSONL/checksum operations |
| Prompts | `src/hsk_agent_bench/prompts/*.py`, `data/test_prompts/` | Three templates and deterministic 144-prompt manifest |
| Generation | `src/hsk_agent_bench/generators/*.py`, `runner.py` | Provider-neutral API clients and resumable async execution |
| Evaluation | `src/hsk_agent_bench/evaluators/*.py` | Lexical, syntactic, grammar, LLM-judge and human-rating metrics |
| Analysis | `src/hsk_agent_bench/analysis/*.py` | Aggregation, hypotheses, corrected inference and plots |
| CLI | `src/hsk_agent_bench/cli.py` | Stage-oriented commands for both machines |
| Tests | `tests/` | Hand-calculated fixtures, mocked APIs, statistics and dry-run pipeline |

## Review Focus

- Unknown or unlisted Chinese tokens must be reported separately and must not be silently counted as above-level vocabulary; pinned by Task 4 lexical tests.
- An interrupted API run must resume without duplicating or overwriting successful records; pinned by Task 3 runner tests.
- A provider that ignores or rejects `seed` must record this fact while retaining the requested seed in the run manifest; pinned by Task 3 adapter tests.
- Malformed or multiple-answer exam items must be rejected before rating; pinned by Task 2 schema tests.
- Paired tests must align China/global values by `prompt_id` and fail loudly on unmatched pairs; pinned by Task 7 statistics tests.

---

### Task 1: Package, Configuration, and Durable Records

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `configs/models.example.yaml`
- Create: `configs/experiment.yaml`
- Create: `configs/judges.yaml`
- Create: `configs/devices/laptop.yaml`
- Create: `configs/devices/pc.yaml`
- Create: `src/hsk_agent_bench/__init__.py`
- Create: `src/hsk_agent_bench/schemas.py`
- Create: `src/hsk_agent_bench/config.py`
- Create: `src/hsk_agent_bench/io.py`
- Create: `tests/test_config.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Produces: `TaskType`, `ModelOrigin`, `PromptRecord`, `GenerationRequest`, `GenerationRecord`, `MetricRecord`, and `RatingRecord` Pydantic models.
- Produces: `load_experiment_config(path: Path) -> ExperimentConfig`.
- Produces: `write_jsonl_shard_atomic(records: Iterable[BaseModel], target: Path) -> Path` returning the SHA-256 sidecar path.
- Produces: `verify_shard(target: Path) -> bool`.

- [ ] **Step 1: Write failing configuration and schema tests**

Test that the example registry contains exactly four enabled families with origins `{china, global}`, provider types `{openai, gemini, qwen, deepseek}`, environment-variable names instead of secrets, and primary/stochastic parameters from the spec. Test invalid HSK levels, missing `prompt_id`, and plaintext API keys are rejected.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_config.py -v`  
Expected: FAIL because package/config schemas do not exist.

- [ ] **Step 3: Implement packaging, schemas, and config loading**

Use Pydantic discriminated models for task-specific payloads. The model config stores `family`, `origin`, `provider`, `model_id`, `api_key_env`, `base_url`, capabilities, and enabled flag. Do not place real model IDs in `experiment.yaml`; they belong in the model registry.

- [ ] **Step 4: Write failing atomic-shard tests**

Test successful write plus checksum verification, refusal to overwrite an existing shard, cleanup of a temporary file after simulated failure, and detection of tampering.

- [ ] **Step 5: Implement atomic JSONL and checksum helpers**

Write to a sibling temporary file, flush and close it, compute SHA-256, then atomically rename. `verify_shard` compares the file with its `.sha256` sidecar.

- [ ] **Step 6: Run Task 1 tests**

Run: `python -m pytest tests/test_config.py tests/test_io.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example configs src/hsk_agent_bench tests/test_config.py tests/test_io.py
git commit -m "feat: bootstrap benchmark schemas and durable storage"
```

### Task 2: Canonical Data Formats and 144 Prompt Manifest

**Files:**
- Create: `data/hsk_vocab_levels/README.md`
- Create: `data/hsk_vocab_levels/schema.example.json`
- Create: `data/hsk_grammar_points/README.md`
- Create: `data/hsk_grammar_points/schema.example.json`
- Create: `data/test_prompts/.gitkeep`
- Create: `data/manifests/.gitkeep`
- Create: `src/hsk_agent_bench/prompts/__init__.py`
- Create: `src/hsk_agent_bench/prompts/common.py`
- Create: `src/hsk_agent_bench/prompts/reading.py`
- Create: `src/hsk_agent_bench/prompts/grammar_explanation.py`
- Create: `src/hsk_agent_bench/prompts/exam_generation.py`
- Create: `src/hsk_agent_bench/prompts/manifest.py`
- Create: `tests/test_prompt_rendering.py`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `PromptRecord` and experiment configuration from Task 1.
- Produces: `render_reading_prompt(record: PromptRecord) -> list[ChatMessage]`.
- Produces: `render_grammar_prompt(record: PromptRecord) -> list[ChatMessage]`.
- Produces: `render_exam_prompt(record: PromptRecord) -> list[ChatMessage]`.
- Produces: `build_manifest(config: ExperimentConfig) -> list[PromptRecord]`.
- Produces: `validate_exam_payload(payload: ExamOutput) -> None`.

- [ ] **Step 1: Write failing prompt-template tests**

For each task, assert identical inputs render identical messages and hashes; assert the prompt includes target level, topic, constraints and JSON schema; assert no provider/model name appears in the content.

- [ ] **Step 2: Run the prompt tests and verify failure**

Run: `python -m pytest tests/test_prompt_rendering.py -v`  
Expected: FAIL because prompt modules do not exist.

- [ ] **Step 3: Implement the three Chinese prompt templates**

Keep a common neutral system message and task-specific user templates. Require no chain-of-thought, strict JSON, and declared fields from the design. Hash canonical UTF-8 JSON of the rendered messages.

- [ ] **Step 4: Write failing manifest and exam-validation tests**

Assert exactly 144 unique prompt IDs, exactly 48 prompts per task, 36 per HSK level, 12 topics per task-level grid, stable ordering, and zero duplicated hashes. Assert duplicated options, missing answer keys, or multiple matching answers raise validation errors.

- [ ] **Step 5: Implement deterministic manifest construction and validation**

Use the 12 topics named in the spec and HSK levels `[3, 4, 5, 6]`. Write no generated manifest during import; expose construction through a function and later CLI command.

- [ ] **Step 6: Run Task 2 tests**

Run: `python -m pytest tests/test_prompt_rendering.py tests/test_manifest.py -v`  
Expected: PASS with 144 prompts.

- [ ] **Step 7: Commit**

```bash
git add data src/hsk_agent_bench/prompts tests/test_prompt_rendering.py tests/test_manifest.py
git commit -m "feat: add standardized HSK prompt manifest"
```

### Task 3: GPT, Gemini, Qwen, and DeepSeek Generation

**Files:**
- Create: `src/hsk_agent_bench/generators/__init__.py`
- Create: `src/hsk_agent_bench/generators/base.py`
- Create: `src/hsk_agent_bench/generators/openai_compatible.py`
- Create: `src/hsk_agent_bench/generators/openai_provider.py`
- Create: `src/hsk_agent_bench/generators/gemini_provider.py`
- Create: `src/hsk_agent_bench/generators/qwen_provider.py`
- Create: `src/hsk_agent_bench/generators/deepseek_provider.py`
- Create: `src/hsk_agent_bench/runner.py`
- Create: `tests/fixtures/provider_responses/*.json`
- Create: `tests/test_providers.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: `GenerationRequest`, rendered messages and model configuration.
- Produces: abstract `Generator.generate(request: GenerationRequest) -> GenerationRecord`.
- Produces: `make_generator(model: ModelConfig) -> Generator`.
- Produces: `run_manifest(manifest: Sequence[PromptRecord], models: Sequence[ModelConfig], run: RunConfig, output_dir: Path) -> RunSummary`.

- [ ] **Step 1: Write failing provider contract tests**

Mock all HTTP/SDK calls. Assert every adapter normalizes text, token usage, request ID, finish reason, latency, requested/applied seed, exact model ID, and prompt hash. Test Gemini separately; share OpenAI-compatible transport only where the provider documents compatibility.

- [ ] **Step 2: Run provider tests and verify failure**

Run: `python -m pytest tests/test_providers.py -v`  
Expected: FAIL because generator adapters do not exist.

- [ ] **Step 3: Implement provider-neutral interface and four adapters**

OpenAI, Qwen, and DeepSeek may reuse an OpenAI-compatible transport while retaining provider-specific capability flags and metadata. Gemini uses the Google Gen AI adapter. When `seed` is unsupported or rejected, record `seed_applied=false`; never silently alter temperature/top-p.

- [ ] **Step 4: Write failing resume, retry, and shard tests**

Simulate one success, one transient rate-limit response, one permanent schema failure, and interruption after a partial run. Assert retries preserve prompt content and parameters, successful IDs are skipped on resume, failures are logged, shards are not overwritten, and separate laptop/PC profiles cannot claim the same output shard.

- [ ] **Step 5: Implement resumable async runner**

Bound concurrency per provider, exponential backoff only for configured transient statuses, and write one immutable shard per run/model/replicate. Close each shard using Task 1 atomic checksum helpers.

- [ ] **Step 6: Run Task 3 tests**

Run: `python -m pytest tests/test_providers.py tests/test_runner.py -v`  
Expected: PASS without network access.

- [ ] **Step 7: Commit**

```bash
git add src/hsk_agent_bench/generators src/hsk_agent_bench/runner.py tests/fixtures tests/test_providers.py tests/test_runner.py
git commit -m "feat: add reproducible multi-provider generation"
```

### Task 4: Lexical Alignment Evaluator

**Files:**
- Create: `src/hsk_agent_bench/evaluators/__init__.py`
- Create: `src/hsk_agent_bench/evaluators/tokenization.py`
- Create: `src/hsk_agent_bench/evaluators/lexical.py`
- Create: `tests/fixtures/hsk_vocab_tiny.json`
- Create: `tests/test_lexical_metrics.py`

**Interfaces:**
- Consumes: normalized Chinese text and canonical mapping `word -> minimum_hsk_level`.
- Produces: `tokenize_chinese(text: str) -> list[str]`.
- Produces: `score_lexical_alignment(text: str, target_level: int, lexicon: HSKLexicon) -> LexicalMetrics` with `v_in_token`, `v_in_type`, `v_oov_token`, `v_oov_severe`, `unknown_token_rate`, and `mattr_50`.

- [ ] **Step 1: Write failing hand-calculated lexical tests**

Use tiny fixtures with repeated words, punctuation, numerals, an above-level word, a word two levels above, and an unknown word. Assert exact fractions, type/token differences, MATTR behavior below and above 50 tokens, and that unknown words are separate from above-level words.

- [ ] **Step 2: Run lexical tests and verify failure**

Run: `python -m pytest tests/test_lexical_metrics.py -v`  
Expected: FAIL because evaluator functions do not exist.

- [ ] **Step 3: Implement deterministic tokenization and lexical metrics**

Pin tokenizer/version in metadata. Remove punctuation from lexical denominators but retain an auditable token trace. Reject empty scorable text instead of dividing by zero.

- [ ] **Step 4: Run Task 4 tests**

Run: `python -m pytest tests/test_lexical_metrics.py -v`  
Expected: PASS with exact expected values.

- [ ] **Step 5: Commit**

```bash
git add src/hsk_agent_bench/evaluators tests/fixtures/hsk_vocab_tiny.json tests/test_lexical_metrics.py
git commit -m "feat: implement HSK lexical alignment metrics"
```

### Task 5: Syntactic Complexity and Grammar Validation

**Files:**
- Create: `src/hsk_agent_bench/evaluators/syntax.py`
- Create: `src/hsk_agent_bench/evaluators/grammar.py`
- Create: `tests/fixtures/dependencies.json`
- Create: `tests/test_syntax_metrics.py`
- Create: `tests/test_grammar_metrics.py`

**Interfaces:**
- Produces: `mean_sentence_length(text: str, tokenizer: Tokenizer) -> float`.
- Produces: `mean_dependency_distance(parsed: ParsedDocument) -> float`.
- Produces: `score_grammar_alignment(text: str, target_level: int, catalog: GrammarCatalog) -> GrammarAlignmentMetrics`.
- Produces: `candidate_grammar_flags(text: str, detector: GrammarDetector) -> list[GrammarFlag]`.
- Produces: `confirmed_violation_rate(flags: Sequence[GrammarFlag], labels: Mapping[str, bool], token_count: int) -> float`.

- [ ] **Step 1: Write failing syntactic metric tests**

Assert sentence segmentation around Chinese punctuation, hand-calculated dependency distances, exclusion of root arcs, handling of parser failures, and deterministic CPU behavior.

- [ ] **Step 2: Run syntax tests and verify failure**

Run: `python -m pytest tests/test_syntax_metrics.py -v`  
Expected: FAIL because syntax evaluator does not exist.

- [ ] **Step 3: Implement syntax metrics behind a parser protocol**

Keep parser loading lazy and configurable. Store parser name/version/device with every metric record. A missing GPU must choose CPU rather than fail the benchmark.

- [ ] **Step 4: Write failing grammar-separation tests**

Assert dependency anomalies never increment confirmed grammar errors without a human label; assert target/above-level grammar matches are counted separately; assert the violation denominator is Chinese tokens per 100.

- [ ] **Step 5: Implement grammar catalog matching and validation records**

Treat automatic detector results as candidates only. Export candidate spans and reasons for blinded human confirmation. Keep MuCGEC-compatible adapters optional and outside the core dependency set.

- [ ] **Step 6: Run Task 5 tests**

Run: `python -m pytest tests/test_syntax_metrics.py tests/test_grammar_metrics.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hsk_agent_bench/evaluators/syntax.py src/hsk_agent_bench/evaluators/grammar.py tests/fixtures/dependencies.json tests/test_syntax_metrics.py tests/test_grammar_metrics.py
git commit -m "feat: add validated syntax and grammar metrics"
```

### Task 6: Blinded Human Ratings and LLM-as-a-Judge

**Files:**
- Create: `docs/annotation_rubric.md`
- Create: `src/hsk_agent_bench/evaluators/human_ratings.py`
- Create: `src/hsk_agent_bench/evaluators/judge.py`
- Create: `tests/test_blinding.py`
- Create: `tests/test_judge.py`

**Interfaces:**
- Produces: `sample_for_human_rating(records: DataFrame, fraction: float, seed: int) -> DataFrame`.
- Produces: `blind_rating_sheet(records: DataFrame, seed: int) -> tuple[DataFrame, BlindingKey]`.
- Produces: `score_intrarater(first: Series, second: Series) -> ReliabilityReport`.
- Produces: `render_judge_request(output: NormalizedOutput, rubric: JudgeRubric) -> GenerationRequest`.
- Produces: `validate_judge_against_human(judge: DataFrame, human: DataFrame) -> JudgeValidationReport`.

- [ ] **Step 1: Write failing sampling and blinding tests**

Assert a 25% sample is balanced by model, task and HSK level; model/provider/origin fields are absent from rating sheets; IDs map back through a separately stored key; same-seed sampling is stable; same-prompt outputs are not adjacent.

- [ ] **Step 2: Run blinding tests and verify failure**

Run: `python -m pytest tests/test_blinding.py -v`  
Expected: FAIL because rating helpers do not exist.

- [ ] **Step 3: Implement rating workflow and rubric**

Document 1--5 anchors for naturalness, pedagogical quality and cultural appropriateness plus task-specific fields. Include the seven-day, 20% re-rating protocol and prohibit the evaluator from accessing the blinding key during scoring.

- [ ] **Step 4: Write failing judge-validation tests**

Use synthetic ratings to assert Spearman correlation, weighted Cohen kappa, exact agreement, adjacent agreement, missing-pair handling, and a configurable validation threshold.

- [ ] **Step 5: Implement judge rendering and validation**

The judge receives output, target level, task and rubric but no provider/model identity. Preserve raw judge responses and parse only strict JSON ratings plus short evidence spans.

- [ ] **Step 6: Run Task 6 tests**

Run: `python -m pytest tests/test_blinding.py tests/test_judge.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/annotation_rubric.md src/hsk_agent_bench/evaluators/human_ratings.py src/hsk_agent_bench/evaluators/judge.py tests/test_blinding.py tests/test_judge.py
git commit -m "feat: add blinded human and judge evaluation"
```

### Task 7: Aggregation, Hypothesis Tests, and Figures

**Files:**
- Create: `src/hsk_agent_bench/analysis/__init__.py`
- Create: `src/hsk_agent_bench/analysis/aggregate.py`
- Create: `src/hsk_agent_bench/analysis/hypothesis_tests.py`
- Create: `src/hsk_agent_bench/analysis/plots.py`
- Create: `tests/fixtures/synthetic_metrics.csv`
- Create: `tests/test_statistics.py`
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes: generation, metric and rating tables joined by `generation_id` and `prompt_id`.
- Produces: `aggregate_group_means(frame: DataFrame, metric: str) -> DataFrame`.
- Produces: `paired_origin_test(frame: DataFrame, metric: str, n_permutations: int, seed: int) -> TestResult`.
- Produces: `paired_wilcoxon(frame: DataFrame, metric: str) -> TestResult`.
- Produces: `holm_adjust(results: Sequence[TestResult], family: str) -> list[TestResult]`.
- Produces: `fit_mixed_effects(frame: DataFrame, formula: str, group: str) -> ModelReport`.
- Produces: `build_all_figures(frame: DataFrame, output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing paired-analysis tests**

Synthetic data must recover a known China/global shift, align pairs only by `prompt_id`, reject duplicated or unmatched group rows, calculate paired rank-biserial correlation, and return deterministic permutation results for a fixed seed.

- [ ] **Step 2: Run statistics tests and verify failure**

Run: `python -m pytest tests/test_statistics.py -v`  
Expected: FAIL because analysis functions do not exist.

- [ ] **Step 3: Implement aggregation and confirmatory tests**

Calculate individual-model results before origin-group means. Implement two-sided blocked sign-flip/permutation testing, Wilcoxon fallback, Holm adjustment by declared metric family, effect sizes, and bootstrap 95% intervals. Mixed-effects failures must return diagnostics rather than silently switching tests.

- [ ] **Step 4: Write failing plot tests**

Assert creation of overall heatmap, task-level grouped bars, HSK difficulty lines, spillover distribution, model stability plot, and judge-human agreement plot from synthetic data. Check files are non-empty and labels name both origin and individual models.

- [ ] **Step 5: Implement deterministic publication-oriented plots**

Use a fixed style and explicit model order from configuration. Save SVG and 300-dpi PNG; never infer significance from overlapping error bars in captions.

- [ ] **Step 6: Run Task 7 tests**

Run: `python -m pytest tests/test_statistics.py tests/test_plots.py -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hsk_agent_bench/analysis tests/fixtures/synthetic_metrics.csv tests/test_statistics.py tests/test_plots.py
git commit -m "feat: add paired inference and benchmark figures"
```

### Task 8: CLI, Dry Run, Documentation, and End-to-End Verification

**Files:**
- Create: `src/hsk_agent_bench/cli.py`
- Create: `src/hsk_agent_bench/evaluators/pipeline.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_dry_run.py`
- Create: `docs/data_dictionary.md`
- Modify: `README.md`
- Create: `outputs/raw/.gitkeep`
- Create: `outputs/normalized/.gitkeep`
- Create: `outputs/metrics/.gitkeep`
- Create: `outputs/ratings/.gitkeep`
- Create: `outputs/tables/.gitkeep`
- Create: `outputs/figures/.gitkeep`

**Interfaces:**
- Consumes: all interfaces from Tasks 1--7.
- Produces Typer commands: `validate-data`, `build-prompts`, `generate`, `evaluate`, `prepare-ratings`, `analyze`, `plot`, and `dry-run`.
- Produces: `evaluate_shard(input_path: Path, output_path: Path, config: EvaluationConfig) -> EvaluationSummary`.

- [ ] **Step 1: Write failing CLI tests**

Assert each command exposes `--help`; invalid device roles and missing source data fail with actionable messages; `generate --dry-run` renders requests but never instantiates a network client.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `python -m pytest tests/test_cli.py -v`  
Expected: FAIL because CLI does not exist.

- [ ] **Step 3: Implement CLI orchestration**

Each command performs one stage, reads explicit input/output paths, writes a run summary, and refuses incompatible device roles. `dry-run` uses fixtures to exercise the complete laptop-to-PC artifact boundary.

- [ ] **Step 4: Write failing end-to-end dry-run test**

Build a reduced manifest, generate mocked records for all four providers, close/verify shards, evaluate lexical/syntax metrics, create a blinded sample, run paired tests, and render at least one figure. Assert no network request and no overwrite on the second run.

- [ ] **Step 5: Implement evaluator pipeline and documentation**

Document Windows setup, editable install, environment variables, laptop commands, checksum transfer, PC commands, source-data schemas, pilot sequence, full-run sequence, troubleshooting, and limitations. Explain that exact model IDs must be frozen before pilot collection.

- [ ] **Step 6: Run the full verification suite**

Run: `python -m pytest -q`  
Expected: all tests PASS and no network access.

Run: `python -m hsk_agent_bench.cli dry-run --config configs/experiment.yaml --device configs/devices/pc.yaml`  
Expected: exit code 0; verified fixture shard, metric table, statistical table, and figure paths reported.

- [ ] **Step 7: Inspect repository hygiene**

Run: `git status --short` and `rg -n "(sk-|AIza|api[_-]?key\s*[:=]\s*['\"][^$])" -g '!references.bib' .`  
Expected: only intended files are present and no credential-like literal is found.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/data_dictionary.md outputs src/hsk_agent_bench/cli.py src/hsk_agent_bench/evaluators/pipeline.py tests/test_cli.py tests/test_dry_run.py
git commit -m "feat: complete reproducible HSK benchmark pipeline"
```

## Final Verification

- [ ] Confirm `references.bib` remains unchanged and contains 10 entries.
- [ ] Confirm the generated manifest contains exactly 144 prompts and the expected balance.
- [ ] Confirm all provider tests use mocks and no test accesses the network.
- [ ] Confirm all required evaluation paths work without GPU acceleration.
- [ ] Confirm laptop and PC profiles reject concurrent writes to the same shard.
- [ ] Confirm statistical outputs contain raw p, Holm-adjusted p, effect size, and 95% CI.
- [ ] Confirm human-rating exports contain no provider, model, or origin identifiers.
- [ ] Confirm README commands work in PowerShell on Python 3.11.
