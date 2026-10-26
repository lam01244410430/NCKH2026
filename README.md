# HSK-AgentBench

HSK-AgentBench is a CPU-first Python benchmark for a controlled comparison of
Qwen and DeepSeek with GPT and Gemini on HSK-aligned learning-material tasks.
It keeps individual model results visible before computing the planned
China/global contrast. The included dry run is synthetic; it proves that the
pipeline works but is not research evidence.

## Windows setup

Python 3.11 or newer is required. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,providers]"
python -m pytest -q
```

Copy `configs/models.example.yaml` to the gitignored `configs/models.yaml`.
Replace every `FREEZE_*_MODEL_ID` value with an exact API model identifier
before the pilot, then set `model_registry: models.yaml` in your private copy of
the experiment config; never change those IDs during a collection window. Keep
only environment-variable names in YAML and set secrets in the shell:

```powershell
$env:OPENAI_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
$env:QWEN_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."
```

The core evaluator is CPU-only and does not require CUDA. An RX 6600 can be
used only through an optional, locally configured parser backend; CPU remains
the supported fallback.

## Required source data

Replace the illustrative files under `data/hsk_vocab_levels/` and
`data/hsk_grammar_points/` with licensed, provenance-bearing HSK 3.0 data.
Their canonical schemas and every generated artifact are defined in
[`docs/data_dictionary.md`](docs/data_dictionary.md). Do not report a study
using the illustrative mappings.

Validate the real sources on the laptop:

```powershell
python -m hsk_agent_bench.cli validate-data `
  --config configs/experiment.yaml `
  --device configs/devices/laptop.yaml `
  --vocab data/hsk_vocab_levels/hsk3.json `
  --grammar data/hsk_grammar_points/hsk3.json `
  --summary outputs/tables/source-validation.json
```

## Reproducible two-machine workflow

Only the laptop writes generation shards. Build a frozen 144-prompt manifest,
then preview all requests without constructing a provider client:

```powershell
python -m hsk_agent_bench.cli build-prompts `
  --config configs/experiment.yaml --device configs/devices/laptop.yaml `
  --output data/manifests/primary.jsonl

python -m hsk_agent_bench.cli generate `
  --config configs/experiment.yaml --device configs/devices/laptop.yaml `
  --protocol primary `
  --manifest data/manifests/primary.jsonl --output-dir outputs/raw `
  --preview-output outputs/tables/request-preview.jsonl --dry-run
```

After inspecting the preview and estimating cost, omit `--dry-run` to call the
configured APIs. Use `--protocol stochastic` for the preregistered reduced
stochastic run; the selected protocol controls temperature, top-p, prompt
fraction, seeds, and replicates in both preview and live execution. `--run-id`
is only an output label and never selects sampling settings. Prompt fractions
are selected deterministically within every task/HSK-level stratum and the same
frozen subset is reused for every model, seed, and replicate.

Provider `text` must be one strict JSON object matching the requested task
schema. Evaluation and rating preparation validate that object (including HSK
level and exam-option uniqueness), then score/show a deterministic learner-facing
projection. The original `text` and raw provider response remain unchanged in
the immutable generation shard. A shard is complete only when its sibling `.sha256` file is
present and verifies. Transfer both files to the PC and verify checksums before
evaluation; never let the laptop and PC write the same shard namespace.

```powershell
Get-FileHash outputs/raw/<run>/<shard>.jsonl -Algorithm SHA256
python -m hsk_agent_bench.cli evaluate `
  --input outputs/raw/<run>/<shard>.jsonl `
  --output outputs/metrics/<shard>.metrics.jsonl `
  --vocab data/hsk_vocab_levels/hsk3.json `
  --grammar data/hsk_grammar_points/hsk3.json `
  --parser-config configs/parser.yaml `
  --device configs/devices/pc.yaml
```

`configs/parser.yaml` identifies a pinned local parser adapter without allowing
the benchmark to download a model:

```yaml
name: your-parser
version: "PINNED_VERSION"
factory: your_local_package.parser:create_backend
device: cpu
```

The factory module is imported lazily on the first evaluated record. Its
`create_backend(device)` function receives `"cpu"` and returns an object with a
`parse(text)` method producing the normalized dependency schema. Install and
cache parser weights separately before collection; the factory must never
download during a benchmark run. Omitting `--parser-config` deliberately emits
a missing MDD plus a `not-configured` diagnostic rather than inventing a score.

Prepare the human sheet while keeping the coordinator key private:

```powershell
python -m hsk_agent_bench.cli prepare-ratings `
  --input outputs/normalized/analysis.csv `
  --output outputs/ratings/blinded.csv `
  --key-output outputs/ratings/private-key.csv `
  --device configs/devices/pc.yaml
```

Analysis input must contain exactly one selected run/replicate, or pass
`--run-id` and `--replicate`. It must include every enabled exact model ID in
frozen configuration order. This prevents model dropout or row order from
silently changing the origin contrast.

```powershell
python -m hsk_agent_bench.cli analyze `
  --input outputs/normalized/analysis.csv --output outputs/tables/primary.csv `
  --config configs/experiment.yaml --device configs/devices/pc.yaml `
  --run-id primary --replicate 0 --family lexical `
  --metric v_in_token --metric v_oov_token

python -m hsk_agent_bench.cli plot `
  --input outputs/normalized/analysis.csv --output-dir outputs/figures `
  --config configs/experiment.yaml --device configs/devices/pc.yaml `
  --run-id primary --replicate 0
```

Confirmatory inference and ordinary figures keep exactly one replicate. To
render the descriptive stability panel across a balanced repeated condition,
use a separate output directory and retain the full repeat roster explicitly:

```powershell
python -m hsk_agent_bench.cli plot `
  --input outputs/normalized/stochastic.csv --output-dir outputs/figures-stability `
  --config configs/experiment.yaml --device configs/devices/pc.yaml `
  --run-id stochastic --include-repeats-for-stability
```

If a metric is unavailable for any selected row (for example MDD when no local
parser is configured), plots label that entire metric unavailable and do not
silently drop rows or change the scored sample.

## Pilot and full run

1. Freeze source snapshots, prompt version, exact model IDs, sampling settings,
   collection dates, parser version, and annotation rubric.
2. Run 12 balanced prompts through all four providers. Audit JSON validity,
   truncation, cost, retry rate, latency, and evaluator failures.
3. Correct protocol defects before collection; never silently repair collected
   model outputs.
4. Run the primary manifest, checksum-transfer closed shards, evaluate on the
   PC, then prepare the blinded rating sample.
5. Select the confirmatory condition explicitly, run paired tests with Holm
   correction, and retain raw p-values, adjusted p-values, effect sizes, and
   95% confidence intervals.
6. Run reliability and stochastic subsets only after the primary collection is
   complete, using distinct run IDs.

The local smoke test exercises the laptop-to-PC boundary with four mocked
providers and no credentials or network calls:

```powershell
python -m hsk_agent_bench.cli dry-run `
  --config configs/experiment.yaml --device configs/devices/pc.yaml
```

It refuses to overwrite an existing `outputs/dry-run`; move or inspect that
directory before rerunning.

## Troubleshooting and limitations

- A missing or mismatched `.sha256` sidecar is an incomplete/corrupt shard;
  restore the transferred pair rather than editing it.
- “Device profile does not allow stage” means the command is on the wrong
  machine role. Generation is laptop-only; evaluation/analysis are PC-only.
- Placeholder model IDs or illustrative HSK data are valid for smoke tests
  only. Freeze exact versions and licensed sources before the pilot.
- Parser failure yields missing dependency distance with diagnostic metadata;
  it is never converted into a grammar error.
- Only one Chinese evaluator limits external validity. Keep ratings blinded,
  re-rate the preregistered subset, and report intra-rater reliability.
- Two models per origin do not support claims about all Chinese or global LLMs.
  Phrase conclusions as applying to the evaluated model set.
- API seeds do not guarantee identical samples across vendors. Stored request
  parameters, hashes, timestamps, versions, and raw responses provide an audit
  trail rather than cross-vendor bitwise reproducibility.
