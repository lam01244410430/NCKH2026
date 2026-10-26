"""Typer entry point for the auditable laptop-to-PC benchmark workflow."""

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
import yaml

from .analysis.aggregate import join_analysis_tables
from .analysis.hypothesis_tests import holm_adjust, paired_origin_test
from .analysis.plots import build_all_figures
from .config import ExperimentConfig, RunConfig, load_experiment_config
from .evaluators.grammar import GrammarCatalog
from .evaluators.human_ratings import blind_rating_sheet, sample_for_human_rating
from .evaluators.lexical import HSKLexicon
from .evaluators.materials import project_material_text
from .evaluators.pipeline import EvaluationConfig, evaluate_shard, load_lexicon, load_parser_config
from .evaluators.syntax import DependencyToken, ParsedDocument, ParsedSentence
from .io import verify_shard, write_jsonl_shard_atomic
from .prompts.exam_generation import render_exam_prompt
from .prompts.grammar_explanation import render_grammar_prompt
from .prompts.manifest import build_manifest
from .prompts.reading import render_reading_prompt
from .runner import run_manifest, select_prompts
from .schemas import (
    GenerationParameters, GenerationRecord, GenerationRequest, MetricRecord, PromptRecord, TokenUsage,
)


app = typer.Typer(no_args_is_help=True, help="Reproducible HSK-AgentBench pipeline.")
PathOption = Annotated[Path, typer.Option()]
RENDERERS = {
    "reading": render_reading_prompt,
    "grammar_explanation": render_grammar_prompt,
    "exam_generation": render_exam_prompt,
}


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(2)


def _profile(path: Path, stage: str) -> dict:
    try:
        profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        _fail(f"cannot read device profile: {path}: {exc}")
    if not isinstance(profile, dict) or not isinstance(profile.get("allowed_stages"), list):
        _fail(f"invalid device profile: {path}")
    if stage not in profile["allowed_stages"]:
        _fail(f"device profile '{profile.get('name', path.stem)}' does not allow stage '{stage}'")
    return profile


def _json_summary(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"run summary already exists: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_records(path: Path, record_type):
    if not verify_shard(path):
        raise ValueError(f"shard checksum integrity failure: {path}")
    return [record_type.model_validate_json(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines()]


def _request_id(prompt: PromptRecord, model_id: str, seed: int, replicate: int) -> str:
    value = f"{prompt.prompt_id}\0{model_id}\0{seed}\0{replicate}".encode()
    return "g-" + hashlib.sha256(value).hexdigest()[:32]


def _render_requests(config: ExperimentConfig, manifest: list[PromptRecord],
                     run: RunConfig, run_id: str) -> list[GenerationRequest]:
    requests = []
    for prompt in select_prompts(manifest, run):
        messages = RENDERERS[str(prompt.task)](prompt)
        for model in (item for item in config.models if item.enabled):
            for seed_index, seed in enumerate(run.seeds):
                for repeat in range(run.replicates):
                    replicate = seed_index * run.replicates + repeat
                    requests.append(GenerationRequest(
                        generation_id=_request_id(prompt, model.model_id, seed, replicate),
                        prompt_id=prompt.prompt_id, task=prompt.task, hsk_level=prompt.hsk_level,
                        family=model.family, origin=model.origin, provider=model.provider,
                        model_id=model.model_id, prompt_hash=prompt.prompt_hash, messages=messages,
                        parameters=GenerationParameters(
                            temperature=run.temperature, top_p=run.top_p,
                            max_output_tokens=run.max_output_tokens[prompt.task],
                            seed=seed if model.capabilities.seed else None,
                        ), run_id=run_id, replicate=replicate,
                    ))
    return requests


def _load_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input table does not exist: {path}")
    return pd.read_csv(path)


def _validated_material_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw task JSON and attach its validated learner-facing projection."""
    required = {"task", "hsk_level", "text"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"material table missing columns: {sorted(missing)}")
    result = frame.copy()
    projected = [project_material_text(task, text, hsk_level=int(level))
                 for task, text, level in zip(result.task, result.text, result.hsk_level)]
    if "material_text" in result and result.material_text.tolist() != projected:
        raise ValueError("material_text does not match the validated projection of raw text")
    result["material_text"] = projected
    return result


def _condition(frame: pd.DataFrame, run_id: str | None, replicate: int | None) -> pd.DataFrame:
    selected = frame
    for column, value in (("run_id", run_id), ("replicate", replicate)):
        if column in selected:
            choices = selected[column].dropna().unique()
            if value is None and len(choices) > 1:
                raise ValueError(f"select exactly one analysis condition with --{column.replace('_', '-')}")
            if value is not None:
                selected = selected.loc[selected[column] == value]
    if selected.empty:
        raise ValueError("selected analysis condition is empty")
    return selected.copy()


def _stability_condition(frame: pd.DataFrame, run_id: str | None) -> pd.DataFrame:
    """Select one run while retaining an identical repeat roster in every cell."""
    selected = frame
    if "run_id" in selected:
        choices = selected.run_id.dropna().unique()
        if run_id is None and len(choices) > 1:
            raise ValueError("select exactly one stability run with --run-id")
        if run_id is not None:
            selected = selected.loc[selected.run_id == run_id]
    required = {"prompt_id", "model_id", "replicate"}
    if selected.empty or not required <= set(selected) or selected[list(required)].isna().any().any():
        raise ValueError("stability plotting requires prompt, model, and replicate identifiers")
    if selected.duplicated(["prompt_id", "model_id", "replicate"]).any():
        raise ValueError("duplicate stability prompt/model/replicate cell")
    rosters = selected.groupby(["prompt_id", "model_id"]).replicate.apply(
        lambda values: tuple(sorted(values.unique())))
    if len(set(rosters)) != 1 or len(rosters.iloc[0]) < 2:
        raise ValueError("stability plotting requires the same two or more repeats in every prompt/model cell")
    return selected.copy()


@app.command("validate-data")
def validate_data(config: PathOption, device: PathOption, vocab: PathOption,
                  grammar: PathOption, summary: PathOption) -> None:
    """Validate canonical source files before prompt construction."""
    _profile(device, "validate-data")
    load_experiment_config(config)
    problems = []
    if not vocab.is_file():
        problems.append(f"missing vocabulary source: {vocab}")
    if not grammar.is_file():
        problems.append(f"missing grammar source: {grammar}")
    if problems:
        _fail("\n".join(problems))
    try:
        lexicon = load_lexicon(vocab)
    except (OSError, ValueError, TypeError) as exc:
        _fail(f"invalid vocabulary source: {exc}")
    catalog = GrammarCatalog.from_canonical(json.loads(grammar.read_text(encoding="utf-8")))
    _json_summary(summary, {"stage": "validate-data", "vocabulary_entries": len(lexicon),
                            "grammar_entries": len(catalog.rules), "grammar_sha256": catalog.sha256})
    typer.echo(summary)


@app.command("build-prompts")
def build_prompts(config: PathOption, device: PathOption, output: PathOption) -> None:
    """Build and checksum the deterministic prompt manifest."""
    _profile(device, "build-prompts")
    records = build_manifest(load_experiment_config(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum = write_jsonl_shard_atomic(records, output)
    _json_summary(output.with_suffix(".summary.json"),
                  {"stage": "build-prompts", "records": len(records), "output": output,
                   "checksum": checksum})
    typer.echo(f"manifest={output}\nchecksum={checksum}\nrecords={len(records)}")


@app.command("generate")
def generate(config: PathOption, device: PathOption, manifest: PathOption,
             output_dir: PathOption,
             protocol: Annotated[Literal["primary", "stochastic"], typer.Option("--protocol")],
             preview_output: Annotated[Path | None, typer.Option()] = None,
             dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
             run_id: Annotated[str, typer.Option()] = "primary") -> None:
    """Generate provider outputs, or render a network-free request preview."""
    profile = _profile(device, "generate")
    experiment = load_experiment_config(config)
    prompts = _read_records(manifest, PromptRecord)
    run = experiment.primary if protocol == "primary" else experiment.stochastic
    if dry_run:
        target = preview_output or manifest.with_name("request-preview.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        requests = _render_requests(experiment, prompts, run, run_id)
        checksum = write_jsonl_shard_atomic(requests, target)
        _json_summary(target.with_suffix(".summary.json"), {
            "stage": "generate-preview", "protocol": protocol, "requests": len(requests),
            "preview": target, "checksum": checksum,
        })
        typer.echo(f"preview={target}\nchecksum={checksum}")
        return
    placeholders = [model.family for model in experiment.models
                    if model.enabled and model.model_id.startswith("FREEZE_")]
    if placeholders:
        _fail("freeze exact model IDs before live generation: " + ", ".join(placeholders))
    summary = asyncio.run(run_manifest(
        prompts, experiment.models, run, output_dir,
        run_id=run_id, shard_namespace=profile["shard_namespace"],
    ))
    summary_path = summary.output_dir / "run-summary.json"
    _json_summary(summary_path, summary.model_dump(mode="json"))
    typer.echo(summary_path)


@app.command("evaluate")
def evaluate(input_path: Annotated[Path, typer.Option("--input")], output: PathOption,
             vocab: PathOption, grammar: Annotated[Path | None, typer.Option()] = None,
             parser_config: Annotated[Path | None, typer.Option("--parser-config")] = None,
             device: PathOption = Path("configs/devices/pc.yaml")) -> None:
    """Evaluate one verified generation shard on the PC profile."""
    profile = _profile(device, "evaluate")
    catalog = None if grammar is None else GrammarCatalog.from_canonical(
        json.loads(grammar.read_text(encoding="utf-8")))
    parser = None if parser_config is None else load_parser_config(parser_config)
    result = evaluate_shard(input_path, output, EvaluationConfig(
        lexicon=load_lexicon(vocab), grammar_catalog=catalog, parser=parser,
        device=profile.get("device", "cpu")))
    _json_summary(output.with_suffix(".summary.json"), result.model_dump(mode="json"))
    typer.echo(output)


@app.command("prepare-ratings")
def prepare_ratings(input_path: Annotated[Path, typer.Option("--input")], output: PathOption,
                    key_output: Annotated[Path, typer.Option("--key-output")], device: PathOption,
                    fraction: Annotated[float, typer.Option()] = .25) -> None:
    """Create a blinded rating sheet and a separately stored coordinator key."""
    _profile(device, "prepare-ratings")
    frame = _validated_material_frame(_load_frame(input_path))
    sample = sample_for_human_rating(frame, fraction=fraction)
    sheet, key = blind_rating_sheet(sample, display_text_column="material_text")
    for path in (output, key_output):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output, index=False)
    key.mapping.to_csv(key_output, index=False)
    _json_summary(output.with_suffix(".summary.json"), {"stage": "prepare-ratings", "rows": len(sheet),
                                                        "rating_sheet": output, "key": key_output})
    typer.echo(output)


@app.command("analyze")
def analyze(input_path: Annotated[Path, typer.Option("--input")], output: PathOption,
            config: PathOption, device: PathOption,
            metric: Annotated[list[str], typer.Option("--metric")],
            family: Annotated[str, typer.Option("--family")],
            run_id: Annotated[str | None, typer.Option()] = None,
            replicate: Annotated[int | None, typer.Option()] = None) -> None:
    """Run prompt-paired inference for one explicitly selected condition."""
    _profile(device, "analyze")
    experiment = load_experiment_config(config)
    frame = _condition(_load_frame(input_path), run_id, replicate)
    expected = [model.model_id for model in experiment.models if model.enabled]
    if set(frame.model_id) != set(expected):
        raise ValueError("analysis table must contain every enabled frozen-config model ID")
    if not family.strip():
        raise ValueError("metric family must be nonempty")
    if not metric or len(metric) != len(set(metric)):
        raise ValueError("declare one or more unique metrics")
    results = holm_adjust([paired_origin_test(frame, name) for name in metric], family=family)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    pd.DataFrame([asdict(result) for result in results]).to_csv(output, index=False)
    _json_summary(output.with_suffix(".summary.json"), {"stage": "analyze", "table": output,
                                                        "model_order": expected,
                                                        "family": family, "metrics": metric})
    typer.echo(output)


@app.command("plot")
def plot(input_path: Annotated[Path, typer.Option("--input")], output_dir: PathOption,
         config: PathOption, device: PathOption,
         run_id: Annotated[str | None, typer.Option()] = None,
         replicate: Annotated[int | None, typer.Option()] = None,
         include_repeats_for_stability: Annotated[
             bool, typer.Option("--include-repeats-for-stability")
         ] = False) -> None:
    """Render deterministic paper figures for one condition."""
    _profile(device, "plot")
    experiment = load_experiment_config(config)
    source = _load_frame(input_path)
    if include_repeats_for_stability:
        if replicate is not None:
            raise ValueError("--replicate is incompatible with --include-repeats-for-stability")
        frame = _stability_condition(source, run_id)
    else:
        frame = _condition(source, run_id, replicate)
    order = [model.model_id for model in experiment.models if model.enabled]
    figure_names = (
        "overall_heatmap", "task_bars", "hsk_level_lines", "spillover_distribution",
        "model_stability", "judge_human_agreement",
    )
    expected_paths = [output_dir / "run-summary.json"] + [
        output_dir / f"{name}.{suffix}" for name in figure_names for suffix in ("svg", "png")
    ]
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        _fail("plot output already exists: " + ", ".join(map(str, existing)))
    paths = build_all_figures(frame, output_dir, model_order=order)
    _json_summary(output_dir / "run-summary.json", {"stage": "plot", "figures": paths,
                                                     "model_order": order,
                                                     "include_repeats_for_stability": include_repeats_for_stability})
    typer.echo("\n".join(map(str, paths)))


class _DryParser:
    name = "dry-run-parser"
    version = "1"
    device = "cpu"

    def parse(self, text: str) -> ParsedDocument:
        return ParsedDocument(sentences=[ParsedSentence(tokens=[
            DependencyToken(index=1, text="我", head=2, relation="nsubj"),
            DependencyToken(index=2, text="学习", head=0, relation="root"),
        ])])


@app.command("dry-run")
def dry_run(config: PathOption, device: PathOption,
            output_dir: Annotated[Path, typer.Option()] = Path("outputs/dry-run")) -> None:
    """Exercise the complete artifact boundary with deterministic local fixtures."""
    _profile(device, "dry-run")
    experiment = load_experiment_config(config)
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _fail(f"output directory already exists: {output_dir}")
    complete_manifest = build_manifest(experiment)
    prompts = [next(prompt for prompt in complete_manifest if prompt.task == task)
               for task in ("reading", "grammar_explanation", "exam_generation")]
    models = [model for model in experiment.models if model.enabled]
    if {model.provider for model in models} != {"openai", "gemini", "qwen", "deepseek"}:
        raise ValueError("dry-run requires one enabled model for each of the four providers")
    manifest = output_dir / "manifest" / "fixture.jsonl"
    manifest.parent.mkdir(parents=True)
    write_jsonl_shard_atomic(prompts, manifest)
    requests = _render_requests(experiment, prompts, experiment.primary.model_copy(
        update={"seeds": [20261025], "replicates": 1}), "dry-run")
    passages = {"china": "我学习中文，因为我喜欢中国文化。", "global": "我学习中文，我喜欢文化。"}
    def fixture_text(request):
        passage = passages[str(request.origin)]
        if request.task == "reading":
            payload = {"title": "学习中文", "passage": passage, "declared_hsk_level": request.hsk_level,
                       "used_required_vocabulary": [], "used_required_grammar": []}
        elif request.task == "grammar_explanation":
            payload = {"explanation": "因为表示原因。", "form": "因为A，所以B",
                       "usage_constraints": ["说明原因"], "examples": [passage],
                       "incorrect_example": {"incorrect": "因为下雨，但是我在家。",
                                             "correction": "因为下雨，所以我在家。"},
                       "practice": {"question": "填空：___下雨，所以我在家。", "answer": "因为"}}
        else:
            payload = {"source_passage": passage, "items": [{"question": "谁学习中文？",
                       "options": ["我", "你", "他", "老师"], "answer": "我",
                       "rationale": "材料中的主语是我。", "target_vocabulary": [], "target_grammar": []}]}
        return json.dumps(payload, ensure_ascii=False)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [GenerationRecord(
        **request.model_dump(), text=fixture_text(request), raw_response={"fixture": True},
        requested_at=now, request_id=f"fixture-{request.generation_id}", latency_ms=0,
        usage=TokenUsage(), finish_reason="stop", retry_count=0, seed_applied=False,
    ) for request in requests]
    raw = output_dir / "raw" / "fixture.jsonl"
    raw.parent.mkdir(parents=True)
    write_jsonl_shard_atomic(records, raw)
    lexicon = HSKLexicon({"我": 1, "学习": 2, "中文": 2, "因为": 3, "喜欢": 2,
                          "中国": 2, "文化": 4})
    metric_shard = output_dir / "metrics" / "metrics.jsonl"
    evaluate_shard(raw, metric_shard, EvaluationConfig(lexicon=lexicon, parser=_DryParser()))
    metric_records = _read_records(metric_shard, MetricRecord)
    generations = _validated_material_frame(pd.DataFrame([
        record.model_dump(mode="json") for record in records
    ]))
    metrics = pd.DataFrame([record.model_dump(mode="json") for record in metric_records])
    table = join_analysis_tables(generations, metrics)
    # This fixture is exactly one run/replicate condition. Preserve frozen config order.
    table = _condition(table, "dry-run", 0)
    order = [model.model_id for model in models]
    table.attrs["model_order"] = order
    metric_table = output_dir / "metrics" / "metrics.csv"
    table.to_csv(metric_table, index=False)
    sample = sample_for_human_rating(generations, fraction=1)
    sheet, key = blind_rating_sheet(sample, display_text_column="material_text")
    rating_sheet = output_dir / "ratings" / "rating_sheet.csv"
    rating_sheet.parent.mkdir(parents=True)
    sheet.to_csv(rating_sheet, index=False)
    coordinator = output_dir / "coordinator" / "blinding_key.csv"
    coordinator.parent.mkdir(parents=True)
    key.mapping.to_csv(coordinator, index=False)
    result = holm_adjust([paired_origin_test(table, "v_in_token", seed=20261025)], "lexical")[0]
    statistical = output_dir / "tables" / "statistics.csv"
    statistical.parent.mkdir(parents=True)
    pd.DataFrame([asdict(result)]).to_csv(statistical, index=False)
    figures = build_all_figures(table, output_dir / "figures", model_order=order)
    summary = {
        "stage": "dry-run", "providers": ["openai", "gemini", "qwen", "deepseek"],
        "verified_manifest_shard": str(manifest), "verified_fixture_shard": str(raw),
        "metric_table": str(metric_table),
        "rating_sheet": str(rating_sheet), "coordinator_key": str(coordinator),
        "statistical_table": str(statistical), "figures": [str(path) for path in figures],
    }
    _json_summary(output_dir / "run_summary.json", summary)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
