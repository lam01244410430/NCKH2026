from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError
from hsk_agent_bench.config import ModelConfig, load_experiment_config
from hsk_agent_bench.schemas import PromptRecord

ROOT = Path(__file__).resolve().parents[1]


def prompt_data():
    return dict(prompt_id="reading-3-daily", task="reading", hsk_level=3,
                topic="daily life", payload=dict(task="reading", target_length_chars=300))


def test_example_registry_and_sampling_protocol():
    config = load_experiment_config(ROOT / "configs/experiment.yaml")
    enabled = [model for model in config.models if model.enabled]
    assert len(enabled) == len({model.family for model in enabled}) == 4
    assert {model.origin for model in enabled} == {"china", "global"}
    assert {model.provider for model in enabled} == {"openai", "gemini", "qwen", "deepseek"}
    assert all(model.api_key_env.endswith("_API_KEY") for model in enabled)
    assert config.levels == [3, 4, 5, 6]
    assert len(config.topics) == 12
    assert (config.primary.temperature, config.primary.top_p, config.primary.seeds) == (0, 1, [20261025])
    assert (config.stochastic.temperature, config.stochastic.top_p) == (0.7, 0.9)
    assert config.stochastic.seeds == [20261025, 20261026, 20261027]
    assert config.stochastic.fraction == 0.25


@pytest.mark.parametrize("level", [0, 2, 7, "3", True])
def test_prompt_rejects_invalid_core_levels(level):
    data = prompt_data()
    data["hsk_level"] = level
    with pytest.raises(ValidationError):
        PromptRecord.model_validate(data)


def test_prompt_requires_id_and_matching_task_payload():
    data = prompt_data()
    del data["prompt_id"]
    with pytest.raises(ValidationError):
        PromptRecord.model_validate(data)
    data = prompt_data()
    data["task"] = "exam_generation"
    with pytest.raises(ValidationError):
        PromptRecord.model_validate(data)


def test_valid_payload_roundtrips():
    record = PromptRecord.model_validate(prompt_data())
    assert PromptRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("secret", ["sk-example-secret", "actual password", "AIzaExampleSecret"])
def test_model_rejects_plaintext_keys(secret):
    data = yaml.safe_load((ROOT / "configs/models.example.yaml").read_text())["models"][0]
    data["api_key_env"] = secret
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(data)
    data["api_key_env"] = "OPENAI_API_KEY"
    data["api_key"] = secret
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(data)


def test_loader_resolves_registry_relative_to_config(tmp_path):
    source = ROOT / "configs"
    (tmp_path / "experiment.yaml").write_text((source / "experiment.yaml").read_text())
    (tmp_path / "models.example.yaml").write_text((source / "models.example.yaml").read_text())
    assert len(load_experiment_config(tmp_path / "experiment.yaml").models) == 4


@pytest.mark.parametrize("task,payload", [
    ("grammar_explanation", {"task": "grammar_explanation", "grammar_point": "把字句"}),
    ("exam_generation", {"task": "exam_generation", "num_items": 2}),
])
def test_task_payload_discriminator_preserves_task_fields(task, payload):
    data = prompt_data()
    data.update(task=task, payload=payload)
    record = PromptRecord.model_validate(data)
    assert record.payload.task == task
    assert PromptRecord.model_validate_json(record.model_dump_json()) == record


def test_loader_rejects_missing_task_budget(tmp_path):
    data = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    del data["primary"]["max_output_tokens"]["reading"]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="token budget"):
        load_experiment_config(path)
