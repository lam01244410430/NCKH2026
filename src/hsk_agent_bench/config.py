"""YAML configuration; credentials are environment variable names only."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field, model_validator

from .schemas import HSKLevel, ModelOrigin, NonEmpty, Provider, Record, TaskType


class ModelCapabilities(Record):
    seed: bool = False
    json_output: bool = False
    temperature: bool = True
    top_p: bool = True


class ModelConfig(Record):
    family: NonEmpty
    origin: ModelOrigin
    provider: Provider
    model_id: NonEmpty
    api_key_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*_API_KEY$")]
    base_url: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    enabled: bool = True


class ModelRegistry(Record):
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_families(self):
        families = [model.family for model in self.models]
        if len(families) != len(set(families)):
            raise ValueError("model family names must be unique")
        return self


class RunConfig(Record):
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(ge=0, le=1)
    seeds: list[int] = Field(min_length=1)
    fraction: float = Field(default=1, gt=0, le=1)
    replicates: int = Field(default=1, gt=0)
    max_output_tokens: dict[TaskType, Annotated[int, Field(gt=0)]]


class ExperimentConfig(Record):
    name: NonEmpty
    model_registry: Path
    models: list[ModelConfig] = Field(default_factory=list)
    levels: list[HSKLevel] = Field(min_length=1)
    tasks: list[TaskType] = Field(min_length=1)
    topics: list[NonEmpty] = Field(min_length=1)
    primary: RunConfig
    stochastic: RunConfig

    @model_validator(mode="after")
    def unique_grid_and_budgets(self):
        for values in (self.levels, self.tasks, self.topics):
            if len(values) != len(set(values)):
                raise ValueError("experiment grid values must be unique")
        for run in (self.primary, self.stochastic):
            if not set(self.tasks) <= run.max_output_tokens.keys():
                raise ValueError("every task requires a maximum output token budget")
        return self


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Resolve the registry next to the experiment, independent of process cwd."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(data)
    registry_path = path.parent / config.model_registry
    registry = ModelRegistry.model_validate(yaml.safe_load(registry_path.read_text(encoding="utf-8")))
    config.models = registry.models
    return config
