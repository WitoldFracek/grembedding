import os
from dataclasses import dataclass, asdict
from typing import Union, Literal, Optional
from loguru import logger

import yaml

from config.mlflow import MLRUNS_STORAGE_ROOT
from stages.models.Model import Model


class SavableYamlDataclassMixin:
    def to_yaml(self):
        data = asdict(self)
        return yaml.dump(data, sort_keys=False)

    def save(self, path):
        full_path = os.path.join(path, 'meta.yaml')
        yaml_data = self.to_yaml()
        with open(full_path, 'w') as file:
            file.write(yaml_data)


@dataclass
class ExperimentMetadata(SavableYamlDataclassMixin):
    artifact_location: str
    creation_time: int
    experiment_id: str
    last_update_time: int
    lifecycle_stage: str
    name: str


@dataclass
class RunMetadata(SavableYamlDataclassMixin):
    artifact_uri: str
    end_time: int
    entry_point_name: str
    experiment_id: str
    lifecycle_stage: str
    run_id: str
    run_name: str
    run_uuid: str
    source_name: str
    source_type: int
    source_version: str
    start_time: int
    status: int
    tags: any
    user_id: str


@dataclass
class EvaluateModelRequest:
    model_instance: Model
    dataset: str
    datacleaner: str
    vectorizer: str
    params_name: str
    params: dict[str, Union[int, float, str]]

    @property
    def model_name(self):
        return self.model_instance.__class__.__name__

    @property
    def experiment_name(self):
        return f"{self.dataset}_{self.datacleaner}_{self.vectorizer}_{self.model_name}_{self.params_name}"

    @property
    def mlruns_location(self):
        return os.path.join(MLRUNS_STORAGE_ROOT, *self.experiment_name.split("_"))

    @property
    def run_tags(self):
        return {
            "dataset": self.dataset,
            "vectorizer": self.vectorizer,
            "datacleaner": self.datacleaner,
            "model": self.model_name
        }

    @staticmethod
    def from_tuple(args: tuple) -> "EvaluateModelRequest":
        return EvaluateModelRequest(*args)


def load_mlflow_meta(metadata_path: Union[str, os.PathLike], errors: Literal['omit', 'raise'] = "omit") -> Optional[
    Union[ExperimentMetadata, RunMetadata]]:
    if not os.path.exists(metadata_path):
        logger.warning(f'file "{metadata_path}" does not exist')
        return None
    with open(metadata_path, "r") as f:
        data = yaml.safe_load(f)
        try:
            if "run_id" in data.keys():
                return RunMetadata(**data)
            elif "experiment_id" in data.keys():
                return ExperimentMetadata(**data)
            else:
                if errors == "raise":
                    raise ValueError(f"Could not determine metadata type from {metadata_path}")
                elif errors == "omit":
                    logger.warning(f"Not known metadata type for {metadata_path}")
                    return None
        except TypeError:
            raise ValueError(f"Failed to instantiate mlflow metadata from {metadata_path}")
