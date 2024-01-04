import os
from dataclasses import dataclass
from typing import Union

import yaml


@dataclass
class ExperimentMetadata:
    artifact_location: str
    creation_time: int
    experiment_id: str
    last_update_time: int
    lifecycle_stage: str
    name: str


@dataclass
class RunMetadata:
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


def load_mlflow_meta(metadata_path: Union[str, os.PathLike]) -> Union[ExperimentMetadata, RunMetadata]:
    with open(metadata_path, "r") as f:
        data = yaml.safe_load(f)
        try:
            if "run_id" in data.keys():
                return RunMetadata(**data)
            elif "experiment_id" in data.keys():
                return ExperimentMetadata(**data)
            else:
                raise ValueError(f"Could not determine metadata type from {metadata_path}")
        except TypeError:
            raise ValueError(f"Failed to instantiate mlflow metadata from {metadata_path}")

