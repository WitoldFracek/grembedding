import os
from dataclasses import dataclass, asdict
from typing import Union, Literal, Optional
from loguru import logger

import yaml


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


def load_mlflow_meta(metadata_path: Union[str, os.PathLike], errors: Literal['omit', 'raise'] = "omit") -> Optional[
    Union[ExperimentMetadata, RunMetadata]]:
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
