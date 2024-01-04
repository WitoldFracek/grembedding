import copy
import functools
import os.path
import shutil
import uuid
from dataclasses import dataclass
from typing import Union

import mlflow
from mlflow import MlflowException
from pathlib import Path
from loguru import logger

from mlflow.entities import Run, ViewType

from config.MlflowConfig import MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT
from stages.models.Model import Model

MLFLOW_RUN_PARENT_TAG_KEY: str = "parent"
MLFLOW_RUN_PARENT_VALUE: str = "y"


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
        return f"{self.dataset}_{self.datacleaner}_{self.vectorizer}_{self.model_name}"

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


def mlflow_context(func):
    """Designed to wrap Model::evaluate, starts and stops MLFlow run, populates experiment_name/id, run_name/id"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO shady
        assert len(args) == 6, "Passed args do not match Model::evaluate"
        evaluate_request = EvaluateModelRequest.from_tuple(args)

        experiment_fqn: str = evaluate_request.experiment_name
        run_name: str = uuid.uuid4().hex  # TODO random for now
        default_tags: dict[str, str] = evaluate_request.run_tags

        # runs: list[Run] = mlflow.search_runs(
        #     experiment_ids=[exp_id],
        #     filter_string=f"tags.vectorizer = '{vectorizer}' and tags.dataset = '{dataset}' and"
        #                   f" tags.datacleaner = '{datacleaner}' and tags.model = '{class_name}'",
        #     run_view_type=ViewType.ALL, output_format="list")
        # eligible_parents: list[Run] = [r for r in runs if MLFLOW_RUN_PARENT_TAG_KEY in r.data.tags.keys()]
        # logger.debug(f"[Parent run search] Found {len(runs)} in total, {len(eligible_parents)} are parents")

        # if len(eligible_parents) == 1:
        #     # Parent run exists
        #     parent_run = eligible_parents[0]
        # elif len(eligible_parents) == 0:
        #     parent_tags = copy.deepcopy(default_tags)
        #     parent_tags[MLFLOW_RUN_PARENT_TAG_KEY] = MLFLOW_RUN_PARENT_VALUE
        #     parent_run = mlflow.start_run(experiment_id=exp_id, run_name=f"Series: {run_name}", tags=parent_tags)
        # else:
        #     raise ValueError(f"Invalid MLFlow configuration - "
        #                      f"more than 1 qualifying parent run for run_name={run_name}, experiment={experiment_name}")

        # probably no real benefit for nested runs since these are not actually nested in the filetree
        # with mlflow.start_run(run_id=parent_run.info.run_id, nested=True) as active_parent_run:

        exp_id: str = _resolve_experiment_id(evaluate_request)

        with mlflow.start_run(experiment_id=exp_id, tags=default_tags) as run:
            result = func(*args, **kwargs)
            run_id = run.info.run_id

        _sync_view_mlruns_dir(evaluate_request, exp_id, run_id)
        return result

    return wrapper


def _resolve_experiment_id(eval_request: EvaluateModelRequest) -> str:
    relative_location: Union[str, os.PathLike] = eval_request.mlruns_location
    exp_location_uri = Path.cwd().joinpath(relative_location).as_uri()
    mlflow.set_tracking_uri(exp_location_uri)
    logger.debug(f"Resolved experiment location: {exp_location_uri}")

    try:
        exp_id = mlflow.create_experiment(name=eval_request.experiment_name)
        logger.info(f"Created new MLFlow experiment: {eval_request.experiment_name} (id: {exp_id})")
    except MlflowException as e:
        exp = mlflow.get_experiment_by_name(eval_request.experiment_name)
        logger.info(f"Reusing mlflow experiment with id: {exp.experiment_id}")
        exp_id = exp.experiment_id
    return exp_id


def _sync_view_mlruns_dir(eval_request: EvaluateModelRequest, exp_id: str, run_id: str) -> None:
    run_artifact_storage_dir = os.path.join(eval_request.mlruns_location)
    dest_location = os.path.join(MLRUNS_VIEW_ROOT, exp_id)
    shutil.copytree(run_artifact_storage_dir, dest_location, dirs_exist_ok=True)
    logger.debug(f"Synced mlruns dir {run_artifact_storage_dir} to {MLRUNS_VIEW_ROOT}")
