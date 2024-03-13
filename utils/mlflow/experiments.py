import functools
import os.path
import shutil
from pathlib import Path
from typing import Union

import mlflow
from loguru import logger
from mlflow import MlflowException

from utils.mlflow.domain import EvaluateModelRequest

MLFLOW_RUN_PARENT_TAG_KEY: str = "parent"
MLFLOW_RUN_PARENT_VALUE: str = "y"


def mlflow_context(func):
    """Designed to wrap Model::evaluate, starts and stops MLFlow run, populates experiment_name/id, run_name/id
    tags, uses sklearn autolog to save models
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert len(args) == 6, "Passed args do not match Model::evaluate"
        evaluate_request = EvaluateModelRequest.from_tuple(args)

        default_tags: dict[str, str] = evaluate_request.run_tags
        exp_id: str = _resolve_experiment_id(evaluate_request)

        mlflow.autolog(log_models=False)

        with mlflow.start_run(experiment_id=exp_id,
                              run_name=evaluate_request.experiment_name,
                              tags=default_tags) as run:
            result = func(*args, **kwargs)

        _zip_mlflow_output(evaluate_request)

        return result

    return wrapper


def _resolve_experiment_id(eval_request: EvaluateModelRequest) -> str:
    # TODO probably not very useful as DVC seems to reset state of outs before applying so previous exps may get lost
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


def _zip_mlflow_output(eval_request: EvaluateModelRequest):
    exp_location: os.PathLike | str = eval_request.mlruns_location

    temp_dir = os.path.join(exp_location, "__temp__")
    os.mkdir(temp_dir)
    logger.debug(f"Creating temp directory: {temp_dir}")

    try:
        for item in os.listdir(exp_location):
            if item not in {"__temp__", ".gitignore"}:
                shutil.move(os.path.join(exp_location, item), os.path.join(temp_dir, item))

        logger.debug(f"Moved files from {exp_location} to temp directory: {temp_dir}. Zipping temp dir to out_mlflow.zip")
        shutil.make_archive(os.path.join(exp_location, "out_mlflow"), "zip", temp_dir)
    finally:
        logger.debug(f"Removing temp directory")
        shutil.rmtree(temp_dir)

