import functools
import os.path
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

        mlflow.autolog(log_models=True)

        with mlflow.start_run(experiment_id=exp_id,
                              run_name=evaluate_request.experiment_name,
                              tags=default_tags) as run:
            result = func(*args, **kwargs)

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
