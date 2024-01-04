import copy
import functools
import os.path
from typing import Union

import mlflow
from mlflow import MlflowException
from pathlib import Path
from loguru import logger

from mlflow.entities import Run, ViewType

MLFLOW_RUN_PARENT_TAG_KEY: str = "parent"
MLFLOW_RUN_PARENT_VALUE: str = "y"


def mlflow_context(func):
    """Designed to wrap Model::evaluate, starts and stops MLFlow run, populates experiment_name/id, run_name/id"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO shady
        assert len(args) == 6, "Passed args do not match Model::evaluate"
        model_instance, dataset, datacleaner, vectorizer, params_name, params = args

        class_name: str = model_instance.__class__.__name__
        run_name: str = f"{datacleaner}-{vectorizer}-{class_name}"
        experiment_name: str = dataset

        default_tags: dict[str, str] = {
            "dataset": dataset,
            "vectorizer": vectorizer,
            "datacleaner": datacleaner,
            "model": class_name
        }

        # Get experiment id
        try:
            exp_id = mlflow.create_experiment(name=experiment_name)
            logger.info(f"Created new MLFlow experiment: {experiment_name} (id: {exp_id})")
        except MlflowException as e:
            exp = mlflow.get_experiment_by_name(experiment_name)
            logger.info(f"Reusing mlflow experiment with id: {exp.experiment_id}")
            exp_id = exp.experiment_id

        runs: list[Run] = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.vectorizer = '{vectorizer}' and tags.dataset = '{dataset}' and"
                          f" tags.datacleaner = '{datacleaner}' and tags.model = '{class_name}'",
            run_view_type=ViewType.ALL, output_format="list")
        eligible_parents: list[Run] = [r for r in runs if MLFLOW_RUN_PARENT_TAG_KEY in r.data.tags.keys()]
        logger.debug(f"[Parent run search] Found {len(runs)} in total, {len(eligible_parents)} are parents")

        if len(eligible_parents) == 1:
            # Parent run exists
            parent_run = eligible_parents[0]
        elif len(eligible_parents) == 0:
            parent_tags = copy.deepcopy(default_tags)
            parent_tags[MLFLOW_RUN_PARENT_TAG_KEY] = MLFLOW_RUN_PARENT_VALUE
            parent_run = mlflow.start_run(experiment_id=exp_id, run_name=f"Series: {run_name}", tags=parent_tags)
        else:
            raise ValueError(f"Invalid MLFlow configuration - "
                             f"more than 1 qualifying parent run for run_name={run_name}, experiment={experiment_name}")

        # TODO: probably no real benefit for nested runs since these are not actually nested in the filetree
        with mlflow.start_run(run_id=parent_run.info.run_id, nested=True) as active_parent_run:
            with mlflow.start_run(run_name=run_name, experiment_id=exp_id, tags=default_tags, nested=True) as run:
                result = func(model_instance, dataset, datacleaner, vectorizer, params_name, params)
                return result

    return wrapper
