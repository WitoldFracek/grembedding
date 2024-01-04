import copy
import glob
import os
import shutil
from typing import Union

from loguru import logger

from config.mlflow import MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT
from utils.mlflow.domain import load_mlflow_meta, RunMetadata, ExperimentMetadata


def traverse(mlruns_store_dir: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT) -> list[Union[str, os.PathLike]]:
    return glob.glob(mlruns_store_dir + "/**/meta.yaml", recursive=True)


def migrate_experiment(meta: ExperimentMetadata) -> None:
    if meta.experiment_id == '0':
        return  # Do not migrate default

    src_dir_uri = meta.artifact_location  # file:///home/rafal/projects/grembedding/mlruns_store/RpTweetsXS/LemmatizerSM/CountVectorizer1000/SVC/SVC2/626233705505186099

    dest_dir_uri = src_dir_uri.replace(MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT)  # file:///home/rafal/projects/grembedding/mlruns_ui/RpTweetsXS/LemmatizerSM/CountVectorizer1000/SVC/SVC2/626233705505186099
    exp_name_dirtree = meta.name.replace("_", "/") + "/"
    dest_dir_uri = dest_dir_uri.replace(exp_name_dirtree,"")  # file:///home/rafal/projects/grembedding/mlruns_ui/626233705505186099
    logger.debug(f"Resolved experiment dest dir uri: {dest_dir_uri}")

    dest_experiment_meta = copy.deepcopy(meta)
    dest_experiment_meta.artifact_location = dest_dir_uri

    shutil.copytree(src_dir_uri.removeprefix("file://"), dest_dir_uri.removeprefix("file://"), dirs_exist_ok=True)


def migrate_run(meta: RunMetadata) -> None:
    pass


def run_sync():
    meta_paths = traverse()
    metas: list[Union[RunMetadata, ExperimentMetadata]] = [load_mlflow_meta(meta_path) for meta_path in meta_paths]
    experiments: list[ExperimentMetadata] = [meta for meta in metas if isinstance(meta, ExperimentMetadata)]

    for exp in experiments:
        migrate_experiment(exp)


if __name__ == "__main__":
    os.chdir("../../")
    run_sync()
