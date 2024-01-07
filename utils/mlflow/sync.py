import os
import re
import shutil
from pathlib import Path
from typing import Union

import mlflow
from loguru import logger
from tqdm.auto import tqdm

from config.mlflow import MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT
from utils.mlflow.domain import load_mlflow_meta, ExperimentMetadata, RunMetadata


def run_sync(force_recreate: bool = True):
    """Copies all experiments from MLRUNS_STORAGE_ROOT to MLRUNS_VIEW_ROOT

    1. Finds all experiments in MLRUNS_STORAGE_ROOT
    2. Creates corresponding experiments in MLRUNS_VIEW_ROOT
    3. Copies all runs from MLRUNS_STORAGE_ROOT to MLRUNS_VIEW_ROOT
    4. Fixes up metadata of the copied runs to point to the correct experiment id and artifact uri
    """
    source_exp_dirs = _find_mlruns_store_experiment_dirs()
    dest_mlruns_root = Path.cwd().joinpath(MLRUNS_VIEW_ROOT)
    logger.info(f"Schedule to copy {len(source_exp_dirs)} experiments to {dest_mlruns_root}")

    # remove previous migrations
    if force_recreate:
        shutil.rmtree(dest_mlruns_root, ignore_errors=True)
        logger.info(f"Removed previous migrations from {dest_mlruns_root}")

    for exp_folder in tqdm(source_exp_dirs):
        dest_experiment_path = _resolve_destination(exp_folder, dest_mlruns_root)
        logger.info(f"For exp: {exp_folder} resolved dest experiment path: {dest_experiment_path}")

        # Walk all direct directories children in src exp_folder & copy
        for root, dirs, files in os.walk(exp_folder):
            for dir in dirs:
                source_run_dir = os.path.join(root, dir)  # mlruns store run folder
                dest_experiment_dir = os.path.join(dest_experiment_path, dir)  # mlruns view experiment folder
                logger.info(f"Copying {source_run_dir} to {dest_experiment_dir}")
                shutil.copytree(source_run_dir, dest_experiment_dir, dirs_exist_ok=True)

                # Fixup metadata, omit datasets folders
                if dir != "datasets":
                    _fixup_dest_run_metadata(dest_experiment_dir, new_experiment_id=Path(dest_experiment_path).name)
            break  # Break after the first iteration to not go deeper


def _fixup_dest_run_metadata(dest_run_dir: Union[str, os.PathLike], new_experiment_id: str) -> None:
    """Modifies runs metadata to point to the correct experiment id and artifact uri after copy to MLRUNS_VIEW_ROOT

    Args:
        dest_run_dir: path to the copied run directory in MLRUNS_VIEW_ROOT
        new_experiment_id: id of the dest experiment in MLRUNS_VIEW_ROOT
    """
    meta_path = Path(dest_run_dir).joinpath("meta.yaml")
    meta: RunMetadata = load_mlflow_meta(meta_path, errors="raise")

    meta.experiment_id = str(new_experiment_id)
    meta.artifact_uri = Path(dest_run_dir).joinpath("artifacts").as_uri()

    meta.save(dest_run_dir)
    logger.info(f"Fixed up metadata: {meta_path}")


def _resolve_destination(exp_folder: Union[str, os.PathLike],
                         base_dest_path: Union[str, os.PathLike] = MLRUNS_VIEW_ROOT) -> Union[str, os.PathLike]:
    """Resolved destination path of the experiment for a given run of mlruns_store
    (creates dest experiment if not exists)

    Args:
        exp_folder: path to the DVC tracked experiment folder
        base_dest_path: path to the mlruns_view root

    Returns:
        Path to the destination experiment folder in MLRUNS_VIEW_ROOT
    """
    meta: ExperimentMetadata = load_mlflow_meta(Path(exp_folder).joinpath("meta.yaml"), errors="raise")
    dataset_name = meta.name.split("_")[0]
    dest_exp_id = _resolve_dest_experiment(dataset_name, base_dest_path)
    dest_exp_path = Path(base_dest_path).joinpath(dest_exp_id)
    return dest_exp_path


def _resolve_dest_experiment(dataset_name: str, base_dest_path: Union[str, os.PathLike] = MLRUNS_VIEW_ROOT) -> str:
    """Resolve destination experiment id by name, create if not exists (convention: name = dataset)"""
    mlflow.set_tracking_uri(base_dest_path)
    exp = mlflow.get_experiment_by_name(dataset_name)
    if exp is None:
        return mlflow.create_experiment(dataset_name)
    else:
        return exp.experiment_id


def _find_mlruns_store_experiment_dirs(base_path: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT) -> list[
    Union[str, os.PathLike]]:
    folder_name_pattern = re.compile(r'^\d+$')

    matched_folders = []

    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if folder_name_pattern.match(dir) and dir != '0':
                matched_folders.append(os.path.abspath(os.path.join(root, dir)))

    return matched_folders


if __name__ == "__main__":
    os.chdir('../../')
    run_sync(force_recreate=False)
