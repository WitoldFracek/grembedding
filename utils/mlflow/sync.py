import glob
import os
import re
import shutil
from pathlib import Path
from typing import Union, Optional

import mlflow
from loguru import logger
from tqdm.auto import tqdm
from typing_extensions import deprecated
import zipfile

from config.mlflow import MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT
from utils.mlflow.domain import load_mlflow_meta, ExperimentMetadata, RunMetadata


def run_sync(force_recreate: bool = True):
    """Copies all experiments from MLRUNS_STORAGE_ROOT to MLRUNS_VIEW_ROOT

    1. Finds all experiments in MLRUNS_STORAGE_ROOT
    2. Creates corresponding experiments in MLRUNS_VIEW_ROOT
    3. Copies all runs from MLRUNS_STORAGE_ROOT to MLRUNS_VIEW_ROOT
    4. Fixes up metadata of the copied runs to point to the correct experiment id and artifact uri
    """
    source_exp_output_zips = _find_mlruns_zip_files()
    dest_mlruns_root = Path.cwd().joinpath(MLRUNS_VIEW_ROOT)
    logger.info(f"Schedule to copy {len(source_exp_output_zips)} experiments to {dest_mlruns_root}")

    # remove previous migrations
    if force_recreate:
        shutil.rmtree(dest_mlruns_root, ignore_errors=True)
        logger.info(f"Removed previous migrations from {dest_mlruns_root}")

    for out_zipped_path in tqdm(source_exp_output_zips):
    

        # Unzip the mlflow output from
        temp_dir = Path(out_zipped_path).parent.joinpath("__unzip_temp__")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            os.mkdir(temp_dir)
            # logger.debug(f"Created temp unzip dir: {temp_dir}. Unpacking from {out_zipped_path} to temp dir")
            with zipfile.ZipFile(out_zipped_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Get experiment path from unzip temp (the experiment is a directory consisting of digits only, not 0 dir)
            exp_source_path = ""
            for item in os.listdir(temp_dir):
                if item.isnumeric() and item != "0":
                    exp_source_path = os.path.join(temp_dir, item)
                    # logger.debug(f"Found exp source location in unzipped temp: {exp_source_path}")
                    break

            dest_experiment_path = _resolve_destination(exp_source_path, dest_mlruns_root)
            if dest_experiment_path is None:
                continue
            # logger.debug(f"For exp: {exp_source_path} resolved dest experiment path: {dest_experiment_path}")

            # Walk all direct directories children in src exp_folder & copy
            for root, dirs, files in os.walk(exp_source_path):
                for dir in dirs:
                    source_run_dir = os.path.join(root, dir)  # mlruns store run folder
                    dest_experiment_dir = os.path.join(dest_experiment_path, dir)  # mlruns view experiment folder
                    shutil.copytree(source_run_dir, dest_experiment_dir, dirs_exist_ok=True)

                    # Fixup metadata, omit datasets folders
                    if dir != "datasets":
                        _fixup_dest_run_metadata(dest_experiment_dir, new_experiment_id=Path(dest_experiment_path).name)
                break  # Break after the first iteration to not go deeper

        finally:
            # logger.debug(f"Removing temp dir {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


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
    # logger.info(f"Fixed up metadata: {meta_path}")


def _resolve_destination(exp_folder: Union[str, os.PathLike],
                         base_dest_path: Union[str, os.PathLike] = MLRUNS_VIEW_ROOT) -> Optional[Union[str, os.PathLike]]:
    """Resolved destination path of the experiment for a given run of mlruns_store
    (creates dest experiment if not exists)

    Args:
        exp_folder: path to the DVC tracked experiment folder like /mlruns_store/Dataloader/.../RandomForest1/0232323
        base_dest_path: path to the mlruns_view root

    Returns:
        Path to the destination experiment folder in MLRUNS_VIEW_ROOT
    """
    meta: ExperimentMetadata = load_mlflow_meta(Path(exp_folder).joinpath("meta.yaml"), errors="raise")
    if meta is None:
        return None
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


@deprecated("non zip approach")
def _find_mlruns_store_experiment_dirs(base_path: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT) -> list[
    Union[str, os.PathLike]]:
    folder_name_pattern = re.compile(r'^\d+$')      # directory name consisting only of digits

    matched_folders = []

    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if folder_name_pattern.match(dir) and dir != '0':
                matched_folders.append(os.path.abspath(os.path.join(root, dir)))

    return matched_folders


def _find_mlruns_zip_files(base_path: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT):
    """This finds all out_mlflow.zip files in mlruns_store"""
    search_pattern = os.path.join(base_path, '**', 'out_mlflow.zip')
    matched_files = glob.glob(search_pattern, recursive=True)
    return matched_files


if __name__ == "__main__":
    run_sync(force_recreate=True)
