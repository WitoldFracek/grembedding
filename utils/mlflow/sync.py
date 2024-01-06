import copy
import glob
import os
import re
import shutil
from pathlib import Path
from typing import Union, Optional

from loguru import logger
from tqdm.auto import tqdm

from config.mlflow import MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT
from utils.mlflow.domain import load_mlflow_meta, RunMetadata, ExperimentMetadata


# def traverse(mlruns_store_dir: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT) -> list[Union[str, os.PathLike]]:
#     return glob.glob(mlruns_store_dir + "/**/meta.yaml", recursive=True)
#
#
# def migrate_experiment(meta: ExperimentMetadata) -> None:
#     if meta.experiment_id == '0':
#         return  # Do not migrate default
#
#     src_dir_uri = meta.artifact_location  # file:///home/rafal/projects/grembedding/mlruns_store/RpTweetsXS/LemmatizerSM/CountVectorizer1000/SVC/SVC2/626233705505186099
#
#     # dest_dir_uri = src_dir_uri.replace(MLRUNS_STORAGE_ROOT, MLRUNS_VIEW_ROOT)  # file:///home/rafal/projects/grembedding/mlruns_ui/RpTweetsXS/LemmatizerSM/CountVectorizer1000/SVC/SVC2/626233705505186099
#     # exp_name_dirtree = meta.name.replace("_", "/") + "/"
#     # dest_dir_uri = dest_dir_uri.replace(exp_name_dirtree,"")  # file:///home/rafal/projects/grembedding/mlruns_ui/626233705505186099
#     # dest_dir_uri = dest_dir_uri.removesuffix(meta.experiment_id)
#
#     # DEST DIR URI
#     # DVC_ROOT + MLRUNS_VIEW_ROOT
#     dest_dir_path = Path.cwd().joinpath(MLRUNS_VIEW_ROOT)
#     # dest_dir_uri = os.path.join(os.environ["DVC_ROOT"], MLRUNS_VIEW_ROOT)
#     logger.debug(f"Resolved experiment dest dir uri: {dest_dir_path}")
#
#     # dest_experiment_meta = copy.deepcopy(meta)
#     # dest_experiment_meta.artifact_location = dest_dir_uri
#
#     shutil.copytree(src_dir_uri.removeprefix("file://"), dest_dir_path, dirs_exist_ok=True)
#     logger.info(f"Success migrating {meta.name}")
#
#
# def migrate_run(meta: RunMetadata) -> None:
#     pass
#
#
# def run_sync():
#     meta_paths = traverse()
#     metas: list[Optional[Union[RunMetadata, ExperimentMetadata]]] = [load_mlflow_meta(meta_path) for meta_path in
#                                                                      meta_paths]
#     metas: list[Union[RunMetadata, ExperimentMetadata]] = [m for m in metas if m is not None]
#     experiments: list[ExperimentMetadata] = [meta for meta in metas if isinstance(meta, ExperimentMetadata)]
#
#     for exp in tqdm(experiments):
#         migrate_experiment(exp)


def run_simple_sync():
    all_exp_folders = find_folders()
    base_dest_path = Path.cwd().joinpath(MLRUNS_VIEW_ROOT)
    logger.info(f"Schedule to copy {len(all_exp_folders)} experiments to {base_dest_path}")

    for exp_folder in tqdm(all_exp_folders):
        current_dest_path = Path(base_dest_path).joinpath(Path(exp_folder).name)
        shutil.copytree(exp_folder, current_dest_path, dirs_exist_ok=True)


def find_folders(base_path: Union[str, os.PathLike] = MLRUNS_STORAGE_ROOT) -> list[Union[str, os.PathLike]]:
    folder_name_pattern = re.compile(r'^\d+$')

    matched_folders = []

    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if folder_name_pattern.match(dir) and dir != '0':
                matched_folders.append(os.path.abspath(os.path.join(root, dir)))

    return matched_folders


if __name__ == "__main__":
    os.chdir('../../')
    run_simple_sync()
