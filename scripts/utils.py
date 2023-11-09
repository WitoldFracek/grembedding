import pathlib
import os

OBJECTS_DIR = 'objects'
DATACLEANERS_DIR = 'datacleaners'

def get_datacleaners_path() -> str:
    path = _get_main_dir_path()
    path = os.path.join(path, OBJECTS_DIR, DATACLEANERS_DIR)
    return path

def _get_main_dir_path() -> str:
    return pathlib.Path(__file__).parent.parent