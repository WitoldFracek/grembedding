import os


def get_root_dir() -> str | os.PathLike:
    """Returns the root directory of the project in DVC environment"""
    if "DVC_ROOT" in os.environ.keys():
        return os.environ["DVC_ROOT"]
    else:
        raise ValueError("DVC_ROOT environment variable not set")