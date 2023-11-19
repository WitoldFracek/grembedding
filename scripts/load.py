import importlib
import os
import sys

from loguru import logger

if "DVC_ROOT" in os.environ.keys():
    root_dir = os.environ["DVC_ROOT"]
    if root_dir not in sys.path:
        sys.path.append(root_dir)
        logger.debug(f"Appending root dir: '{root_dir}' to sys.path")

from stages.dataloaders.DataLoader import DataLoader


def main():
    dataloader: str = sys.argv[1]
    logger.info(f"Executing load with dataloader: '{dataloader}'")

    dataloader_cls = getattr(importlib.import_module(f"stages.dataloaders.{dataloader}"), dataloader)
    loader: DataLoader = dataloader_cls()
    logger.info(f"Instantiated datacleaner: '{loader.__class__.__name__}'")

    loader.create_dataset()


if __name__ == "__main__":
    main()
