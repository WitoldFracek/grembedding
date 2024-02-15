import os
from typing import Union

import pandas as pd

from stages.dataloaders.DataLoader import DataLoader


class PAN18PL(DataLoader):


    DATASET_DIR = "pan18pl"

    PL_TRAIN_PROBLEMS = [
        "problem00007",
        "problem00008"
    ]

    PL_TEST_PROBLEMS = [
        "problem00013",
        "problem00014",
        "problem00015",
        "problem00016"
    ]

    def __init__(self):
        super(PAN18PL, self).__init__()


    def create_dataset(self) -> None:
        pass

    def _load_problem_texts(self, ) -> pd.DataFrame:
        pass
