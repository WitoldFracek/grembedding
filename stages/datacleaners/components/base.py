from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from tqdm import tqdm


class DataProcessingStep(ABC):

    def __call__(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        result = df if inplace else df.copy()
        self.do_transform_inplace(result)
        return result

    @abstractmethod
    def do_transform_inplace(self, df: pd.DataFrame) -> None:
        """Perform transformation inplace. Expects `text` column to be present"""
        raise NotImplementedError


class DataCleanerPipeline(ABC):

    def __init__(self, steps: list[DataProcessingStep]):
        self.steps = steps

    def __call__(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        data = df if inplace else df.copy()
        for step in tqdm(self.steps, desc="Executing pipeline"):
            data = step(data, inplace)
        return data

    def __str__(self):
        return f"DataCleanerPipeline(steps={self.steps})"
