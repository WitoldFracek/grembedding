import os

import pandas as pd

from loguru import logger
from stages.dataloaders.utils import make_split
from stages.dataloaders.DataLoader import DataLoader


class TweeterCyberbullying(DataLoader):
    
    DATASET_DIR: str = 'tweet_cyberbullying'

    def create_dataset(self) -> None:
        df: pd.DataFrame = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'data.parquet')
        )
        logger.info(f'Train test split')
        train_df, test_df = make_split(df, stratify=True, test_size=0.1)
        self._save_dataset(train_df, test_df)
