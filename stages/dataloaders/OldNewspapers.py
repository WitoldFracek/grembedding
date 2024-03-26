from loguru import logger
from stages.dataloaders.DataLoader import DataLoader
import pandas as pd
import os


SEED = 0xC0FFEE


class OldNewspapers(DataLoader):

    DATASET_DIR = 'old_newspapers'

    def create_dataset(self) -> None:
        train_df: pd.DataFrame = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'train.parquet')
        )
        train_df.dropna(inplace=True)
        test_df: pd.DataFrame = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'test.parquet')
        )
        test_df.dropna(inplace=True)

        train_df = train_df.drop(columns=['source'])
        test_df = test_df.drop(columns=['source'])

        # TODO sample this stratified on label
        train_df = train_df.sample(frac=0.1, random_state=SEED)
        test_df = test_df.sample(frac=0.1, random_state=SEED)

        self._save_dataset(train_df, test_df)