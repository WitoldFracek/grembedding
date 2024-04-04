from loguru import logger
from sklearn.model_selection import train_test_split

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

        _, train_df = train_test_split(train_df, test_size=0.1, random_state=SEED, stratify=train_df['label'])
        _, test_df = train_test_split(test_df, test_size=0.1, random_state=SEED, stratify=test_df['label'])
        logger.debug(f"OldNewspapers has {len(train_df)} train samples and {len(test_df)} test samples. "
                     f"Avg train text length: {train_df['text'].str.len().mean():.2f}, "
                     f"Std train text length: {train_df['text'].str.len().std():.2f}")

        self._save_dataset(train_df, test_df)
