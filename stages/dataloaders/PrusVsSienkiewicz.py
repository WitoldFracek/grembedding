import os

import pandas as pd
import random
from loguru import logger
from stages.dataloaders.utils import make_split
from stages.dataloaders.DataLoader import DataLoader


class PrusVsSienkiewicz(DataLoader):

    def __init__(self):
        super().__init__()
        self.authors_mapping = {}

    DATASET_DIR: str = 'prus_vs_sienkiewicz'

    def create_dataset(self) -> None:
        df: pd.DataFrame = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'books.parquet')
        )
        authors = df['author'].unique()
        self.authors_mapping = {
            author: i
            for i, author
            in enumerate(sorted(list(authors)))
        }
        df['author'] = df['author'].apply(lambda a: self.authors_mapping[a])
        train_df, test_df = self.title_wise_split(df)
        self._save_dataset(train_df, test_df)

    def title_wise_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.DataFrame(columns=['text', 'label'])
        test_df = pd.DataFrame(columns=['text', 'label'])

        for author in self.authors_mapping.values():
            titles = list(df[df['author'] == author]['title'].unique())
            random.shuffle(titles)
            split_index = round(len(titles) / 5 * 3)  # 3/5 to train, 2/5 to test
            train_titles = titles[:split_index]
            test_titles = titles[split_index:]

            train_titles_df = df[df['title'].isin(train_titles)].copy()
            train_titles_df.drop(columns=['title'], inplace=True)
            train_titles_df.rename(columns={'author': 'label'}, inplace=True)
            train_df = pd.concat([train_df, train_titles_df], ignore_index=True)

            test_titles_df = df[df['title'].isin(test_titles)].copy()
            test_titles_df.drop(columns=['title'])
            test_titles_df.rename(columns={'author': 'label'}, inplace=True)
            test_df = pd.concat([test_df, test_titles_df], ignore_index=True)

        return train_df, test_df





