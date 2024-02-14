import os

import pandas as pd
import random
from loguru import logger
from stages.dataloaders.utils import make_split
from stages.dataloaders.DataLoader import DataLoader

SEED = 0xC0FFEE


class PrusVsSienkiewicz(DataLoader):
    SIENKIEWICZ_TEST_BOOKS = ['w_pustyni_i_w_puszczy']
    SIENKIEWICZ_TRAIN_BOOKS = ['krzyzacy', 'ogniem_i_mieczem']
    PRUS_TEST_BOOKSS = ['anielka']
    PRUS_TRAIN_BOOKS = ['faraon', 'lalka']

    def __init__(self, max_text_len: int = 1000):
        super().__init__()
        self.authors_mapping = {'prus': 0, 'sienkiewicz': 1}
        self.max_text_len = max_text_len

    DATASET_DIR: str = 'prus_vs_sienkiewicz'

    def create_dataset(self) -> None:
        df = self.__load_df()
        logger.info(f'Authors mapping: {self.authors_mapping}')
        df['author'] = df['author'].apply(lambda a: self.authors_mapping[a])
        df.rename(columns={'author': 'label'}, inplace=True)

        df = self.reduce_text_len(df)

        logger.info('Title wise train test split')
        train_df, test_df = self.title_wise_split(df)
        self._save_dataset(train_df, test_df)

    def __load_df(self) -> pd.DataFrame:
        prus_titles = self.PRUS_TEST_BOOKSS + self.PRUS_TRAIN_BOOKS
        sienkiewicz_titles = self.SIENKIEWICZ_TRAIN_BOOKS + self.SIENKIEWICZ_TEST_BOOKS
        used_titles = prus_titles + sienkiewicz_titles

        df: pd.DataFrame = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'books.parquet')
        )
        df = df[df['title'].isin(used_titles)]
        return df

    def reduce_text_len(self, df: pd.DataFrame) -> pd.DataFrame:
        ret_df = pd.DataFrame(columns=['text', 'title', 'label'])
        titles = df['title'].unique()
        for title in titles:
            title_df = df[df['title'] == title]
            full_text: str = title_df.iloc[0]['text']
            label: int = title_df.iloc[0]['label']
            texts = []
            line = ''
            for word in full_text.split(' '):
                temp = f'{line} {word}'
                if len(temp) > self.max_text_len:
                    texts.append(line)
                    line = ''
                else:
                    line = f'{line} {word}'
            title_df = pd.DataFrame.from_dict({
                'text': texts,
                'title': [title] * len(texts),
                'label': [label] * len(texts)
            })
            ret_df = pd.concat([ret_df, title_df], ignore_index=True)
        return ret_df

    def title_wise_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = df[df['title'].isin(self.SIENKIEWICZ_TRAIN_BOOKS + self.PRUS_TRAIN_BOOKS)]
        train_df.drop(columns=['title'], inplace=True)
        test_df = df[df['title'].isin(self.SIENKIEWICZ_TEST_BOOKS + self.PRUS_TEST_BOOKSS)]
        test_df.drop(columns=['title'], inplace=True)
        return train_df, test_df

        # train_df = pd.DataFrame(columns=['text', 'label'])
        # test_df = pd.DataFrame(columns=['text', 'label'])
        #
        # for author in self.authors_mapping.values():
        #     titles = list(df[df['author'] == author]['title'].unique())
        #     random.shuffle(titles)
        #     split_index = round(len(titles) / 5 * 3)  # 3/5 to train, 2/5 to test
        #     train_titles = titles[:split_index]
        #     test_titles = titles[split_index:]
        #
        #     train_titles_df = df[df['title'].isin(train_titles)].copy()
        #     train_titles_df.drop(columns=['title'], inplace=True)
        #     train_titles_df.rename(columns={'author': 'label'}, inplace=True)
        #     train_df = pd.concat([train_df, train_titles_df], ignore_index=True)
        #
        #     test_titles_df = df[df['title'].isin(test_titles)].copy()
        #     test_titles_df.drop(columns=['title'])
        #     test_titles_df.rename(columns={'author': 'label'}, inplace=True)
        #     test_df = pd.concat([test_df, test_titles_df], ignore_index=True)
        #
        # return train_df, test_df





