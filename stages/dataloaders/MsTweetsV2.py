import os

import pandas as pd
from loguru import logger

from stages.dataloaders.DataLoader import DataLoader
from stages.dataloaders.utils import make_split, temporal_train_test_split

from typing import Tuple


class MsTweetsV2(DataLoader):

    DATASET_DIR: str = "rp_tweets"
    ALLOWED_TWEET_LANGUAGE: str = "pl"

    def create_dataset(self) -> None:
        df = self.load_and_filter_data()
        train, test = temporal_train_test_split(df, timestamp_col="date", test_size=0.2)
        self._save_dataset(train, test)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tweets_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "tweets.parquet")
        users_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "users.parquet")
        df_tweets = pd.read_parquet(tweets_fp, filters=[("lang", "==", self.ALLOWED_TWEET_LANGUAGE)])
        df_users = pd.read_parquet(users_fp)
        return df_tweets, df_users

    def load_and_filter_data(self) -> pd.DataFrame:
        tweets, users = self.load_raw_data()

        # Pozbycie się duplikatów
        tweets = tweets.drop_duplicates(subset=['id'])

        # Usunięcie odpowiedzi do tweetów i ludzi
        tweets = tweets[pd.isna(tweets['inReplyToTweetId'])]
        tweets = tweets[tweets['inReplyToUser'].isnull()]
        tweets = tweets[tweets['quotedTweet'].isnull()]

        # Wybranie tylko potrzebnych kolumn
        tweets = tweets[['author_user_id', 'rawContent', 'date']]

        # Połączenie z userami
        df = pd.merge(tweets, users, left_on='author_user_id', right_on='id', how='right')
        df = df[['rawContent', 'affiliation_id', 'date']]

        # Wybranie partii
        affiliationids = ["pis", "ko", "konfederacja", "lewica"]
        # make sure no spelling mistakes
        assert all([aid in df["affiliation_id"].unique() for aid in affiliationids])

        df = df.query("affiliation_id in @affiliationids")
        print(df['affiliation_id'].value_counts())

        # Zmiana nazw kolumn
        df = df.rename(columns={"rawContent": "text", "affiliation_id": "label_str"})

        # Ogarnięcie dlugości tekstu
        df = df.query('text.str.len() <= 500 and text.str.len() >= 50')
        df = df[['text', 'label_str', 'date']]

        # Mappowanie labela
        df['label'] = pd.factorize(df['label_str'])[0]

        logger.info(f"Tweets merged dataset shape = {df.shape}")

        return df

