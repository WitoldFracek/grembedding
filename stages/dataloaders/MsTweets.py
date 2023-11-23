import os

import pandas as pd
from loguru import logger

from stages.dataloaders.DataLoader import DataLoader
from stages.dataloaders.utils import make_split

from typing import Tuple


class MsTweets(DataLoader):
    """Full MS Tweets dataset"""

    DATASET_DIR: str = "rp_tweets"
    ALLOWED_TWEET_LANGUAGE: str = "pl"
    USERS_RELEVANT_COLS: list[str] = ["id", "affiliation_id"]
    TWEETS_RELEVANT_COLS: list[str] = ["rawContent", "author_user_id"]

    def create_dataset(self) -> None:
        df = self.load_and_filter_data()
        train, test = make_split(df, stratify=True, random_state=0)
        self._save_dataset(train, test)

    def load_tweets(self) -> pd.DataFrame:
        """
        Load tweets and users parquet files and merge them on the `author_user_id` column. Keep only relevant columns
        """

        tweets_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "tweets.parquet")
        users_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "users.parquet")

        tweets = pd.read_parquet(tweets_fp,
                                 columns=self.TWEETS_RELEVANT_COLS,
                                 filters=[("lang", "==", self.ALLOWED_TWEET_LANGUAGE)],
                                 engine="pyarrow")

        users = pd.read_parquet(users_fp,
                                columns=self.USERS_RELEVANT_COLS,
                                engine="pyarrow")

        df = tweets.merge(users, left_on="author_user_id", right_on="id", how="left")

        df = df[["rawContent", "affiliation_id"]]
        df.rename(columns={"rawContent": "text", "affiliation_id": "label"}, inplace=True)

        df.dropna(subset=["text", "label"], inplace=True)

        logger.debug(f"Loaded tweets data: Tweets={tweets.shape}, Users={users.shape}")
        logger.info(f"Tweets merged dataset shape = {df.shape}")

        return df
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tweets_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "tweets.parquet")
        users_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "users.parquet")
        df_tweets = pd.read_parquet(tweets_fp, filters=[("lang", "==", self.ALLOWED_TWEET_LANGUAGE)],)
        df_users = pd.read_parquet(users_fp)
        return df_tweets, df_users
    
    def load_and_filter_data(self) -> pd.DataFrame:
        df_tweets, df_users = self.load_raw_data()
        # Pozbycie się duplikatów
        df_tweets = df_tweets.drop_duplicates(subset=['id'])
        # Usunięcie odpowiedzi do tweetów i ludzi
        df_tweets = df_tweets[pd.isna(df_tweets['inReplyToTweetId'])]
        df_tweets = df_tweets[df_tweets['inReplyToUser'].isnull()]
        df_tweets = df_tweets[df_tweets['quotedTweet'].isnull()]
        # Wybranie tylko potrzebnych kolumn
        df_tweets = df_tweets[['author_user_id', 'rawContent']]
        # Połączenie z userami
        df_data = pd.merge(df_tweets, df_users, left_on='author_user_id', right_on='id', how='left')
        df_data = df_data[['rawContent', 'affiliation_displayname']]
        # Wybranie partii
        affiliation = ['Prawo i Sprawiedliwość', 'Konfederacja', 'PlatformaObywatelska', 'Lewica']
        df_data = df_data[df_data['affiliation_displayname'].isin(affiliation)]
        print(df_data['affiliation_displayname'].value_counts())
        # Zmiana nazw kolumn
        df_data = df_data.rename(columns = {"rawContent": "text", "affiliation_displayname": "label"})
        # Mappowanie labela
        df_data['label'] = pd.factorize(df_data['label'])[0]
        logger.info(f"Tweets merged dataset shape = {df_data.shape}")
        
        return df_data

    