import os

import pandas as pd

from stages.dataloaders.DataLoader import DataLoader
from stages.dataloaders.utils import make_split


class RpTweets(DataLoader):
    """Full RP Tweets dataset"""

    DATASET_DIR: str = "rp_tweets"
    USERS_RELEVANT_COLS: list[str] = ["id", "affiliation_id"]
    TWEETS_RELEVANT_COLS: list[str] = ["rawContent", "author_user_id"]

    def create_dataset(self) -> None:
        df = self.load_tweets()
        train, test = make_split(df, stratify=True)
        self._save_dataset(train, test)

    def load_tweets(self) -> pd.DataFrame:
        """
        Load tweets and users parquet files and merge them on the `author_user_id` column. Keep only relevant columns
        """

        tweets_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "tweets.parquet")
        users_fp = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, "users.parquet")

        tweets = pd.read_parquet(tweets_fp,
                                 columns=self.TWEETS_RELEVANT_COLS,
                                 engine="pyarrow")

        users = pd.read_parquet(users_fp,
                                columns=self.USERS_RELEVANT_COLS,
                                engine="pyarrow")

        df = tweets.merge(users, left_on="author_user_id", right_on="id")

        df = df[["rawContent", "affiliation_id"]]
        df.dropna(subset=["rawContent", "affiliation_id"], inplace=True)
        df.rename(columns={"rawContent": "text", "affiliation_id": "label"}, inplace=True)

        return df
