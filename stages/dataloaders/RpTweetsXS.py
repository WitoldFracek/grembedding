from stages.dataloaders.RpTweets import RpTweets


class RpTweetsXS(RpTweets):
    """XSmall version of RpTweets dataset"""

    SUBSET_SIZE_PERCENTAGE: float = 0.1

    def create_dataset(self) -> None:
        df = self.load_tweets()
        train, test = self._make_split(df, stratify=True, subset=self.SUBSET_SIZE_PERCENTAGE)
        self._save_dataset(train, test)
