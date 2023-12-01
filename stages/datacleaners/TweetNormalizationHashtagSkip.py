from loguru import logger

from stages.datacleaners.DataCleaner import DataCleaner
from stages.datacleaners.components.EmojiCleanerStep import EmojiCleanerStep
from stages.datacleaners.components.HashtagCleaner import HashtagCleaner
from stages.datacleaners.components.HtmlTagCleanerStep import HtmlTagCleanerStep
from stages.datacleaners.components.HttpLinkCleanerStep import HttpLinkCleanerStep
from stages.datacleaners.components.TwitterHandleCleanerStep import TwitterHandleCleanerStep
from stages.datacleaners.components.base import DataCleanerPipeline


class TweetNormalizationHashtagSkip(DataCleaner):

    def __init__(self):
        self.pipeline: DataCleanerPipeline = DataCleanerPipeline(
            steps=[
                HttpLinkCleanerStep(),
                EmojiCleanerStep(),
                HtmlTagCleanerStep(),
                TwitterHandleCleanerStep(placeholder="@user"),
                HashtagCleaner(placeholder="")
            ]
        )

    def clean_data(self, dataset: str) -> None:
        train, test = self.load_dataset(dataset)

        logger.info(f"Executing steps: {[s.__class__.__name__ for s in self.pipeline.steps]}")

        self.pipeline(train, inplace=True)
        self.pipeline(test, inplace=True)

        train["text"] = train["text"].apply(lambda txt: txt.replace("\n", " "))
        test["text"] = test["text"].apply(lambda txt: txt.replace("\n", " "))

        train.rename(columns={"text": "clean_text"}, inplace=True)
        test.rename(columns={"text": "clean_text"}, inplace=True)

        self.save_dataframe_as_parquet(dataset, train, test)
