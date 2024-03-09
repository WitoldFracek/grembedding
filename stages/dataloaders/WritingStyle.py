import os
from typing import Literal

import pandas as pd

from stages.dataloaders.DataLoader import DataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.dataloader.text_splitter import split_text


class WritingStyle(DataLoader):
    DATASET_DIR = "styl_literacki"
    DATASET_FILE = "styl_literacki.parquet"

    DEFAULT_MAX_TEXT_LEN = 5_000    # use texts of this size or less

    WRITING_STYLE_LABEL_MAPPING: dict[Literal['literacki', 'naukowy'], int] = {
        "literacki": 0,
        "naukowy": 1
    }

    def __init__(self, max_text_len: int = DEFAULT_MAX_TEXT_LEN):
        super().__init__()
        self.max_text_len = max_text_len
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_text_len,
            chunk_overlap=100
        )

    def create_dataset(self) -> None:
        df = self._load_df()
        df = split_text(df, self.splitter)

        train_df = df.query("fold == 'train'")
        test_df = df.query("fold == 'test'")

        # Take relevant cols and rename
        train_df = train_df[["text", "label"]]
        test_df = test_df[["text", "label"]]

        assert "text" in train_df.columns
        assert "text" in test_df.columns
        assert "label" in train_df.columns
        assert "label" in test_df.columns

        self._save_dataset(train_df, test_df)

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_parquet(
            os.path.join(self.raw_datasets_dir, self.DATASET_DIR, self.DATASET_FILE)
        )
        df["label"] = df["style"].apply(lambda s: self.WRITING_STYLE_LABEL_MAPPING[s])
        return df


if __name__ == "__main__":
    os.environ["DVC_ROOT"] = "."
    dl = WritingStyle()
    dl.create_dataset()
