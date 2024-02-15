import os
from typing import Literal
from loguru import logger

import numpy as np
import pandas as pd

from stages.dataloaders.DataLoader import DataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WritingStyle(DataLoader):
    DATASET_DIR = "styl_literacki"
    DATASET_FILE = "styl_literacki.parquet"

    DEFAULT_MAX_TEXT_LEN = 2_500    # use texts of this size or less

    WRITING_STYLE_LABEL_MAPPING: dict[Literal['literacki', 'naukowy'], int] = {
        "literacki": 0,
        "naukowy": 1
    }

    def __init__(self, max_text_len: int = DEFAULT_MAX_TEXT_LEN):
        super().__init__()
        self.max_text_len = max_text_len
        self.splitter = RecursiveCharacterTextSplitter(
            **dict(
                chunk_size=self.max_text_len,
                chunk_overlap=100
            )
        )

    def create_dataset(self) -> None:
        df = self._load_df()

        df = self._split_text(df)
        exploded_df = df.explode(column="chunked_text")

        train_df = exploded_df.query("fold == 'train'")
        test_df = exploded_df.query("fold == 'test'")

        # Take relevant cols and rename
        train_df = train_df[["chunked_text", "label"]]
        test_df = test_df[["chunked_text", "label"]]
        train_df.rename(columns={
            "chunked_text": "text"
        }, inplace=True)
        test_df.rename(columns={
            "chunked_text": "text"
        }, inplace=True)

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

    def _split_text(self, df: pd.DataFrame) -> pd.DataFrame:
        chunked_text = df["text"].apply(lambda txt: self.splitter.split_text(txt))

        chunk_lengths = [len(chunk) for chunks in chunked_text for chunk in chunks]
        mean_chunk_len = np.mean(chunk_lengths)
        stddev_chunk_len = np.std(chunk_lengths)
        smallest_chunk = min(chunk_lengths)

        logger.info(f"Using SpacyTextSplitter (chunk_size={self.max_text_len}) - avg chunk len: {mean_chunk_len},"
                    f" std: {stddev_chunk_len}, smallest chunk: {smallest_chunk}")

        result = df.copy()
        result["chunked_text"] = chunked_text
        return result


if __name__ == "__main__":
    os.environ["DVC_ROOT"] = "."
    dl = LiteraryStyle()
    dl.create_dataset()
