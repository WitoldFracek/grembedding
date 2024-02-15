import os

import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from stages.dataloaders.DataLoader import DataLoader


class Classics5Authors35Books(DataLoader):
    DATASET_DIR = os.path.join("5authors_35books", "books")
    DEFAULT_MAX_TEXT_LEN = 5_000    # use texts of this size or less

    SKIP_FIRST_N_CHARS = 300        # skip name of author

    TEST_BOOKS = [
        # author 1
        "orzeszkowa_hekuba",
        "orzeszkowa_dziurdziowie",
        # author 2
        "prus_placowka",
        # author 3
        "reymont_obiecana",
        "reymont_wampir",
        # author 4
        "sienkiewicz_pan",
        "sienkiewicz_pustynia",
        "sienkiewicz_dogmatu",
        # author 5
        "zeromski_ludziebezdomni",      # adnotacje podobne jak u orzeszkowej, inne wydanie niż pozostałe
        "zeromski_wiernarzeka"
    ]

    EXCLUDE_BOOKS = [
        "reymont_fermenty",     # cant open and check
        "reymont_chlopi",       # no correct decoding found
    ]

    ENCODING_FORMAT = "cp1250"

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
        df = self._load_books()

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

        self._save_dataset(train_df, test_df)

    def _load_books(self) -> pd.DataFrame:
        records = []

        found_books = os.listdir(os.path.join(self.raw_datasets_dir, self.DATASET_DIR))
        assert len(found_books) == 35, "Dataset has 35 books"
        found_books = [f for f in found_books if f not in self.EXCLUDE_BOOKS]
        assert len(found_books) == 35 - len(self.EXCLUDE_BOOKS), "Some book not excluded"

        for file_name in found_books:
            record = self._load_book_text(file_name)

            if file_name in self.TEST_BOOKS:
                record["fold"] = "test"
            else:
                record["fold"] = "train"

            records.append(record)

        df = pd.DataFrame.from_records(records)

        # Check all books from TEST_BOOKS in test_df
        print(sorted(set(df.query("fold == 'test'")["file_name"])))
        assert all(b in set(df.query("fold == 'test'")["file_name"]) for b in self.TEST_BOOKS), "Some test book not found in df"

        # Categorical to numbers
        df["author"] = df["author"].astype("category")
        df["label"] = df["author"].cat.codes

        return df

    def _load_book_text(self, file_name: str, skip_first_n: int = SKIP_FIRST_N_CHARS) -> dict[str, str]:
        with open(os.path.join(self.raw_datasets_dir, self.DATASET_DIR, file_name), "rb") as f:
            text_binary = f.read()

        text = text_binary.decode(self.ENCODING_FORMAT)  # I guessed it
        text = text[skip_first_n:]

        name_parsed = file_name.split("_")
        author_name = name_parsed[0]
        book_name = "_".join(name_parsed[1:])

        return dict(
            text=text,
            author=author_name,
            book=book_name,
            file_name=file_name
        )

    # TODO to utils?
    def _split_text(self, df: pd.DataFrame) -> pd.DataFrame:
        chunked_text = df["text"].apply(lambda txt: self.splitter.split_text(txt))

        chunk_lengths = [len(chunk) for chunks in chunked_text for chunk in chunks]
        mean_chunk_len = np.mean(chunk_lengths)
        stddev_chunk_len = np.std(chunk_lengths)
        smallest_chunk = min(chunk_lengths)

        # TODO this splitter has some problems - small chunks, invalid sentence breaks, etc. will fix
        logger.info(f"Using {self.splitter.__class__} (chunk_size={self.max_text_len}) - avg chunk len: {mean_chunk_len},"
                    f" std: {stddev_chunk_len}, smallest chunk: {smallest_chunk}")

        result = df.copy()
        result["chunked_text"] = chunked_text
        return result
