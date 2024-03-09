import os

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stages.dataloaders.DataLoader import DataLoader
from utils.dataloader.text_splitter import split_text
from utils.dataloader.utils import make_split


class PrusVsSienkiewczAV(DataLoader):
    """
    Columns: text, title, author
    All titles Prus:
        ['kamizelka',
         'katarynka',
         'dziwna_historia',
         'powiastki_cmentarne',
         'faraon',
         'lalka',
         'anielka']
    All titles Sienk:
        ['pan_wolodyjowski',
         'quo_vadis',
         'w_pustyni_i_w_puszczy',
         'krzyzacy',
         'ogniem_i_mieczem',
         'potp']
    """

    DATASET_DIR: str = 'prus_vs_sienkiewicz'
    AUTHOR_IDS = {
        "prus": 0,
        "sienkiewicz": 1
    }

    TRAIN_BOOKS = [
        'krzyzacy', 'ogniem_i_mieczem',     # sienkiewicz
        'faraon', 'lalka'                   # prus
    ]
    TEST_BOOKS = [
        'w_pustyni_i_w_puszczy'             # sienkiewicz
        'anielka'                           # prus
    ]

    RANDOM_SEED = 0
    MAX_TEXT_LENGTH = 5_000

    def __init__(self, max_text_len: int = MAX_TEXT_LENGTH):
        super().__init__()
        self.max_text_len = max_text_len
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_text_len,
            chunk_overlap=100
        )

    def create_dataset(self) -> None:
        df = self._load_df()
        df = split_text(df, self.splitter)
        train_df, test_df = make_split(df, stratify="author", test_size=0.25, random_state=self.RANDOM_SEED)

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_parquet(os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'books.parquet'))
        return df
