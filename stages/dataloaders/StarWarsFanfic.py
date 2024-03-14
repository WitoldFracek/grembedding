from loguru import logger
from stages.dataloaders.DataLoader import DataLoader
import pandas as pd
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


class StarWarsFanfic(DataLoader):

    DATASET_DIR = 'star_wars_fanfic'
    SPLIT_PERCENT = 0.9

    def create_dataset(self) -> None:
        nowy_wrog_path = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'nowy_wrog.txt')
        wrog_republiki_path = os.path.join(self.raw_datasets_dir, self.DATASET_DIR, 'wrog_republiki.txt')

        nw_text = self.load_and_clean(nowy_wrog_path)
        wr_text = self.load_and_clean(wrog_republiki_path)

        splitter = RecursiveCharacterTextSplitter(
            separators=['—', '\n', '.', ',', '!', '?', '-', ' ', ''],
            chuk_size=1000,
            chunk_overlap=0,
        )

        nw_documents: list[Document] = splitter.create_documents([nw_text])
        wr_documents: list[Document] = splitter.create_documents([wr_text])
        
        index = int(len(nw_documents) * self.SPLIT_PERCENT)
        nw_train_data = {
            'text': [d.page_content for d in nw_documents[:index]],
            'label': [0] * index
        }
        nw_test_data = {
            'text': [d.page_content for d in nw_documents[index:]],
            'label': [0] * (len(nw_documents) - index)
        }
        nw_train_df = pd.DataFrame.from_dict(nw_train_data)
        nw_test_df = pd.DataFrame.from_dict(nw_test_data)

        index = int(len(wr_documents) * self.SPLIT_PERCENT)
        wr_train_data = {
            'text': [d.page_content for d in wr_documents[:index]],
            'label': [1] * index
        }
        wr_test_data = {
            'text': [d.page_content for d in wr_documents[index:]],
            'label': [1] * (len(wr_documents) - index)
        }
        wr_train_df = pd.DataFrame.from_dict(wr_train_data)
        wr_test_df = pd.DataFrame.from_dict(wr_test_data)

        train_df = pd.concat([nw_train_df, wr_train_df], ignore_index=True)
        test_df = pd.concat([nw_test_df, wr_test_df], ignore_index=True)

        train_df = train_df.sample(frac=1.0, random_state=0xC0FFEE)
        test_df = test_df.sample(frac=1.0, random_state=0xC0DE)

        self._save_dataset(train_df, test_df)

    def is_not_empty(self, text: str) -> bool:
        return text.strip() != ''

    def is_not_chapter_sep(self, text: str) -> bool:
        pattern = r'^[\~\*]+'
        return not re.match(pattern, text.strip())
    
    def load_and_clean(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as file:
            return ''.join(filter(self.is_not_chapter_sep, filter(self.is_not_empty, file)))
