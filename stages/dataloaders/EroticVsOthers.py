from loguru import logger
from stages.dataloaders.DataLoader import DataLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split
import pandas as pd

class EroticVsOthers(DataLoader):

    DATASET_DIR = "LOL24Dataset"
    TEST_SIZE = 0.2

    def create_dataset(self) -> None:
        X_erotic, X_others = self.load_texts()
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '.', ',', '!', '?', '-', ' ', ''],
            chunk_size=1000,
            chunk_overlap=0,
            keep_separator=False
        )
       
        X_erotic_train, X_erotic_test, X_others_train, X_others_test = self.split(X_erotic, X_others)
       
        X_erotic_train = splitter.create_documents(X_erotic_train)
        X_erotic_test = splitter.create_documents(X_erotic_test)
        X_others_train = splitter.create_documents(X_others_train)
        X_others_test = splitter.create_documents(X_others_test)
       
        dict_train_erotic = {
            "text": [d.page_content for d in X_erotic_train],
            "label": [1] * len(X_erotic_train)
        }

        dict_train_others = {
            "text": [d.page_content for d in X_others_train],
            "label": [0] * len(X_others_train)
        }

        dict_test_erotic = {
            "text": [d.page_content for d in X_erotic_test],
            "label": [1] * len(X_erotic_test)
        }

        dict_test_others = {
            "text": [d.page_content for d in X_others_test],
            "label": [0] * len(X_others_test)
        }
       
        df_train_erotic = pd.DataFrame.from_dict(dict_train_erotic)
        df_train_others = pd.DataFrame.from_dict(dict_train_others)
        df_test_erotic = pd.DataFrame.from_dict(dict_test_erotic)
        df_test_others = pd.DataFrame.from_dict(dict_test_others)
       
        df_train = pd.concat([df_train_erotic, df_train_others], ignore_index=True)
        df_test = pd.concat([df_test_erotic, df_test_others], ignore_index=True)
       
        self._save_dataset(df_train, df_test)

    def load_texts(self):
        X_erotic, X_others = {}, {}
        path = os.path.join("datasets_raw", self.DATASET_DIR)
        for category in os.listdir(path):
            for author in os.listdir(os.path.join(path, category)):
                author_path = os.path.join(path, category, author)
                for story_id in os.listdir(author_path):
                    filepath = os.path.join(author_path, story_id)
                    with open(filepath, "r", encoding="utf-8") as file:
                        if category == "erotyczne":
                            if not author in X_erotic:
                                 X_erotic[author] = []
                            X_erotic[author].append(file.read())
                        else:
                            if not author in X_others:
                                 X_others[author] = []
                            X_others[author].append(file.read())
        return X_erotic, X_others

    def split(self, X_erotic, X_others):
        common_authors = []
        for author in X_erotic.keys():
            if author in X_others:
                common_authors.append(author)
        X_erotic_test = []
        X_erotic_train = []
        X_others_test = []
        X_others_train = []
        for author in X_erotic.keys():
            if author in common_authors:
                X_erotic_test += X_erotic[author]
            else:
                X_erotic_train += X_erotic[author]
        for author in X_others.keys():
            if author in common_authors:
                X_others_test += X_others[author]
            else:
                X_others_train += X_others[author]
        return X_erotic_train, X_erotic_test, X_others_train, X_others_test    