import pandas as pd
from typing_extensions import deprecated

from stages.vectorizers.Vectorizer import Vectorizer
import numpy as np
import spacy
from sklearn import preprocessing
from loguru import logger
from tqdm import tqdm
import json
import os

from utils.spacy_gpu import autoconfigure_spacy_mode, resolve_spacy_batch_size


class FullMorphTagVectorizer(Vectorizer):

    PROCESSING_BATCH_SIZE: int = resolve_spacy_batch_size()

    def __init__(self) -> None:
        super().__init__()
        autoconfigure_spacy_mode(self.__class__)

        dir_ = Vectorizer.get_vectoriser_data_dir()
        path = os.path.join(dir_, 'full_morphological_tags.json')
        if not os.path.exists(path):
            raise Exception(f'path "{path}" does not exist. Could not load morphological tags.')

        with open(path, 'r', encoding='utf-8') as file:
            self.full_tags: dict[str, int] = json.load(file)

        self.nlp = spacy.load("pl_core_news_lg")

    def vectorize(self, dataset: str, datacleaner: str):
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        logger.info(f'Generating train document grammatical vectors')
        X_train, train_errors_new = self.batch_get_document_vectors(df_train['clean_text'])
        logger.info(f'errors count (new): {train_errors_new}')

        logger.info(f'Generating test document grammatical vectors')
        X_test, test_errors_new = self.batch_get_document_vectors(df_test['clean_text'])
        logger.info(f'errors count (new): {test_errors_new}')

        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)

    def batch_get_document_vectors(self, texts: list[str]) -> tuple[np.ndarray, int]:
        ret = np.zeros((len(texts), len(self.full_tags)))
        error_count = 0
        for i, doc in enumerate(tqdm(self.nlp.pipe(texts, batch_size=self.PROCESSING_BATCH_SIZE), total=len(texts))):
            for token in doc:
                if token.morph:
                    full_tag = ':'.join([
                        f'{tag_name.lower()}_{value.lower()}'
                        for tag_name, value
                        in sorted(token.morph.to_dict().items())
                    ])
                    index = self.full_tags.get(full_tag, -1)
                    if index == -1:
                        error_count += 1
                    ret[i][index] += 1
        return preprocessing.normalize(ret), error_count

    @deprecated("Tested on Cyberbullying - returns the same results as batch_get_document_vectors")
    def sequential_get_document_vectors(self, df: pd.DataFrame) -> tuple[np.ndarray, int]:
        X = np.zeros((len(df), len(self.full_tags)))
        errors = 0
        for i, (_, data) in enumerate(tqdm(df.iterrows(), total=len(df))):
            X[i], ec = self.get_document_vector(data['clean_text'])
            errors += ec
        return X, errors

    @deprecated("used in sequential")
    def get_document_vector(self, text: str) -> tuple[np.ndarray, int]:
        ret = np.zeros(len(self.full_tags))
        doc = self.nlp(text)
        error_count = 0
        for token in doc:
            if token.morph:
                full_tag = ':'.join([
                    f'{tag_name.lower()}_{value.lower()}'
                    for tag_name, value
                    in sorted(token.morph.to_dict().items())
                ])
                index = self.full_tags.get(full_tag, -1)
                if index == -1:
                    error_count += 1
                ret[index] += 1
        return preprocessing.normalize([ret]), error_count
