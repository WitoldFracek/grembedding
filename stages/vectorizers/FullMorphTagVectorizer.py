from stages.vectorizers.Vectorizer import Vectorizer
import numpy as np
import spacy
from sklearn import preprocessing
from loguru import logger
from tqdm import tqdm
import json
import os


class FullMorphTagVectorizer(Vectorizer):
    def __init__(self) -> None:
        super().__init__()
        dir_ = Vectorizer.get_vectoriser_data_dir()
        path = os.path.join(dir_, 'full_morphological_tags.json')
        if not os.path.exists(path):
            raise Exception(f'path "{path}" does not exist. Could not load morphological tags.')
        
        with open(path, 'r', encoding='utf-8') as file:
            self.full_tags: dict[str, int] = json.load(file)

        spacy.require_gpu()
        self.nlp = spacy.load("pl_core_news_lg")

    def vectorize(self, dataset: str, datacleaner: str):
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        X_train = np.zeros((len(df_train), len(self.full_tags)))
        X_test = np.zeros((len(df_test), len(self.full_tags)))

        logger.info(f'generating train document gramatical vectors')
        errors = 0
        for i, (_, data) in enumerate(tqdm(df_train.iterrows(), total=len(df_train))):
            X_train[i], ec = self.get_document_vector(data['clean_text'])
            errors += ec
        logger.info(f'errors count: {errors}')

        logger.info(f'generating test document gramatical vectors')
        errors = 0
        for i, (_, data) in enumerate(tqdm(df_test.iterrows(), total=len(df_test))):
            X_test[i], ec = self.get_document_vector(data['clean_text'])
            errors += ec
        logger.info(f'errors count: {errors}')
        
        
        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)
    
    def get_document_vector(self, text: str) -> np.ndarray:
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