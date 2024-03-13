import os

import pandas as pd
from typing_extensions import deprecated

from stages.vectorizers.Vectorizer import Vectorizer
import numpy as np
import spacy
from sklearn import preprocessing
from loguru import logger
from tqdm import tqdm

from utils.spacy_gpu import autoconfigure_spacy_mode, resolve_spacy_batch_size

TAGS = {'abbr_yes': 0, 'adptype_post': 1, 'adptype_prep': 2, 'animacy_hum': 3, 'animacy_inan': 4, 'animacy_nhum': 5,
        'aspect_imp': 6, 'aspect_imp,perf': 7, 'aspect_perf': 8, 'case_acc': 9, 'case_dat': 10, 'case_gen': 11,
        'case_ins': 12, 'case_loc': 13, 'case_nom': 14, 'case_voc': 15, 'clitic_yes': 16, 'conjtype_comp': 17,
        'conjtype_oper': 18, 'conjtype_pred': 19, 'degree_cmp': 20, 'degree_pos': 21, 'degree_sup': 22,
        'emphatic_yes': 23, 'foreign_yes': 24, 'gender_fem': 25, 'gender_masc': 26, 'gender_neut': 27, 'hyph_yes': 28,
        'mood_imp': 29, 'mood_ind': 30, 'number[psor]_plur': 31, 'number[psor]_sing': 32, 'number_plur': 33,
        'number_plur,sing': 34, 'number_ptan': 35, 'number_sing': 36, 'numform_digit': 37, 'numform_roman': 38,
        'numform_word': 39, 'numtype_card': 40, 'numtype_ord': 41, 'numtype_sets': 42, 'parttype_int': 43,
        'parttype_mod': 44, 'person_0': 45, 'person_1': 46, 'person_2': 47, 'person_3': 48, 'polarity_neg': 49,
        'polarity_pos': 50, 'polite_depr': 51, 'poss_yes': 52, 'prepcase_npr': 53, 'prepcase_pre': 54,
        'prontype_dem': 55, 'prontype_ind': 56, 'prontype_int': 57, 'prontype_neg': 58, 'prontype_prs': 59,
        'prontype_rel': 60, 'prontype_tot': 61, 'pun_no': 62, 'pun_yes': 63, 'punctside_fin': 64, 'punctside_ini': 65,
        'puncttype_brck': 66, 'puncttype_colo': 67, 'puncttype_comm': 68, 'puncttype_dash': 69, 'puncttype_elip': 70,
        'puncttype_excl': 71, 'puncttype_peri': 72, 'puncttype_qest': 73, 'puncttype_quot': 74, 'puncttype_semi': 75,
        'puncttype_slsh': 76, 'reflex_yes': 77, 'tense_fut': 78, 'tense_past': 79, 'tense_pres': 80, 'variant_long': 81,
        'variant_short': 82, 'verbform_conv': 83, 'verbform_fin': 84, 'verbform_inf': 85, 'verbform_part': 86,
        'verbform_vnoun': 87, 'verbtype_mod': 88, 'verbtype_quasi': 89, 'voice_act': 90, 'voice_pass': 91}


class SpacyMorphTagVectorizer(Vectorizer):

    PROCESSING_BATCH_SIZE: int = resolve_spacy_batch_size()

    def __init__(self) -> None:
        super().__init__()
        autoconfigure_spacy_mode(self.__class__)
        self.nlp = spacy.load("pl_core_news_lg")

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        logger.info(f'Generating train document grammatical vectors')
        X_train = self.batch_get_document_vectors(df_train['clean_text'])

        logger.info(f'Generating test document grammatical vectors using')
        X_test = self.batch_get_document_vectors(df_test['clean_text'])

        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)

    def batch_get_document_vectors(self, texts: list[str]) -> np.ndarray:
        ret = np.zeros((len(texts), len(TAGS)))
        for i, doc in enumerate(tqdm(self.nlp.pipe(texts, batch_size=self.PROCESSING_BATCH_SIZE), total=len(texts))):
            for token in doc:
                if token.morph:
                    for tag_name, value in token.morph.to_dict().items():
                        t = f'{tag_name.lower()}_{value.lower()}'
                        index = TAGS.get(t, -1)
                        if index == -1:
                            continue
                        ret[i][index] += 1
        return preprocessing.normalize(ret)

    @deprecated("Tested on Cyberbullying - returns the same results as batch_get_document_vectors")
    def sequential_get_document_vectors(self, df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), len(TAGS)))
        for i, (_, data) in enumerate(tqdm(df.iterrows(), total=len(df))):
            X[i] = self.get_document_vector(data['clean_text'])
        return X

    @deprecated("used for sequential")
    def get_document_vector(self, text: str) -> np.ndarray:
        ret = np.zeros(len(TAGS))
        doc = self.nlp(text)
        for token in doc:
            if token.morph:
                for tag_name, value in token.morph.to_dict().items():
                    t = f'{tag_name.lower()}_{value.lower()}'
                    index = TAGS.get(t, -1)
                    if index == -1:
                        continue
                    ret[index] += 1
        return preprocessing.normalize([ret])
