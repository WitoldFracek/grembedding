import spacy

from stages.vectorizers.Vectorizer import Vectorizer
from sklearn.model_selection import train_test_split
import stylo_metrix as sm
import numpy as np

from utils.spacy_gpu import autoconfigure_spacy_mode

TRAIN_SAMPLES = 10_000

class StyloMetrix(Vectorizer):
    def __init__(self) -> None:
        autoconfigure_spacy_mode(self.__class__)
        super().__init__()

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        if len(df_train) > TRAIN_SAMPLES:
            ratio = TRAIN_SAMPLES/len(df_train)
            df_train, _ = train_test_split(df_train, train_size=ratio, stratify=df_train["label"])
            df_test, _ = train_test_split(df_test, train_size=ratio, stratify=df_test["label"])

        stylo = sm.StyloMetrix('pl')
        X_train: np.ndarray = stylo.transform(df_train["clean_text"]).drop(columns="text").to_numpy()
        X_test: np.ndarray = stylo.transform(df_test["clean_text"]).drop(columns="text").to_numpy()
        
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)