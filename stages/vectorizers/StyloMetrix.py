import spacy

from stages.vectorizers.Vectorizer import Vectorizer
import stylo_metrix as sm
import numpy as np

from utils.spacy_gpu import autoconfigure_spacy_mode


class StyloMetrix(Vectorizer):
    def __init__(self) -> None:
        autoconfigure_spacy_mode(self.__class__)
        super().__init__()

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        stylo = sm.StyloMetrix('pl')
        X_train: np.ndarray = stylo.transform(df_train["clean_text"]).drop(columns="text").to_numpy()
        X_test: np.ndarray = stylo.transform(df_test["clean_text"]).drop(columns="text").to_numpy()
        
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)