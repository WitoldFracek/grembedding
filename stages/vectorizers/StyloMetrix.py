import spacy

from stages.vectorizers.Vectorizer import Vectorizer
import stylo_metrix as sm

from utils.spacy_gpu import autoconfigure_spacy_mode


class StyloMetrix(Vectorizer):
    def __init__(self) -> None:
        autoconfigure_spacy_mode(self.__class__)
        super().__init__()

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        stylo = sm.StyloMetrix('pl')
        X_train = stylo.transform(df_train["clean_text"]).drop(columns="text").to_numpy()
        X_test = stylo.transform(df_test["clean_text"]).drop(columns="text").to_numpy()
        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)