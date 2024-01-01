import sklearn.feature_extraction.text as sklearntext

from stages.vectorizers.Vectorizer import Vectorizer


class TfidfVectorizer(Vectorizer):

    def __init__(self, params) -> None:
        super().__init__()
        self._tfidf = sklearntext.TfidfVectorizer(**params)

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        """
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)
        self._tfidf.fit(df_train['clean_text'].values.tolist())

        X_train = self._tfidf.transform(df_train['clean_text'].values).todense()
        X_test = self._tfidf.transform(df_test['clean_text'].values).todense()
        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)
