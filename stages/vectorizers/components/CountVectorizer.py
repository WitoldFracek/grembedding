import sklearn.feature_extraction.text as sklearntext

from stages.vectorizers.Vectorizer import Vectorizer


class CountVectorizer(Vectorizer):

    def __init__(self, max_features: int) -> None:
        super().__init__()
        self._count_vectorizer = sklearntext.CountVectorizer(max_features=max_features)

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        """
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)
        self._count_vectorizer.fit(df_train['clean_text'].values.tolist())

        X_train = self._count_vectorizer.transform(df_train['clean_text'].values).todense()
        X_test = self._count_vectorizer.transform(df_test['clean_text'].values).todense()
        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    x = CountVectorizer(1000)
    x.vectorize("poleval2019_cyberbullying", "LemmatizerSM")
