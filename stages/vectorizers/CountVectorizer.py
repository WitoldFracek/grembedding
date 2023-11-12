from stages.vectorizers.Vectorizer import Vectorizer
import sklearn.feature_extraction.text as sklearntext
from typing import List


class CountVectorizer(Vectorizer):
    
    def __init__(self) -> None:
        super().__init__()
        self._count_vectorizer = sklearntext.CountVectorizer()

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        """
        :dataset: name of dataset
        :datacleaner: name of datacleaner that was used to clean the data
        """
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)
        self._count_vectorizer.fit(df_train['clean_text'].values.tolist())
        df_train["vectorized_text"] = df_train['clean_text'].apply(self.transform)
        df_test["vectorized_text"] = df_test['clean_text'].apply(self.transform)
        df_train = df_train.drop(columns = 'clean_text')
        df_test = df_test.drop(columns = 'clean_text')
        self.save_dataframe_as_parquet(dataset, datacleaner, df_train, df_test)

    def transform(self, clean_text: str) -> List[int]:
        return self._count_vectorizer.transform([clean_text]).todense().tolist()[0]


if __name__ == "__main__":
    x = CountVectorizer()
    x.vectorize("poleval2019_cyberbullying", "LemmatizerSM")
