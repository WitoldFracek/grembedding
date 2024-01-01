import spacy

from stages.datacleaners.DataCleaner import DataCleaner
from spacy.lang.pl import stop_words as pl_stop_words


class LemmatizerSM(DataCleaner):

    def __init__(self) -> None:
        super().__init__()
        self._nlp = spacy.load("pl_core_news_sm")

    def clean_data(self, dataset: str) -> None:
        """Lemmatize text in dataset and save it as parquet files

        The output file has columns `clean_text` and `label`

        Args:
            dataset: The name of the dataset to be cleaned
        """
        df_train, df_test = self.load_dataset(dataset)

        df_train['clean_text'] = df_train['text'].apply(self.lemmatize_text)
        df_test['clean_text'] = df_test['text'].apply(self.lemmatize_text)

        df_train = df_train.drop(columns='text')
        df_test = df_test.drop(columns='text')

        self.save_dataframe_as_parquet(dataset, df_train, df_test)

    def lemmatize_text(self, text: str) -> str:
        doc = self._nlp(text)
        lemmas = [w.lemma_ for w in doc if w.lemma_ not in pl_stop_words.STOP_WORDS]    # NOTE: dodane stopwordsy
        clean_text = ' '.join(lemmas)
        return clean_text


if __name__ == "__main__":
    x = LemmatizerSM()
    x.clean_data("poleval2019_cyberbullying")
