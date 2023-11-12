import spacy

from objects.datacleaners.DataCleaner import DataCleaner
from loguru import logger


class LemmatizerSM(DataCleaner):

    def __init__(self) -> None:
        super().__init__()
        self._nlp = spacy.load("pl_core_news_sm")

    def clean_data(self, dataset: str) -> None:
        """
        :dataset: name of dataset
        :return: dict of train and test data
        """
        logger.info(f"Executing clean_data for dataset: {dataset}")
        df_train, df_test = self.load_dataset(dataset)
        df_train['clean_text'] = df_train['text'].apply(self.lemmatize_text)
        df_test['clean_text'] = df_train['text'].apply(self.lemmatize_text)
        df_train = df_train.drop(columns = 'text')
        df_test = df_test.drop(columns = 'text')
        self.safe_dataframe_as_parquet(dataset, df_train, df_test)



    def lemmatize_text(self, text: str) -> str:
        doc = self._nlp(text)
        lemmas = [w.lemma_ for w in doc]
        clean_text = ' '.join(lemmas)
        return clean_text
    
# x = LemmatizerSM()
# x.clean_data("poleval2019_cyberbullying")
