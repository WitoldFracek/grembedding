import spacy
from loguru import logger
from tqdm import tqdm
from typing_extensions import deprecated

from stages.datacleaners.DataCleaner import DataCleaner
from spacy.lang.pl import stop_words as pl_stop_words

from utils.spacy_gpu import autoconfigure_spacy_mode, resolve_spacy_batch_size


class LemmatizerSM(DataCleaner):
    PROCESSING_BATCH_SIZE: int = resolve_spacy_batch_size()

    def __init__(self) -> None:
        super().__init__()
        autoconfigure_spacy_mode(self.__class__)
        self.nlp = spacy.load("pl_core_news_sm")

    def clean_data(self, dataset: str) -> None:
        """Lemmatize text in dataset and save it as parquet files"""
        df_train, df_test = self.load_dataset(dataset)

        df_train['clean_text'] = self.batch_lemmatize_text(df_train['text'])
        df_test['clean_text'] = self.batch_lemmatize_text(df_test['text'])

        df_train = df_train.drop(columns=['text'])
        df_test = df_test.drop(columns=['text'])

        self.save_dataframe_as_parquet(dataset, df_train, df_test)

    def batch_lemmatize_text(self, texts: list[str]) -> list[str]:
        clean_texts = []
        for doc in tqdm(self.nlp.pipe(texts, batch_size=self.PROCESSING_BATCH_SIZE), total=len(texts)):
            lemmas = [w.lemma_ for w in doc if w.lemma_ not in pl_stop_words.STOP_WORDS]
            clean_text = ' '.join(lemmas)
            clean_texts.append(clean_text)
        return clean_texts

    @deprecated("use new")
    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        lemmas = [w.lemma_ for w in doc if w.lemma_ not in pl_stop_words.STOP_WORDS]
        clean_text = ' '.join(lemmas)
        return clean_text


if __name__ == "__main__":
    x = LemmatizerSM()
    x.clean_data("poleval2019_cyberbullying")
