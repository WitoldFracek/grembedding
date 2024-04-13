from loguru import logger
import spacy
from spacy.language import Language
from tqdm import tqdm
from stages.datacleaners.DataCleaner import DataCleaner
from typing import Iterable
from utils.spacy_gpu import autoconfigure_spacy_mode, resolve_spacy_batch_size


class ProperNamesMasker(DataCleaner):

    PROCESSING_BATCH_SIZE: int = resolve_spacy_batch_size()
    MASKING_TOKEN = '[MASK]'

    def clean_data(self, dataset: str) -> None:
        train, test = self.load_dataset(dataset)

        autoconfigure_spacy_mode(self.__class__)
        nlp = spacy.load("pl_core_news_sm")

        logger.info(f'Masking proper names in train set')
        masked_train = self.mask_proper_names(train['text'], nlp)
        train['clean_text'] = masked_train
        train.drop(columns=['text'], inplace=True)

        logger.info(f'Masking proper names in test set')
        masked_test = self.mask_proper_names(test['text'], nlp)
        test['clean_text'] = masked_test
        test.drop(columns=['text'], inplace=True)

        self.save_dataframe_as_parquet(dataset, train, test)
    
    def mask_proper_names(self, texts: list[str], nlp: Language) -> Iterable[str]:
        ret = []
        for doc in tqdm(nlp.pipe(texts, batch_size=self.PROCESSING_BATCH_SIZE), total=len(texts)):
            clean_text = ''
            for token in doc:
                pos = token.pos_
                if pos == 'PROPN':
                    ...
                else:
                    clean_text += token.text + ' '
            ret.append(clean_text)
        return ret
