from stages.vectorizers.Vectorizer import Vectorizer
from flair.embeddings import BytePairEmbeddings
from flair.data import Sentence, Token
from typing import Literal
import torch
import numpy as np
from utils.impl import todo


class DPEBPVectoriser(Vectorizer):
    def __init__(self, dim: Literal[100, 200, 400, 600] = 100, agg_metohd: Literal['avg', 'sum'] = 'avg'):
        assert dim in [100, 200, 400, 600], f'dim value "{dim}" is not allowed. Allowed values are [100, 200, 400, 600]'
        assert agg_metohd in ['avg', 'sum'], f'agg_method "{agg_metohd}" is not allowed. Allowed values are [\'sum\', \'avg\']'
        self.embedder = BytePairEmbeddings('pl', dim=dim//2)
        self.__dim = dim
        self.__agg_method = agg_metohd
    
    def vectorize(self, dataset: str, datacleaner: str) -> None:
        df_train, df_test = self.load_train_test_dataframes(dataset, datacleaner)

        train_sentences = list(map(Sentence, df_train['clean_text']))
        test_sentences = list(map(Sentence, df_test['clean_text']))
        X_train = np.vstack(self.embed_documents(train_sentences))
        X_test = np.vstack(self.embed_documents(test_sentences))

        y_train = df_train['label'].values
        y_test = df_test['label'].values

        self.save_as_npy(dataset, datacleaner, X_train, X_test, y_train, y_test)
        
    
    def embed_documents(self, sentences: list[Sentence]) -> list[np.ndarray]:
        res: list[Sentence] = self.embedder.embed(sentences)
        embeddings = list(map(self.__get_sentence_embedding, res))
        return embeddings
    
    def __get_sentence_embedding(self, sentence: Sentence) -> np.ndarray:
        tokens = sentence.tokens
        embeddings = list(map(lambda t: t.embedding, tokens))
        concat = torch.vstack(embeddings)
        if self.__agg_method == 'avg':
            return torch.mean(concat, dim=0).detach().numpy()
        return torch.sum(concat, dim=0).detach().numpy()
    
    @property
    def agg_method(self) -> str:
        return self.__agg_method