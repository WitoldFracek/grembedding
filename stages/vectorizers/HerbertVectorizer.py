import os
from typing import Union, Literal

import numpy as np
import pandas as pd
from loguru import logger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, AutoModel, BertModel

from stages.vectorizers.Vectorizer import Vectorizer
from utils.dataloader.DataframeBasedTokenizedDataset import DataframeBasedTokenizedDataset


class HerbertVectorizer(Vectorizer):
    TOKENIZER_PATH: str = "allegro/herbert-base-cased"
    MODEL_PATH: str = "allegro/herbert-base-cased"

    INFERENCE_BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4

    def __init__(self, tokenizer_max_length: Union[int, Literal['auto']] = 'auto'):
        super().__init__()

        # Env config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_workers = self.NUM_WORKERS
        self.batch_size = self.INFERENCE_BATCH_SIZE


        # Model config
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)
        self.model: BertModel = AutoModel.from_pretrained(self.MODEL_PATH)
        self.model = self.model.to(self.device)
        _ = self.model.eval()

        self.max_length = self._resolve_max_length(tokenizer_max_length)
        assert self.max_length <= self.tokenizer.model_max_length, (f"max_length value {self.max_length} is too big."
                                                                    f" Max value is {self.tokenizer.model_max_length}")

        logger.info(f"Huggingface '{self.MODEL_PATH}' loaded. Using device: {self.device}."
                    f" Number of workers: {self.num_workers}. Inference batch size: {self.batch_size}")

    def _resolve_max_length(self, max_length: Union[int, Literal['auto']]) -> int:
        if max_length == 'auto':
            return self.tokenizer.model_max_length
        else:
            raise NotImplementedError("Could dynamically check longest text in dataset and set max_length accordingly")

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        train_df, test_df = self.load_train_test_dataframes(dataset, datacleaner)

        train_ds = self.create_dataset(train_df)
        test_ds = self.create_dataset(test_df)

        logger.debug("Dataloader instantiation started")
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)
        logger.debug("Dataloaders ready")

        train_embeds: np.ndarray = self.create_embeddings(train_dl).cpu().detach().numpy()
        test_embeds: np.ndarray = self.create_embeddings(test_dl).cpu().detach().numpy()

        # Sanity check
        assert train_embeds.shape[0] == len(train_df)
        assert train_embeds.shape[1] == self.model.config.hidden_size
        assert test_embeds.shape[0] == len(test_df)
        assert test_embeds.shape[1] == self.model.config.hidden_size

        self.save_as_npy(dataset, datacleaner,
                         X_train=train_embeds, X_test=test_embeds,
                         y_train=train_df['label'].values, y_test=test_df['label'].values)

    def create_dataset(self, df: pd.DataFrame):
        return DataframeBasedTokenizedDataset(
            df,
            self.tokenizer,
            max_length=self.max_length
        )

    def create_embeddings(self, dataloader: DataLoader) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []

        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

            pooler_output = self.convert_batch(input_ids, attention_mask)
            embeddings.append(pooler_output)

        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def convert_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outs.pooler_output


if __name__ == '__main__':
    os.chdir("../..")
    os.environ["DVC_ROOT"] = os.getcwd()

    vectorizer = HerbertVectorizer()
    vectorizer.vectorize("RpTweetsXS", "LemmatizerSM")
