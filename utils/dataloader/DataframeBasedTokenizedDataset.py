import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


class DataframeBasedTokenizedDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=512, batch_size=-1):
        """Initializes the dataset - if batch size = -1 tokenizes all the data at once"""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids, self.attention_mask = self._tokenize_in_batches(dataframe['clean_text'], batch_size)

    def _tokenize_in_batches(self, texts: pd.Series, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_masks = []

        if batch_size == -1:
            batch_size = len(texts)

        logger.info(f"Starting batch tokenization of {len(texts)} texts with effective batch size of {batch_size}")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size].tolist()
            tokenized_batch = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )

            input_ids.append(tokenized_batch['input_ids'])
            attention_masks.append(tokenized_batch['attention_mask'])

        input_ids_tensor = torch.cat(input_ids, dim=0)
        attention_masks_tensor = torch.cat(attention_masks, dim=0)

        # Sanity check
        assert input_ids_tensor.shape[0] == len(texts)
        assert input_ids_tensor.shape[1] == self.max_length
        assert attention_masks_tensor.shape[0] == len(texts)
        assert attention_masks_tensor.shape[1] == self.max_length

        return input_ids_tensor, attention_masks_tensor

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }
