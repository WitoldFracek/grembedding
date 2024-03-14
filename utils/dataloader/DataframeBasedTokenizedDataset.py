import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger
from transformers import PreTrainedTokenizer


class DataframeBasedTokenizedDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=512, batch_size=-1):
        """Initializes the dataset - if batch size = -1 tokenizes all the data at once"""
        self.df = dataframe.reset_index()
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode_plus(
            self.df.iloc[idx]['clean_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        return {
            "input_ids": tokens['input_ids'].squeeze(),
            "attention_mask": tokens['attention_mask'].squeeze(),
            "label": self.df.iloc[idx]['label']
        }
