import os
from typing import Union, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
import pytorch_lightning as L

from stages.vectorizers.Vectorizer import Vectorizer
from stages.vectorizers.components.BertClassifier import BertClassifier
from utils.dataloader.DataframeBasedTokenizedDataset import DataframeBasedTokenizedDataset
from utils.larger_model_gpu import resolve_training_batch_size, resolve_inference_batch_size, resolve_num_workers, \
    resolve_torch_matmul_precision, resolve_fine_tune_epochs


class BertLikeTunableModel(Vectorizer):
    TRAINING_BATCH_SIZE: int = resolve_training_batch_size()
    INFERENCE_BATCH_SIZE: int = resolve_inference_batch_size()
    NUM_WORKERS: int = resolve_num_workers()
    TORCH_PRECISION: str = resolve_torch_matmul_precision()

    EPOCHS: int = resolve_fine_tune_epochs()
    HIDDEN_DIM: int = 256
    LR: float = 3e-5
    WARMUP_PERCENT = 0.1

    def __init__(self, model_path: str, tokenizer_path: str, max_tokens_num: int, is_frozen: bool = False):
        super().__init__()
        torch.set_float32_matmul_precision(self.TORCH_PRECISION)

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_frozen = is_frozen

        # Model config
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        self.num_tokens = max_tokens_num
        logger.info(f"Number of workers: {self.NUM_WORKERS}. Batch size: {self.TRAINING_BATCH_SIZE}")

    def vectorize(self, dataset: str, datacleaner: str) -> None:
        train_df, test_df = self.load_train_test_dataframes(dataset, datacleaner)
        initial_train_size, initial_test_size = len(train_df), len(test_df)
        n_labels: int = self._resolve_number_of_labels(train_df)

        finetune_train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42,
                                                     stratify=train_df['label'])

        # MODEL
        model = BertClassifier(
            model_path=self.model_path,
            hidden_dim=self.HIDDEN_DIM,
            lr=self.LR,
            n_classes=n_labels,
            n_batches=self._resolve_steps_per_epoch(finetune_train_df),
            n_epochs=self.EPOCHS,
            warmup_pct=self.WARMUP_PERCENT,
            n_samples_in_train=len(finetune_train_df)
        )
        if self.is_frozen:
            model.freeze()
            logger.warning("Model is frozen")

        # CREATING DATALOADERS
        logger.info(f"Creating dataloaders - max tokens length: {self.num_tokens}")

        finetune_train_ds = self.create_dataset(finetune_train_df, max_tokens_length=self.num_tokens)
        train_ds = self.create_dataset(train_df, max_tokens_length=self.num_tokens)
        val_ds = self.create_dataset(val_df, max_tokens_length=self.num_tokens)
        test_ds = self.create_dataset(test_df, max_tokens_length=self.num_tokens)
        logger.info(f"Created datasets: finetune_train len={len(finetune_train_ds)}, "
                    f"train len={len(train_ds)}, val  len={len(val_ds)}, test  len={len(test_ds)})")

        finetune_train_dl = DataLoader(finetune_train_ds, batch_size=self.TRAINING_BATCH_SIZE,
                                       num_workers=self.NUM_WORKERS, shuffle=True,
                                       persistent_workers=self.NUM_WORKERS > 0, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=self.TRAINING_BATCH_SIZE, shuffle=False,
                            num_workers=self.NUM_WORKERS, persistent_workers=self.NUM_WORKERS > 0,
                            drop_last=True)

        # Not shuffled full DataLoaders for prediction
        train_dl_predict = DataLoader(train_ds, batch_size=self.INFERENCE_BATCH_SIZE, num_workers=self.NUM_WORKERS,
                                      shuffle=False, persistent_workers=self.NUM_WORKERS > 0)
        test_dl_predict = DataLoader(test_ds, batch_size=self.INFERENCE_BATCH_SIZE, num_workers=self.NUM_WORKERS,
                                     shuffle=False, persistent_workers=self.NUM_WORKERS > 0)
        logger.info(f"DataLoaders created. Using train batch size: {self.TRAINING_BATCH_SIZE},"
                    f" inference batch size: {self.INFERENCE_BATCH_SIZE}.")

        # TRAINER
        lightning_loggers_dir = self.get_output_dir(dataset, datacleaner)
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        best_model = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                     dirpath=os.path.join(lightning_loggers_dir, "lightning_logs", "checkpoints"),
                                     filename="monitored_{epoch}_{val_loss:.2f}")
        csv_logger = CSVLogger(save_dir=lightning_loggers_dir, name="lightning_logs")
        trainer = L.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', max_epochs=self.EPOCHS, logger=csv_logger, callbacks=[early_stop, best_model])

        if not self.is_frozen:
            logger.info("Starting fine tuning")
            trainer.fit(model, finetune_train_dl, val_dl)
            logger.info(f"Training complete, best checkpoint: {best_model.best_model_path} - creating embeds")

        logger.info("Preparing embeddings via predict")
        prediction_model_path: Optional[str] = best_model.best_model_path if not self.is_frozen else None

        train_embeds: np.ndarray = self._create_embeddings(
            model, trainer, train_dl_predict, model_path=prediction_model_path).cpu().detach().numpy()
        test_embeds: np.ndarray = self._create_embeddings(
            model, trainer, test_dl_predict, model_path=prediction_model_path).cpu().detach().numpy()

        # Sanity check
        assert train_embeds.shape[0] == initial_train_size
        assert test_embeds.shape[0] == initial_test_size

        self.save_as_npy(dataset, datacleaner,
                         X_train=train_embeds, X_test=test_embeds,
                         y_train=train_df['label'].values, y_test=test_df['label'].values)

        if not self.is_frozen and os.path.exists(best_model.best_model_path):
            logger.info("Removing best model checkpoint")
            os.remove(best_model.best_model_path)

    def create_dataset(self, df: pd.DataFrame, max_tokens_length: int = 512):
        return DataframeBasedTokenizedDataset(
            df,
            self.tokenizer,
            max_length=max_tokens_length
        )

    @staticmethod
    def _create_embeddings(model: BertClassifier, trainer: L.Trainer,
                           dataloader: DataLoader, model_path: str) -> torch.Tensor:
        embeddings: list[torch.Tensor] = trainer.predict(
            model, dataloaders=dataloader, return_predictions=True, ckpt_path=model_path
        )
        result = torch.cat(embeddings, dim=0)
        logger.info(f'Embeddings shape: {result.shape}')
        return result

    @staticmethod
    def _resolve_number_of_labels(train_df: pd.DataFrame) -> int:
        return train_df['label'].nunique()

    def _resolve_steps_per_epoch(self, train_df: pd.DataFrame):
        """Predicted number of optimization steps per epoch"""
        num_train_samples = len(train_df)
        n_steps = num_train_samples // self.TRAINING_BATCH_SIZE
        return n_steps


if __name__ == '__main__':
    os.chdir("../../..")
    os.environ["DVC_ROOT"] = os.getcwd()
    os.environ["GRE_LARGE_MODEL_TRAIN_BATCH_SIZE"] = "32"

    vectorizer = BertLikeTunableModel()
    vectorizer.vectorize("Classics5Authors35Books", "DummyDatacleaner")
