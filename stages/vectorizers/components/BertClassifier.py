from functools import lru_cache
from typing import Any

import pytorch_lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, AUROC
from transformers import BertModel, AutoModel, get_linear_schedule_with_warmup


class BertClassifier(L.LightningModule):

    def __init__(self, model_path: str, hidden_dim: int, n_classes: int, n_samples_in_train: int, n_batches: int,
                 n_epochs: int, warmup_pct: float, lr: float = 3e-5):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.bert: BertModel = AutoModel.from_pretrained(model_path)
        self.fc_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, n_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=n_classes)
        self.aucroc = AUROC(task="multiclass", num_classes=n_classes)

        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.warump_pct = warmup_pct

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc_head(output.pooler_output)

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, step_type="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx, step_type="val")
        return loss, logits

    def test_step(self, batch, batch_idx):
        loss, logits = self.common_step(batch, batch_idx, step_type="test")
        return loss, logits

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output

    def common_step(self, batch, batch_idx, step_type):
        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch["attention_mask"]
        labels: torch.Tensor = batch["label"]

        logits = self.forward(input_ids, attention_mask)
        loss = self.criterion(logits, labels.long())

        self.log(f'{step_type}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self.accuracy(logits, labels)
        f1 = self.f1_score(logits, labels)
        auc = self.aucroc(logits, labels)
        self.log(f'{step_type}_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f'{step_type}_aucroc', auc, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f'{step_type}_f1', f1, on_epoch=True, on_step=False, prog_bar=True)

        return loss, logits

    @lru_cache()
    def total_steps(self):
        return self.hparams.n_samples_in_train // self.n_batches * self.n_epochs

    @lru_cache()
    def warmup_steps(self):
        return int(self.warump_pct * self.total_steps())

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps(),
            num_training_steps=self.total_steps()
        )
        return [optimizer], [scheduler]
