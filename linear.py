# !python -m spacy download en_core_web_md

import argparse
import os

import lightning as L
import numpy as np
import pandas as pd
import spacy
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

import wandb

wandb.login()

# load the dataset
ruddit = pd.read_csv("ruddit_with_text.csv")
nlp_md = spacy.load("en_core_web_md")

# exclude deleted comments
filter = ruddit[ruddit["comment_body"] != "[deleted]"]
comments = filter["comment_body"].tolist()
# build embedding vector for each comment
embeddings = np.array([nlp_md(comment).vector for comment in tqdm(comments)])
scores = filter["offensiveness_score"].tolist()
scores = torch.tensor(scores)


class RudditDataset(Dataset):
    def __init__(
        self,
        training: bool = True,
        validation: bool = False,
        test: bool = False,
    ):
        super().__init__()
        data = list(zip(embeddings, scores))
        validation_split = int(round(len(data) * 0.7))
        test_split = int(round(len(data) * 0.8))

        if training:
            self.data = data[:validation_split]
        elif validation:
            self.data = data[validation_split:test_split]
        elif test:
            self.data = data[test_split:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return torch.tensor(item[0]), item[1]


train_dataset = RudditDataset()
validation_dataset = RudditDataset(validation=True)
test_dataset = RudditDataset(test=True)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
)


class LinearModel(L.LightningModule):
    """A simple 5-layered linear model that will be used as a baseline to compare the performance with the other more sophisticated models.

    Args:
        accuracy_threshold (float): maximum value a prediction can be away from the true value before it is considered inaccurate (default: `0.05`).
        lr (float): learning rate (default: `1e-3`).
    """  # noqa: E501

    def __init__(
        self,
        accuracy_threshold: float = 0.05,
        lr: int = 1e-3,  # type: ignore
    ):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = accuracy_threshold
        self.lr = lr

        self.regressor = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        y = self.regressor(X).squeeze()
        return y

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        comments, scores = batch
        ratings = self(comments)
        loss = mse_loss(ratings, scores)
        self.log("training loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        comments, scores = batch
        ratings = self(comments)
        loss = mse_loss(ratings, scores)
        self.log("validation loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        comments, scores = batch
        ratings = self(comments)
        score = r2_score(ratings, scores)
        accuracy = torch.sum(torch.abs(scores - ratings) < self.threshold).item() / len(
            scores
        )
        loss = l1_loss(ratings, scores)
        self.log("test R2 score", score, prog_bar=True)
        self.log("test accuracy", accuracy, prog_bar=True)
        self.log("test loss", loss, prog_bar=True)
        # combine the 3 metrics
        return score * accuracy - loss

    # predict a score for a single comment
    def predict(self, X: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            y = self(X.unsqueeze(0))
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy_threshold", "-t", type=float, default=0.05)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    args = parser.parse_args()

    model = LinearModel(
        accuracy_threshold=args.accuracy_threshold,
        lr=args.lr,
    )

    logger = WandbLogger(project="rudeness_determinator")

    trainer = L.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, test_loader)
    wandb.finish()
