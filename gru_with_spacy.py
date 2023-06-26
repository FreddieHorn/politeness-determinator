# !python -m spacy download en_core_web_md

import argparse
import os

import lightning as L
import numpy as np
import pandas as pd
import spacy
import torch
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import r2_score

wandb.login()

# load the dataset
ruddit = pd.read_csv("ruddit_with_text.csv")
nlp_md = spacy.load("en_core_web_md")

# exclude deleted comments
filter = ruddit[ruddit["comment_body"] != "[deleted]"]
comments = filter["comment_body"].tolist()
# tokenize the comments and build embedding vectors
comments = [nlp_md(comment) for comment in comments]
scores = filter["offensiveness_score"].tolist()
scores = torch.tensor(scores)
# replace each token with its embedding vector
embeddings = [np.array([token.vector for token in comment]) for comment in comments]


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


# pads the comments in each batch to the size of the longest comment
# instead of padding all the database
def collate(batch):
    batch_comments, batch_scores = zip(*batch)
    return pad_sequence(batch_comments, batch_first=True), torch.tensor(batch_scores)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
    collate_fn=collate,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
    collate_fn=collate,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=os.cpu_count(),  # type: ignore
    collate_fn=collate,
)


class GRURegressor(L.LightningModule):
    """A GRU model with a linear regression layer

    Args:
        hidden_size (int): hidden size of the GRU layer (default: `128`).
        num_layers (int): number of GRU cells within the GRU layer (default: `1`).
        dropout (float): GRU layer dropout (default: `0.0`).
        accuracy_threshold (float): maximum value a prediction can be away from the true value before it is considered inaccurate (default: `0.05`).
        lr (float): learning rate (default: `1e-3`).
    """  # noqa: E501

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        accuracy_threshold: float = 0.05,
        lr: int = 1e-3,  # type: ignore
    ):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = accuracy_threshold
        self.lr = lr

        self.gru = nn.GRU(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
        )
        # regression layer
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        output, h = self.gru(X.permute(1, 0, 2))
        y = output[-1]
        y = self.regressor(y).squeeze()
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
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", "-n", type=int, default=3)
    parser.add_argument("--dropout", "-d", type=float, default=0.0)
    parser.add_argument("--accuracy_threshold", "-t", type=float, default=0.05)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--epochs", "-e", type=int, default=30)
    args = parser.parse_args()

    gru = GRURegressor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        accuracy_threshold=args.accuracy_threshold,
        lr=args.lr,
    )

    logger = WandbLogger(project="rudeness_determinator")

    trainer = L.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        enable_model_summary=False,
    )

    trainer.fit(gru, train_loader, validation_loader)
    trainer.test(gru, test_loader)
    wandb.finish()
