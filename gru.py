# !python -m spacy download en_core_web_md

import argparse
import os
from operator import itemgetter
from random import shuffle

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

import wandb
from data_processing import DFProcessor, DataPreprocessor


class RudditDataset(Dataset):
    def __init__(self, split: str):
        super().__init__()
        if split == "training":
            self.data = ruddit[:validation_split]
        elif split == "validation":
            self.data = ruddit[validation_split:test_split]
        elif split == "test":
            self.data = sorted(ruddit[test_split:], key=itemgetter(1), reverse=True)
        else:
            raise ValueError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return torch.tensor(item[0]), item[1]


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
        accuracy_threshold: float = 0.1,
        lr: int = 5e-3,  # type: ignore
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
            nn.GELU(),
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
        predictions = self(comments)
        loss = mse_loss(predictions, scores)
        self.log("training loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        comments, scores = batch
        predictions = self(comments)
        loss = mse_loss(predictions, scores)
        self.log("validation loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.scores, self.predictions = [], []

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        comments, scores = batch
        predictions = self(comments)
        self.scores += scores.tolist()
        self.predictions += predictions.tolist()
        score = r2_score(predictions, scores)
        accuracy = torch.sum(
            torch.abs(scores - predictions) < self.threshold
        ).item() / len(scores)
        loss = l1_loss(predictions, scores)
        self.log("test R2 score", score, prog_bar=True)
        self.log("test accuracy", accuracy, prog_bar=True)
        self.log("test loss", loss, prog_bar=True)
        # combine the 3 metrics
        return score * accuracy - loss

    def on_test_epoch_end(self) -> None:
        fig = plt.figure(figsize=(15, 10))
        plt.xlabel("comments")
        plt.ylabel("offensiveness score")
        plt.title("predicted scores vs true scores")
        sns.scatterplot(
            pd.DataFrame(
                list(zip(self.predictions, self.scores)),
                columns=["predicted scores", "true scores"],
            )
        )
        wandb.log({"comparison": wandb.Image(fig)})

    # predict a score for a single comment
    def predict(self, X: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            y = self(X.unsqueeze(0))
        return y

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", "-n", type=int, default=3)
    parser.add_argument("--dropout", "-d", type=float, default=0.0)
    parser.add_argument("--accuracy_threshold", "-t", type=float, default=0.1)
    parser.add_argument("--lr", "-l", type=float, default=5e-3)
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--run", "-r", type=str, default=None)
    parser.add_argument("--posts", "-p", type=bool, default=False)
    args = parser.parse_args()

    filename = "ruddit_with_text.csv"
    if args.posts:
        ruddit_df = DFProcessor(filename).process_df()
        comments = ruddit_df["title_body"].tolist()
    else:
        ruddit_df = pd.read_csv(filename)
        # exclude deleted comments
        ruddit_df = ruddit_df[ruddit_df["comment_body"] != "[deleted]"]
        comments = ruddit_df["comment_body"].tolist()

    # tokenize the comments and build embedding vectors
    print("Building spaCy embeddings for the comments")
    nlp_md = spacy.load("en_core_web_md")
    preprocessor = DataPreprocessor()
    comments = [
        nlp_md(preprocessor.process(comment))
        for comment in tqdm(comments)
        if preprocessor.process(comment)
    ]
    scores = ruddit_df["offensiveness_score"].tolist()
    scores = torch.tensor(scores)
    # replace each token with its embedding vector
    embeddings = [np.array([token.vector for token in comment]) for comment in comments]

    ruddit = list(zip(embeddings, scores))
    shuffle(ruddit)
    validation_split = int(round(len(ruddit) * 0.7))
    test_split = int(round(len(ruddit) * 0.8))

    train_dataset = RudditDataset(split="training")
    validation_dataset = RudditDataset(split="validation")
    test_dataset = RudditDataset(split="test")

    # pads the comments in each batch to the size of the longest comment
    # instead of padding all the database
    def collate(batch):
        batch_comments, batch_scores = zip(*batch)
        return pad_sequence(batch_comments, batch_first=True), torch.tensor(
            batch_scores
        )

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

    gru = GRURegressor(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        accuracy_threshold=args.accuracy_threshold,
        lr=args.lr,
    )

    wandb.login()
    logger = WandbLogger(project="rudeness_determinator", name=args.run)

    trainer = L.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        enable_model_summary=False,
    )

    trainer.fit(gru, train_loader, validation_loader)
    trainer.test(gru, test_loader)
    wandb.finish()
