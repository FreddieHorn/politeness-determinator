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


class LinearModel(L.LightningModule):
    """A simple 5-layered linear model that will be used as a baseline to compare the performance with the other more sophisticated models.

    Args:
        accuracy_threshold (float): maximum value a prediction can be away from the true value before it is considered inaccurate (default: `0.05`).
        lr (float): learning rate (default: `1e-3`).
    """  # noqa: E501

    def __init__(
        self,
        accuracy_threshold: float = 0.1,
        lr: int = 1e-3,  # type: ignore
    ):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = accuracy_threshold
        self.lr = lr

        self.regressor = nn.Sequential(
            nn.Linear(300, 300),
            nn.GELU(),
            nn.Linear(300, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        y = self.regressor(X).squeeze()
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
    parser.add_argument("--accuracy_threshold", "-t", type=float, default=0.1)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--epochs", "-e", type=int, default=50)
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

    # build embedding vector for each comment
    print("Building spaCy embeddings for the comments")
    nlp_md = spacy.load("en_core_web_md")
    preprocessor = DataPreprocessor()
    embeddings = np.array(
        [
            nlp_md(preprocessor.process(comment)).vector
            for comment in tqdm(comments)
            if preprocessor.process(comment)
        ]
    )
    scores = ruddit_df["offensiveness_score"].tolist()
    scores = torch.tensor(scores)

    ruddit = list(zip(embeddings, scores))
    shuffle(ruddit)
    validation_split = int(round(len(ruddit) * 0.7))
    test_split = int(round(len(ruddit) * 0.8))

    train_dataset = RudditDataset(split="training")
    validation_dataset = RudditDataset(split="validation")
    test_dataset = RudditDataset(split="test")

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

    model = LinearModel(
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

    trainer.fit(model, train_loader, validation_loader)
    trainer.test(model, test_loader)
    wandb.finish()
