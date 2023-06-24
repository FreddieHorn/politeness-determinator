import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from data_processing import (
    DataPreprocessor,
    DFProcessor,
    create_dataloaders,
    format_time,
    tokenize_words,
)


# Those functions (train, evaluate, r2_score) probably won't be used in the future. Leaving them for now.  # noqa: E501
def train(
    model, optimizer, loss_function, epochs, train_dataloader, device, clip_value=2
):
    train_loss, test_r2 = [], []
    for epoch in range(epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch + 1, epochs))
        print("Training...")
        best_loss = 1e10
        total_train_loss = 0
        t0 = time.time()
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(1).float(), batch_labels.float())
            total_train_loss += loss.item()
            loss.backward()
            # clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print(f"  Average training loss: {avg_train_loss}")
        print("  Training epoch took: {:}".format(training_time))
        train_loss.append(avg_train_loss)
    return model, train_loss


def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss, test_r2 = [], []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2


def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class TextAugmenterForBert:
    def __init__(self, tokenizer_name, df) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.df = df
        # adding tokens from our dataset that are not already in the tokenizer
        token_list = tokenize_words(self.df, "clean_text")
        num_added_toks = self.tokenizer.add_tokens(token_list)
        print(f"Added {num_added_toks} to the tokenizer")

    def augment_data(self):
        encoded_corpus = self.tokenizer(
            text=self.df.clean_text.to_list(),
            add_special_tokens=True,
            padding="max_length",
            truncation="longest_first",
            return_attention_mask=True,
        )

        input_ids = encoded_corpus["input_ids"]
        attention_mask = encoded_corpus["attention_mask"]
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        labels = self.df.offensiveness_score.to_numpy()
        return input_ids, labels, attention_mask


class Regressor(L.LightningModule):
    def __init__(self, bertlike_model, dropout=0.2, lr=1e-3) -> None:
        super().__init__()
        D_in, D_out = 768, 1  # bert (or its derivatives) has 768 outputs
        self.model = bertlike_model
        self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(D_in, D_out))
        self.loss = nn.MSELoss()
        self.lr = lr

        # # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def forward(self, input_ids, attention_masks):
        outputs = self.model(input_ids, attention_masks)
        class_label_output = outputs[
            1
        ]  # Last layer hidden-state of the first token of the sequence (classification token) (but also can be used for regression)
        outputs = self.regressor(class_label_output)
        return torch.tanh(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = tuple(b for b in train_batch)
        outputs = self(batch_inputs, batch_masks)
        loss = self.loss(outputs.squeeze(1).float(), batch_labels.float())
        self.log("train/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = tuple(b for b in val_batch)
        outputs = self(batch_inputs, batch_masks)
        val_loss = self.loss(outputs.squeeze(1).float(), batch_labels.float())
        self.log("val/loss", val_loss)
        return val_loss


class RobertaRegressor(nn.Module):
    def __init__(self, len_tokenizer, dropout=0.2, model_name="roberta-base"):
        super(RobertaRegressor, self).__init__()
        D_in, D_out = 768, 1
        self.roberta = RobertaModel.from_pretrained(model_name, num_labels=1)
        self.roberta.resize_token_embeddings(len(len_tokenizer))
        self.regressor = nn.Sequential(nn.Dropout(dropout), nn.Linear(D_in, D_out))

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[
            1
        ]  # Last layer hidden-state of the first token of the sequence (classification token) (but also can be used for regression)
        outputs = self.regressor(class_label_output)
        return torch.tanh(outputs)


if __name__ == "__main__":
    # TODO move hyperparamethers to a seperate CONFIG file
    test_size = 0.2
    val_size = 0.5
    batch_size = 8
    epochs = 10

    model_name = "roberta-base"

    text_cleaner = DataPreprocessor()
    df_processor = DFProcessor(filename="ruddit_with_text.csv")

    new_df = df_processor.process_df(text_cleaner)

    data_augmentator = TextAugmenterForBert(tokenizer_name=model_name, df=new_df)
    # here we do everything tokenizer-related i.e. encoding corpus, getting input ids etc.
    input_ids, labels, attention_mask = data_augmentator.augment_data()

    # 80% train 10% test 10% val is proposed
    (
        train_inputs,
        test_inputs,
        train_labels,
        test_labels,
        train_masks,
        test_masks,
    ) = train_test_split(
        input_ids, labels, attention_mask, test_size=test_size, shuffle=True
    )
    (
        val_inputs,
        test_inputs,
        val_labels,
        test_labels,
        val_masks,
        test_masks,
    ) = train_test_split(test_inputs, test_labels, test_masks, test_size=val_size)

    train_dataloader = create_dataloaders(
        train_inputs, train_masks, train_labels, batch_size
    )
    test_dataloader = create_dataloaders(
        test_inputs, test_masks, test_labels, batch_size
    )
    val_dataloader = create_dataloaders(val_inputs, val_masks, val_labels, batch_size)

    # here declare bert-like model, just change the "RobertaModel" into something else.
    bertlike_model = RobertaModel.from_pretrained(model_name, num_labels=1)
    bertlike_model.resize_token_embeddings(len(data_augmentator.tokenizer))

    model = Regressor(bertlike_model=bertlike_model)
    wandb.login()
    wandb_logger = WandbLogger(project="rudeness_determinator")

    # TODO early stopping.
    # TODO get more metrics to wandb
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            every_n_train_steps=100,
        ),
    ]
    trainer = L.Trainer(
        accelerator="gpu", max_epochs=50, logger=wandb_logger, callbacks=callbacks
    )

    trainer.fit(model, train_dataloader, val_dataloader)


# NON-LIGHTNING DOWN HERE.
# model = RobertaRegressor(len_tokenizer=len(tokenizer), dropout=0.2)

#     torch.cuda.empty_cache()
#    # model.load_state_dict(torch.load(f"saved_models/{model_name}_{epochs}.pt"))
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("Using GPU.")
#     else:
#         print("No GPU available, using the CPU instead.")
#         device = torch.device("cpu")

#     wandb_logger = WandbLogger(project = "<project_name>")
# model.to(device)

# optimizer = AdamW(model.parameters(),
#               lr=1e-3,
#               eps=1e-8)
# total_steps = len(train_dataloader) * epochs

# loss_function = nn.MSELoss()

# model, train_loss = train(model, optimizer, loss_function, epochs,
#            train_dataloader, device)

# #torch.save(model.state_dict() ,f"saved_models/{model_name}_{epochs}_IMPROVED_TOKENIZER_tanh.pt")
# plt.plot(train_loss, 'b-o', label="Test")
# #plt.plot(test_r2, label="r2")
# plt.show()
