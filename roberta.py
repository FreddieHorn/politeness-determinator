from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification, AutoTokenizer
from data_processing import DataPreprocessor, create_dataloaders, DFProcessor, format_time, tokenize_words
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import time
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchmetrics import R2Score

class TextAugmenterForBert:
    def __init__(self, tokenizer_name, df) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.df = df
        #adding tokens from our dataset that are not already in the tokenizer - RUINED RESULTS!!
        # token_list = tokenize_words(self.df, "comment_body")
        # num_added_toks = self.tokenizer.add_tokens(token_list)
        # print(f"Added {num_added_toks} to the tokenizer")
    
    def augment_data(self):
        encoded_corpus = self.tokenizer(text = self.df.comment_body.to_list(),
                            add_special_tokens=True,
                            padding="max_length",
                            truncation=True,
                            max_length=200,
                            return_attention_mask=True)

        input_ids = encoded_corpus['input_ids']
        attention_mask = encoded_corpus['attention_mask']
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        labels = self.df.offensiveness_score.to_numpy()
        return input_ids, labels, attention_mask

class Regressor(L.LightningModule):
    def __init__(self, bertlike_model, dropout=0.2, lr=1e-3) -> None:
        super().__init__()
        D_in, D_out = 768, 1 #bert (or its derivatives) has 768 outputs 
        self.model = bertlike_model
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D_in, D_out))
        self.loss = nn.MSELoss()
        self.R2 = R2Score()
        self.MAE = nn.L1Loss()
        self.lr = lr
        self.drop = nn.Dropout(dropout)
        
        # # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def forward(self, input_ids, attention_masks):
        outputs = self.model(input_ids, attention_masks)
        pooled_output = outputs[0].mean(dim=1) #Last layer hidden-state of the first token of the sequence (classification token) (but also can be used for regression)
        output = self.regressor(pooled_output)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        self.model.train()
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in train_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs)
        loss = self.loss(preds.squeeze(1).float(), 
                        batch_labels.float())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        self.model.eval()
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in val_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs)
        val_loss = self.loss(preds.squeeze(1).float(), 
                        batch_labels.float())
        self.log("val_loss", val_loss)
        return val_loss
    
    def test_step(self, test_batch, batch_idx):
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in test_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs) 
        mae_loss = self.loss(preds.squeeze(1).float(), 
                        batch_labels.float())
        r2_score = self.R2(preds.squeeze(1).float(), #warning! using batch_size = 8 made the last test step have only one sample in 
                            batch_labels.float())   #preds and batch labels. r2 needs > 1 samples so I increased batch size to 16
        self.log("test/mae_loss", mae_loss)
        self.log("test/r2_score", r2_score)
        return mae_loss, r2_score
    
if __name__=="__main__":
    #TODO move hyperparamethers to a seperate CONFIG file
    test_size = 0.2
    val_size = 0.5
    batch_size = 16
    epochs = 5

    model_name = 'distilbert-base-uncased'

    text_cleaner = DataPreprocessor()
    df_processor = DFProcessor(filename='ruddit_with_text.csv')

    new_df = df_processor.process_df(text_cleaner)

    data_augmentator = TextAugmenterForBert(tokenizer_name=model_name, df = new_df)
    #here we do everything tokenizer-related i.e. encoding corpus, getting input ids etc.
    input_ids, labels, attention_mask = data_augmentator.augment_data()
    
    #80% train 10% test 10% val is proposed
    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = \
            train_test_split(input_ids, labels, attention_mask, test_size=test_size, shuffle=True) #note that shuffling is done here not in dataloaders
    val_inputs, test_inputs, val_labels, test_labels, val_masks, test_masks = \
            train_test_split(test_inputs, test_labels, test_masks, test_size=val_size)
    
    train_dataloader = create_dataloaders(train_inputs, train_masks, 
                                        train_labels, batch_size)
    test_dataloader = create_dataloaders(test_inputs, test_masks, 
                                        test_labels, batch_size)
    val_dataloader = create_dataloaders(val_inputs, val_masks, 
                                        val_labels, batch_size)
    
    #here declare bert-like model, just change the "RobertaModel" into something else.
    bertlike_model = DistilBertModel.from_pretrained(model_name, num_labels = 1)
    #bertlike_model.resize_token_embeddings(len(data_augmentator.tokenizer)) #OPTIONAL - IN PRACTISE RUINED RESULTS!!!
    model = Regressor(bertlike_model=bertlike_model, lr=2e-4)
    wandb_logger = WandbLogger(project = "rudeness_determinator")
    #TODO early stopping.
    #TODO get more metrics to wandb
    wandb.init()
    wandb.login()
    callbacks = [
        ModelCheckpoint(
            dirpath = "checkpoints",
            monitor="val_loss", 
            save_top_k=1,
            filename="DistilBERT-NO_ADDED_TOKENS-YES_PREPROCESS-{epoch:02d}-{val_loss:.2f}",
        ),
    ]
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs = 5 , 
        logger = wandb_logger, 
        callbacks = callbacks
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    #trainer.test(model, ckpt_path="CHECKPOINT_NAME", dataloaders=test_dataloader) #Uncomment to TEST
    #wandb.save('checkpoints/*ckpt*') #save checkpoint to wandb 