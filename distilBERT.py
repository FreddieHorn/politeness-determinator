import torch.nn as nn
import torch
import numpy as np
import wandb
import lightning as L

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import DistilBertModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score

from data_processing import DataPreprocessor, create_dataloaders, DFProcessor

class TextAugmenterForBert:
    """Stores tokenizer denoted by tokenizer_name and possesses augment data method, which tokenizes 
    comments which the dataset consists of.
    Args:
        tokenizer_name (str): name of the tokenizer to retrieve from AutoTokenizer
        df (DataFrame): dataset ['comment_body', 'offensiveness score']
    """
    def __init__(self, tokenizer_name: str, df) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.df = df
        #adding tokens from our dataset that are not already in the tokenizer - RUINED RESULTS!!
        # token_list = tokenize_words(self.df, "comment_body")
        # num_added_toks = self.tokenizer.add_tokens(token_list)
        # print(f"Added {num_added_toks} to the tokenizer")
    
    def encode_data(self, posts_included = True):
        """Tokenizes input data using tokenizer declared in __init__. 

        Args:
            posts_included (bool, optional): Whether we want to include posts or not. Defaults to True.

        Returns:
            input_ids, labels, attention_mask: Tokenized input sentences, rudeness level and attention mask. This format
            is required for transformers BERT-like models.
        """
        if posts_included:
            encoded_corpus = self.tokenizer(self.df.post_title.to_list(), self.df.comment_body.to_list(),
                                add_special_tokens=True,
                                padding="max_length",
                                truncation=True,
                                max_length=200,
                                return_attention_mask=True)
        else:
            encoded_corpus = self.tokenizer(self.df.comment_body.to_list(),
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
    """This module consists of distilBERT with regression layer on top of it so it suits the given task of determining the offensiveness score. 

    Args:
        bertlike_model: model from the big family of BERTs. distilBERT is used, but it is capable of handling other models (like RoBERTA) 
        with a little to no tweaks in the code.
        total_training_steps (int): number of training steps. This number is used in a scheduler which modifies the lr based on the current
        training step
        dropout (float, optional): Dropout in a regression layer placed on top of distilBERT. Defaults to 0.2.
        lr (float, optional): learning rate parameter. Defaults to 1e-3.
    """
    def __init__(self, bertlike_model, total_training_steps = 0, dropout=0.2, lr=1e-3,
                 accuracy_threshold: float = 0.05) -> None:
        super().__init__()
        D_in, D_out = 768, 1 #bert (or its derivatives) has 768 outputs 
        self.model = bertlike_model
        #self.model.enable_input_require_grads()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D_in, D_out)) #can experiment with bigger regression part, buuut freezing bert would be needed 
        self.loss = nn.MSELoss()
        self.R2 = R2Score()
        self.MAE = nn.L1Loss()
        self.lr = lr
        self.drop = nn.Dropout(dropout)
        self.total_steps = total_training_steps #param for get_linear_schedule_with_warmup
        self.threshold = accuracy_threshold
        
        # # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def forward(self, input_ids, attention_masks):
        outputs = self.model(input_ids, attention_masks)
        pooled_output = outputs[0].mean(dim=1) #Last layer hidden-state of the first token of the sequence (classification token) (but also can be used for regression)
        output = self.regressor(pooled_output)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        #scheduler is crucial. it deals with fine-tuning instability described in https://arxiv.org/pdf/2006.04884.pdf
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*self.total_steps), num_training_steps=self.total_steps)
        return [optimizer], [scheduler] 
    
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in train_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs)
        loss = self.loss(preds.squeeze(1).float(), 
                        batch_labels.float())
        std = torch.std(preds.squeeze(1).float())
        self.log("train_loss", loss)
        self.log("train_std", std) #for experiments. dont include in official version
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in val_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs)
        val_loss = self.loss(preds.squeeze(1).float(), 
                        batch_labels.float())
        std = torch.std(preds.squeeze(1).float())
        self.log("val_loss", val_loss)
        self.log("val_std", std) #for experiments. dont include in official version
        return val_loss
    
    def test_step(self, test_batch, batch_idx):
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in test_batch)
        outputs = self(batch_inputs, batch_masks)
        preds = torch.tanh(outputs) 
        mae_loss = self.MAE(preds.squeeze(1).float(), 
                        batch_labels.float())
        r2_score = self.R2(preds.squeeze(1).float(), #warning! using batch_size = 8 made the last test step have only one sample in 
                            batch_labels.float())
        std = torch.std(preds.squeeze(1).float())   #preds and batch labels. r2 needs > 1 samples so I increased batch size to 16
        accuracy = torch.sum(torch.abs(batch_labels - preds.squeeze(1)) < self.threshold).item() / len(
            batch_labels
        )
        self.log("test accuracy", accuracy)
        self.log("mae_loss", mae_loss)
        self.log("r2_score", r2_score)
        self.log("test_std", std) #for experiments. dont include in official version
        return mae_loss, r2_score
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_inputs, batch_masks = \
                        tuple(b for b in batch)
        return torch.tanh(self(batch_inputs, batch_masks))
    
if __name__=="__main__":
    #TODO move hyperparamethers to a seperate CONFIG file
    test_size = 0.2
    val_size = 0.5
    batch_size = 64
    epochs = 20

    model_name = 'distilbert-base-uncased'

    text_cleaner = DataPreprocessor()
    df_processor = DFProcessor(filename='ruddit_with_text.csv')

    new_df = df_processor.process_df_BERT(text_cleaner, posts_included=False)

    data_augmentator = TextAugmenterForBert(tokenizer_name=model_name, df = new_df)
    #here we do everything tokenizer-related i.e. encoding corpus, getting input ids etc.
    input_ids, labels, attention_mask = data_augmentator.encode_data(posts_included=False)
    
    # 80% train 10% test 10% val is proposed
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
    # here declare bert-like model
    bertlike_model = DistilBertModel.from_pretrained(model_name, num_labels = 1)
    #bertlike_model.resize_token_embeddings(len(data_augmentator.tokenizer)) #OPTIONAL - IN PRACTISE RUINED RESULTS!!!
    model = Regressor(bertlike_model=bertlike_model, total_training_steps=len(train_dataloader) * epochs, lr=2e-5)
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
            filename="DistilBERT-with-posts-cleaned-dropout-{epoch:02d}-{val_loss:.2f}", 
                
        ),
    ]
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs = epochs, 
        logger = wandb_logger, 
        callbacks = callbacks
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, dataloaders=test_dataloader) #Uncomment to TEST
    wandb.save('checkpoints/*ckpt*') #save checkpoint to wandb 