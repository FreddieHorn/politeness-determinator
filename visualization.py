import torch 
import matplotlib.pyplot as plt
from transformers import DistilBertModel
import lightning as L
import numpy as np

from distilBERT import Regressor, TextAugmenterForBert
from data_processing import DataPreprocessor, DFProcessor, create_dataloaders_testing, tensors_to_numpy

batch_size = 64

model_name = 'distilbert-base-uncased'

text_cleaner = DataPreprocessor()
df_processor = DFProcessor(filename='ruddit_with_text.csv')

new_df = df_processor.process_df_BERT(text_cleaner, posts_included=False)

data_augmentator = TextAugmenterForBert(tokenizer_name=model_name, df = new_df)
#here we do everything tokenizer-related i.e. encoding corpus, getting input ids etc.
input_ids, labels, attention_mask = data_augmentator.encode_data(posts_included=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inds = labels.argsort()
sortedLabels = labels[inds]
sortedInputids = input_ids[inds]
sortedAttentionmask = attention_mask[inds]
# sortedInputids = sortedInputids.to(device)
# sortedAttentionmask = sortedAttentionmask.to(device)
testing_dataloader = create_dataloaders_testing(sortedInputids, sortedAttentionmask, batch_size=batch_size)

bertlike_model = DistilBertModel.from_pretrained(model_name, num_labels = 1)
model = Regressor.load_from_checkpoint("checkpoints/DistilBERT-with-posts-cleaned-dropout-epoch=19-val_loss=0.04.ckpt",
                                        bertlike_model=bertlike_model, lr=2e-5)
trainer = L.Trainer(
        accelerator="gpu"
    )
# model = model.to(device)
outputs = trainer.predict(model, dataloaders=testing_dataloader)

outputs = tensors_to_numpy(outputs)

sortedInputindex = np.arange(0, sortedInputids.shape[0], 1)
plt.scatter(sortedInputindex, sortedLabels, label = "true labels")
plt.scatter(sortedInputindex, outputs, label = "model predictions")
plt.xlabel("Input sentence id")
plt.ylabel("Rudeness level")
plt.title("True vs predicted values of the rudeness level")
plt.show()