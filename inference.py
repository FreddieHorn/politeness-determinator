from distilBERT import TextAugmenterForBert, Regressor
from data_processing import DataPreprocessor

from transformers import AutoTokenizer, DistilBertModel

import torch
import numpy as np

if __name__ == "__main__":

    model_name = 'distilbert-base-uncased'
    checkpoint_path = "checkpoints/DistilBERT-NAT-NP-WITH_SCHEDULER_RUN2_NO_FREEZE-epoch=19-val_loss=0.03.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    distilbert = DistilBertModel.from_pretrained(model_name, num_labels = 1)
    processor = DataPreprocessor()

    model = Regressor.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=torch.device(device), bertlike_model = distilbert)
    model.to(device)
    model.eval()

    text = "I love going on long walks"

    cleaned_text = processor.processBERT(text)

    encoded_text = tokenizer(cleaned_text,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=200,
                    return_attention_mask=True,
                    return_tensors="pt")
    
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    with torch.no_grad():
        prediction = model(input_ids, attention_mask)

    print(prediction)
    