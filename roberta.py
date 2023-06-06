from transformers import RobertaTokenizer, RobertaModel
from data_processing import DataPreprocessor, tokenize_function
from sklearn.model_selection import train_test_split
import torch.nn as nn
#from preprocessing import preprocessing_pipeline
import pandas as pd 

class RobertaRegressor(nn.Module):
    def __init__(self, dropout=0.2, model_name = 'roberta-base'):
        super(RobertaRegressor, self).__init__()
        D_in, D_out = 768, 1
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D_in, D_out))
        
    def forward(self, input_ids, attention_masks):
        
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs
    
if __name__=="__main__":
    
    model_name = 'roberta-base'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    data_preprocessor = DataPreprocessor()
    file = "ruddit_with_text.csv"

    dataset = pd.read_csv(file)
    dataset = dataset[dataset['comment_body'] != '[deleted]']#deleting deleted comments from the dataset since they are useless
    dataset['clean_text'] = dataset['comment_body'].apply(lambda x: data_preprocessor.process(x))
    new_df = dataset.filter(items=['clean_text', 'offensiveness_score'])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    print(new_df.head(10))