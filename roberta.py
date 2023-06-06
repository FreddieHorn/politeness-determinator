from transformers import RobertaTokenizer, RobertaModel
from data_processing import DataPreprocessor, tokenize_function
from sklearn.model_selection import train_test_split
import torch.nn as nn
#from preprocessing import preprocessing_pipeline
import pandas as pd
import nltk
import numpy as np

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
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download()
    # nltk.download('wordnet')
    # print(nltk.find('corpora/wordnet.zip'))
    model_name = 'roberta-base'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    data_preprocessor = DataPreprocessor()
    file = "ruddit_with_text.csv"

    dataset = pd.read_csv(file)
    dataset = dataset[dataset['comment_body'] != '[deleted]']#deleting deleted comments from the dataset since they are useless
    dataset['clean_text'] = dataset['comment_body'].apply(lambda x: data_preprocessor.process(x))
    new_df = dataset.filter(items=['clean_text', 'offensiveness_score'])
    #tokenizer(examples["clean_text"], padding="max_length", truncation=True)
    encoded_corpus = tokenizer(text = new_df.clean_text.to_list(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            return_attention_mask=True)
    
    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    input_ids = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    input_ids = np.array(input_ids)
    attention_mask = np.array(attention_mask)
    labels = new_df.offensiveness_score.to_numpy()
    print(tokenizer.decode(encoded_corpus["input_ids"][0]))

    test_size = 0.1
    seed = 42
    train_inputs, test_inputs, train_labels, test_labels = \
            train_test_split(input_ids, labels, test_size=test_size, 
                             random_state=seed)
    train_masks, test_masks, _, _ = train_test_split(attention_mask, 
                                            labels, test_size=test_size, 
                                            random_state=seed)