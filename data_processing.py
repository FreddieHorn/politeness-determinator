import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader

def tokenize_function(examples, tokenizer):
    
    """ This function tokenizes the text in the examples dictionary.
        We pass it to the map function of the dataset so that we can batch the tokenization for efficiency by
        tokenizing batches in parallel.
    """
    return tokenizer(examples["clean_text"], padding="max_length", truncation=True)

class DataPreprocessor:
    def __init__(self) -> None:
        # Initialize the lemmatizer
        self.wl = WordNetLemmatizer()
    def _preprocess(self, text):
        text = text.lower() 
        text = text.strip()  
        text = re.compile('<.*?>').sub('', text) 
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
        text = re.sub('\s+', ' ', text)  
        text = re.sub(r'\[[0-9]*\]',' ',text) 
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d',' ',text) 
        text = re.sub(r'\s+',' ',text) 
        return text
    def _stopword(self, string):
        a= [i for i in string.split() if i not in stopwords.words('english')]
        return ' '.join(a)
    
    # This is a helper function to map NTLK position tags
    def _get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    # Tokenize the sentence
    def _lemmatizer(self, string):
        word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
        a=[self.wl.lemmatize(tag[0], self._get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
        return " ".join(a)

    def process(self, string):
        return self._lemmatizer(self._stopword(self._preprocess(string)))

class DFProcessor:
    def __init__(self, filename) -> None:
        self.filename = filename

    def process_df(self, text_cleaner):
        dataset = pd.read_csv(self.filename)
        dataset = dataset[dataset['comment_body'] != '[deleted]']#deleting deleted comments from the dataset since they are useless
        dataset['clean_text'] = dataset['comment_body'].apply(lambda x: text_cleaner.process(x))
        new_df = dataset.filter(items=['clean_text', 'offensiveness_score'])
        return new_df

if __name__ == "__main__":
    pass