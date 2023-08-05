import datetime
import itertools
import re
import string
import numpy as np

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset

PUNCT_TO_REMOVE = string.punctuation

def tensors_to_numpy(array_of_tensors):
    # Check if the input is a list or tuple (array) of tensors
    if not isinstance(array_of_tensors, (list, tuple)):
        raise ValueError("Input should be a list or tuple of tensors.")
    
    # Get the total number of elements in all tensors combined
    total_elements = sum(np.prod(tensor.shape) for tensor in array_of_tensors)

    # Create a numpy array to hold all elements from the tensors
    numpy_array = np.zeros(total_elements)

    # Flatten and concatenate each tensor into the numpy_array
    index = 0
    for tensor in array_of_tensors:
        tensor = tensor.detach().to('cpu').numpy()
        numpy_array[index:index + np.prod(tensor.shape)] = tensor.flatten()
        index += np.prod(tensor.shape)

    return numpy_array

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs, dtype=torch.int32)
    mask_tensor = torch.tensor(masks, dtype=torch.int32)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=5)
    return dataloader


def tokenize_function(examples, tokenizer):
    """This function tokenizes the text in the examples dictionary.
    We pass it to the map function of the dataset so that we can batch the tokenization for efficiency by
    tokenizing batches in parallel.
    """  # noqa: E501
    return tokenizer(examples["clean_text"], padding="max_length", truncation=True)


def tokenize_words(dataframe, column):
    dataframe["tokenized_sentences"] = dataframe[column].apply(nltk.word_tokenize)
    token_list = list(itertools.chain(*dataframe["tokenized_sentences"].tolist()))
    return token_list


class DataPreprocessor:
    """Class that is responsible for cleaning text.
    """
    def __init__(self) -> None:
        # Initialize the lemmatizer
        self.wl = WordNetLemmatizer()

    def _preprocess(self, text: str):
        text = text.lower()
        text = text.strip()
        text = text.translate(str.maketrans("", "", PUNCT_TO_REMOVE))
        text = re.compile("<.*?>").sub("", text)
        text = re.sub(r"\[[0-9]*\]", " ", text)
        text = re.sub(r"[^\w\s]", "", str(text).lower().strip())
        text = re.sub(r"\d", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def _preprocessBERT(self, text):
        # Remove any punctuation
        cleaned_text = re.sub(r'[^\w\s]', '', text)

        # Remove strings that only consist of punctuation (except words starting with two hashtags)
        cleaned_text = re.sub(r'(?<!#)\b[^\w\s]+\b', '', cleaned_text)

        return cleaned_text
    
    def _stopword(self, string):
        a = [i for i in string.split() if i not in stopwords.words("english")]
        return " ".join(a)

    # This is a helper function to map NTLK position tags
    def _get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # Tokenize the sentence

    def _lemmatizer(self, string):
        word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
        a = [
            self.wl.lemmatize(tag[0], self._get_wordnet_pos(tag[1]))
            for idx, tag in enumerate(word_pos_tags)
        ]  # Map the position tag and lemmatize the word/token
        return " ".join(a)

    def process(self, string):
        return self._lemmatizer(self._stopword(self._preprocess(string)))
    
    def processBERT(self, string):
        return self._lemmatizer(self._stopword(self._preprocessBERT(string)))


def check(df):
    punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    for item in df["clean_text"]:
        match = re.search(punctuation_pattern, item)
        if bool(match):
            print(item.index())


class DFProcessor:
    """Handles processing of a given column of the dataframe.

    Args:
        filename (str): filename
    """
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def process_df(self, text_cleaner=None):
        dataset = pd.read_csv(self.filename)
        dataset = dataset[dataset["comment_body"] != "[deleted]"]
        # deleting deleted comments from the dataset since they are useless
        # dataset['comment_body'] = dataset['comment_body'].apply(lambda x: text_cleaner.process(x))  # noqa: E501

        dataset = dataset[dataset["post_title"] != "[deleted]"]
        dataset = dataset[dataset["post_title"] != "[deleted by user]"]

        dataset["title_body"] = dataset["comment_body"] + " " + dataset["post_title"]
        new_df = dataset.filter(items=["title_body", "offensiveness_score"])
       return new_df
    
    def process_df_BERT(self, text_cleaner, post_included = True):
        dataset = pd.read_csv(self.filename)
        dataset = dataset[dataset['comment_body'] != '[deleted]'] #deleting deleted comments from the dataset since they are useless
        dataset['comment_body'] = dataset['comment_body'].apply(lambda x: text_cleaner.processBERT(x)) #in practise it did not help much
        if post_included:
            dataset = dataset[dataset['post_title'] != '[deleted]']
            dataset = dataset[dataset['post_title'] != '[deleted by user]']
            dataset['post_title'] = dataset['post_title'].apply(lambda x: text_cleaner.processBERT(x))
            new_df = dataset.filter(items=['post_title','comment_body', 'offensiveness_score'])
        else:
            new_df = dataset.filter(items=['comment_body', 'offensiveness_score'])

        return new_df

if __name__ == "__main__":
    pass
