import datetime
import itertools
import re
import string

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset


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
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
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
    def __init__(self) -> None:
        # Initialize the lemmatizer
        self.wl = WordNetLemmatizer()

    def _preprocess(self, text):
        text = text.lower()
        text = text.strip()
        text = re.compile("<.*?>").sub("", text)
        text = re.compile("[%s]" % re.escape(string.punctuation)).sub(" ", text)
        text = re.sub("\s+", " ", text)
        text = re.sub(r"\[[0-9]*\]", " ", text)
        text = re.sub(r"[^\w\s]", "", str(text).lower().strip())
        text = re.sub(r"\d", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

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


class DFProcessor:
    def __init__(self, filename) -> None:
        self.filename = filename

    def process_df(self, text_cleaner):
        dataset = pd.read_csv(self.filename)
        dataset = dataset[dataset['comment_body'] != '[deleted]'] #deleting deleted comments from the dataset since they are useless
        #dataset['comment_body'] = dataset['comment_body'].apply(lambda x: text_cleaner.process(x))
        new_df = dataset.filter(items=['comment_body', 'offensiveness_score'])
        print(new_df.dtypes)
        return new_df


if __name__ == "__main__":
    pass
