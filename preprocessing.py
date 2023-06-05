import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class DataPreprocessing:
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
        a=[self.wl.lemmatize(tag[0], self.get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
        return " ".join(a)

    def process(self, string):
        return self._lemmatizer(self._stopword(self._preprocess(string)))