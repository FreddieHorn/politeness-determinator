
from data_processing import DataPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

if __name__ == "__main__":
    #you may need to download that
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('stopwords')
    #nltk.download('punkt')
    data_preprocessor = DataPreprocessor()
    file = "ruddit_with_text.csv"

    dataset = pd.read_csv(file)
    dataset = dataset[dataset['comment_body'] != '[deleted]']#deleting deleted comments from the dataset since they are useless
    dataset = dataset[dataset['post_title'] != '[deleted]']
    dataset = dataset[dataset['post_title'] != '[deleted by user]']

    dataset['title_body'] = dataset['comment_body']+ ' ' + dataset['post_title']
    dataset['clean_text'] = dataset['comment_body'].apply(lambda x: data_preprocessor.process(x))
    dataset['clean_title_body'] = dataset['title_body'].apply(lambda x: data_preprocessor.process(x))
    
    #SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
    X_train, X_test, y_train, y_test = train_test_split(dataset["clean_text"], dataset['offensiveness_score'],test_size=0.2,shuffle=True)

    print(f"Length of the train data: {len(X_train)} posts || test data: {len(X_test)}")