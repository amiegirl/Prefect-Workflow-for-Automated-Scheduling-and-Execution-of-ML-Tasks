import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from prefect import task, flow

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)


@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y
	

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
	
def clean(raw_text):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub("[^a-zA-Z]|READ MORE", " ", raw_text)
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(cleaned_tokens)

@task
def preprocess(X_train, X_test, y_train, y_test):
    """
    cleaning the text data before hand.
    """
    vect = CountVectorizer(preprocessor=clean, max_features=5000)
    X_train_bow = vect.fit_transform(X_train)
    X_test_bow = vect.transform(X_test)
    return X_train_bow, X_test_bow, y_train, y_test
	

@task
def train_model(X_train_bow, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    model = LogisticRegression(**hyperparameters)
    model.fit(X_train_bow, y_train)
    return model
	

@task
def evaluate_model(model, X_train_bow, y_train, X_test_bow, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train_bow)
    y_test_pred = model.predict(X_test_bow)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score


# Workflow
@flow(name="Logistic Regression Training Flow")
def workflow():
    DATA_PATH = 'product_reviews.csv'
    INPUT = 'review_text'
    OUTPUT = 'sentiment'
    HYPERPARAMETERS = {'max_iter':1000,
                       'C':1,
                       'class_weight':'balanced',
                       'l1_ratio':0.6,
                       'penalty':'elasticnet',
                       'solver':'saga'}
    # Load data
    review = load_data(DATA_PATH)

    # Identify Inputs and Output
    X, y = split_inputs_output(review, INPUT, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocess the data
    X_train_bow, X_test_bow, y_train, y_test = preprocess(X_train, X_test, y_train, y_test)

    # Build a model
    model = train_model(X_train_bow, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train_bow, y_train, X_test_bow, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)

if __name__ == "__main__":
    workflow.serve(name="logistic-regression-deployment",
                   cron="0 12 * * *")