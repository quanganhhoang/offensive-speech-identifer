import streamlit as st

import nltk

from nltk import sent_tokenize, word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import stopwords

from collections import defaultdict

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier

import textstat
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from joblib import dump, load


def class_to_name(class_label):
    """
    This function is used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Offensive speech"
    elif class_label == 1:
        return "Not offensive speech"
    else:
        return "No label"


@st.cache(allow_output_mutation=True)
def load_model():
    """
    This function loads the hate speech ML model.
    """
    return load('model.joblib')


def get_prediction(message: str):
    """
    Runs the model on user input and returns prediction
    """
    pred = model.predict([message])

    return class_to_name(pred[0])


@st.cache
def generate_stopwords():
    """
    Generate stopwords for NLP
    """
    all_stopwords = stopwords.words('english')
    excludes = ['rt', '&#57361;']
    all_stopwords.extend(excludes)

    return all_stopwords


def analyzer(doc: str):
    all_stopwords = generate_stopwords()
    stemmer = PorterStemmer()
    words = word_tokenize(doc)
    filtered_words = [word for word in words if not word in all_stopwords and word.isalnum()]
    
    return [stemmer.stem(word) for word in filtered_words if word not in ['URLHERE', 'MENTIONHERE']]


def preprocess(text: str):
    """
    Preprocess text before feeding to model
    """
    space_pattern = '\s+'
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    symbol_regex = '&#[^\s]+'
    mention_regex = '@[^\s]+'
    
    parsed_text = text.lower()
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = re.sub(symbol_regex, ' ', parsed_text)
    parsed_text = re.sub(url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)

#     words = word_tokenize(parsed_tweet)
    
#     filtered_words = [word for word in words if not word in all_stopwords and word.isalnum()]
#     porter = PorterStemmer()
#     stemmed = [porter.stem(word) for word in filtered_words if word not in ['URLHERE', 'MENTIONHERE']]
    
#     pos = pos_tag(filtered_words)
    
    return parsed_text


def tokenize(tweet):
    stemmer = PorterStemmer()
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def getSentiment(df):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    scores = defaultdict(list)
    for i in range(len(df)):
        score_dict = sentiment_analyzer.polarity_scores(df[i])
        scores['neg'].append(score_dict['neg'])
        scores['neu'].append(score_dict['neu'])
        scores['pos'].append(score_dict['pos'])
        scores['compound'].append(score_dict['compound'])
    return np.array(pd.DataFrame(scores))


model = load_model()

st.title('Offensive Speech Identifier')
user_input = st.text_input("Enter input here: ", value="Hi, this is not an offensive speech")
st.write('Processing message: ', user_input)
st.write('Result:', get_prediction(user_input))
