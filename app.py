import streamlit as st

import nltk

from nltk import sent_tokenize, word_tokenize, PorterStemmer, pos_tag
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from joblib import dump, load

st.title('Hate Speech Identifier')


def class_to_name(class_label):
    """
    This function is used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"


@st.cache
def load_model():
    """
    This function loads the hate speech ML model.
    """

    return load('model.joblib')


def get_prediction(message: str):
    """
    Runs the model on user input and returns prediction
    """
    model = load_model()
    pred = model.predict([message])

    return class_to_name(pred[0])


@st.cache
def generate_stopwords():
    """
    Generate stopwords for NLP
    """
    all_stopwords = stopwords.words('english')
    excludes = ['@user', '@', '!', 'RT']
    all_stopwords.extend(excludes)

    return all_stopwords


def preprocess(text: str):
    """
    Preprocess text before feeding to model
    """
    space_pattern = '\s+'
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
#     mention_regex = '@[\w\-]+'
    mention_regex = '@[^\s]+'
    
    parsed_text = text.lower()
    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    parsed_text = re.sub(url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)

#     words = word_tokenize(parsed_tweet)
    
#     filtered_words = [word for word in words if not word in all_stopwords and word.isalnum()]
#     porter = PorterStemmer()
#     stemmed = [porter.stem(word) for word in filtered_words if word not in ['URLHERE', 'MENTIONHERE']]
    
#     pos = pos_tag(filtered_words)
    
    return parsed_text


def stem_words(text: str):
    """
    Lemmatize words
    """
    all_stopwords = generate_stopwords()
    words = word_tokenize(text)
    filtered_words = [word for word in words if not word in all_stopwords and word.isalnum()]
    porter = PorterStemmer()

    return [porter.stem(word) for word in filtered_words if word not in ['URLHERE', 'MENTIONHERE']]


user_input = st.text_input("Enter input here: ", value="Hi, this is not a hate speech")
st.write('Processing message: ', user_input)
st.write('Result:', get_prediction(user_input))