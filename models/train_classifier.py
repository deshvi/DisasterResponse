# import packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from IPython import display
import collections
from time import time
from itertools import chain
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import seaborn as sns
#import qgrid
import datetime
import sympy as S
import scipy
#
global category_names
import re
import string
import sqlalchemy
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle


def load_data(database):
    # read in file
    engine = create_engine('sqlite:///'+database)
    df = pd.read_sql_table('MLTable',engine)
    X = df['message']
    X.fillna(' ', inplace=True)
    Y = df.drop('message', axis=1)
    Y = Y.drop('original',axis=1)
    Y = Y.drop('genre',axis=1)
    Y.fillna(0, inplace=True)
    Y.drop(['child_alone'], axis = 1, inplace=True) 
    category_names = Y.columns.tolist()


    # clean data


    # load to database


    # define features and label arrays


    return X, Y, category_names

def tokenize(text):
    default_stopwords = set(stopwords.words("english"))  
    default_lemmatizer = WordNetLemmatizer()

    text = re.sub("[^a-zA-Z]"," ", str(text))    
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]
   
    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def lemmatize_text(text, lemmatizer=default_lemmatizer):
        tokens = tokenize_text(text)
        return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]    
        return ' '.join(tokens)
    
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = remove_special_characters(text) # remove punctuation and symbols
    text = lemmatize_text(text) # stemming
    text = remove_stopwords(text) # remove stopwords
    return text

def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    cv = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)



    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return cv

def evaluate_model(X_test, Y_test, model, category_names):
    preds = model.predict(X_test)
    
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test.iloc[:,idx], preds[:,idx]))
    

#def train(X, y, model):
    # train test split


    # fit model


    # output model test results


#    return model


#def export_model(model):
    # Export model as a pickle file



##def run_pipeline(data_file):
##    X, y = load_data(data_file)  # run ETL pipeline
##    model = build_model()  # build model pipeline
##    model = train(X, y, model)  # train model pipeline
##    export_model(model)  # save model

def main():
    """
    Create machine learning models and save output to pickle file
    """
    if len(sys.argv) == 3:
        database, model_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database))
        X, Y, category_names = load_data(database)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(X_test, Y_test, model, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_path))
        model_pkl = open(model_path, 'wb')
        pickle.dump(model, model_pkl)
        model_pkl.close()
#        pickle.dump(model, open(model, 'wb'))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')

 
if __name__ == '__main__':
    main()
    
#if __name__ == '__main__':
#    data_file = sys.argv[1]  # get filename of dataset
#    run_pipeline(data_file)  # run data pipeline