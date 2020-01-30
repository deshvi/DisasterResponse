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
import datetime
import sympy as S
import scipy
import sqlalchemy
from sqlalchemy import create_engine

# Load messages and categories
def load_data(messagesFile, categoriesFile):
    """
    Load data into the dataframes
    Input: messagesFile - String
            File Name of Messages csv
            categoriesFile - String
            File Name of Categories csv
    Output: df = DataFrame
            Merged DataFrame of message and categories data
    """
    
    # load messages & categories
    messages = pd.read_csv(messagesFile)
    categories = pd.read_csv(categoriesFile)
    # Merge dataframes
    df = pd.merge(messages, categories, how='left', on=['id'])

    return df


def clean_data(df):
    """
    Extracts categories and flags from categories data, remove duplicates
    Input: df - DataFrame
            Dataframe output from load_data function
    Output: df - DataFrame
            Cleansed dataframe of the input data
    """
    # Get the categories column from the DF, then find the strings attached
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.slice(0,-2,1)
    categories.columns = category_colnames
    #
    for column in categories:
        # set each value upto the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    #
    df.drop(columns='categories',inplace=True)
    #
    df = df.join(categories,how='left')
    #
    #
    df = df.drop_duplicates(keep='last')

    return df


def save_data(df, database):
    """
    Save data to database
    Input: df - DataFrame
            DataFrame from clean_data dataframe
           database_filename - String
           Database file location of where data is to be stored
           The name of the table is MLTable 
    """
    #
    engine = create_engine('sqlite:///'+database)
    #
    from pandas.io import sql
    sql.execute('DROP TABLE IF EXISTS MLTable', engine)
    #
    df.to_sql('MLTable', engine, index=False)
    
    print("Data was saved to {} in the {} table".format(database, 'MLTable'))


def main():
    """
    Run ETL of messages and categories data
    """
    if len(sys.argv) == 4:

        messagesFile, categoriesFile, database = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messagesFile, categoriesFile))
        df = load_data(messagesFile, categoriesFile)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database))
        save_data(df, database)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively and '\
              'database to save the cleaned data as the third argument.'\
              '\n\nExample: python data/process_data.py data/disaster_messages.csv '\
              'data/disaster_categories.csv '\
              'data/DisasterResponse.db ')

if __name__ == '__main__':
    main()