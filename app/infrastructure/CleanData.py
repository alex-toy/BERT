import re
import os
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import app.config as cf

from joblib import dump, load


class CleanData :
    """
    cleans text data from raw file
    """

    def __init__(self, path='', cols=[], cols_to_keep=[]) :
        self.path = path
        self.cols = cols
        self.cols_to_keep = cols_to_keep



    def get_df_from_path(self) :
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            df = pd.read_csv(self.path, names=self.cols)
        elif extension == '.parquet':
            df = pd.read_parquet(self.path, names=self.cols)
        else:
            raise FileExistsError('Extension must be parquet or csv.')

        return df[self.cols_to_keep]



    def clean_tweet(self, tweet):
        tweet = BeautifulSoup(tweet, "lxml").get_text()
        tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
        tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
        tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet



    def get_cleaned_df(self):
        df = self.get_df_from_path()
        df['text'] = df['text'].apply(lambda tweet : self.clean_tweet(tweet))
        return df



    def get_data_clean(self) :
        data_clean = self.get_cleaned_df()['text'].values
        return data_clean


    def get_data_labels(self) :
        data_labels = self.get_cleaned_df()['sentiment'].values
        return data_labels



if __name__ == "__main__":

    cd = CleanData(
        path='/Users/alexei/BERT/data/testdata.manual.2009.06.14.csv', 
        cols=["sentiment", "id", "date", "query", "user", "text"],
        cols_to_keep=["sentiment", "text"]
    )

    data_labels = cd.get_data_labels()

    print(data_labels)


