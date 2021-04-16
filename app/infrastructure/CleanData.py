import re
import os
import bert
from bert.tokenization import bert_tokenization
from bs4 import BeautifulSoup
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
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



    def get_tokenizer(self):
        #FullTokenizer = bert.tokenization.FullTokenizer
        FullTokenizer = bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=False
        )
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)
        
        def encode_sentence(sent):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

        return encode_sentence



    def get_data_inputs(self) :
        data_clean = self.get_data_clean()
        encode_sentence = self.get_tokenizer()
        return [encode_sentence(sentence) for sentence in data_clean]






if __name__ == "__main__":

    cd = CleanData(
        path=cf.INPUTS_FILE, 
        cols=["sentiment", "id", "date", "query", "user", "text"],
        cols_to_keep=["sentiment", "text"]
    )

    data_inputs = cd.get_data_inputs()

    print(data_inputs)


