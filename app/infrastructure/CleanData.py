import re
import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import app.config as cf

from joblib import dump, load


class CleanData :
    """
    cleans text data from raw file
    """

    def __init__(self, path='', cols=[]) :
        self.path = path
        self.cols = cols



    def get_df_from_path(self) :
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            df = pd.read_csv(self.path)
        elif extension == '.parquet':
            df = pd.read_parquet(self.path)
        else:
            raise FileExistsError('Extension must be parquet or csv.')

        return df



    def get_data(self):
        with open(self.input_data_path, mode = "r", encoding = "utf-8") as f:
            input_corpus = f.read()







if __name__ == "__main__":

    cd = CleanData(path='', cols=[])

    dataset = cd.get_dataset()

    print(dataset)


    #cd.path_to_csv()


