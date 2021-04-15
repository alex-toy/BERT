import app.config as cf
import os
from app.infrastructure.CleanData import CleanData
from app.model.CustomSchedule import CustomSchedule
from app.model.Transformer import Transformer
from app.application.train_model import train_model
from app.application.pickling import to_pickle, from_pickle

import tensorflow as tf
from joblib import dump, load



def train_model_step_1() :

    cd = CleanData(
        input_data_path=cf.europarl_en_path, 
        input_nb_prefix_path=cf.nb_prefix_en_path, 
        output_data_path=cf.europarl_es_path, 
        output_nb_prefix_path=cf.nb_prefix_es_path,
        MAX_LENGTH=cf.MAX_LENGTH,
        BATCH_SIZE=cf.BATCH_SIZE,
        BUFFER_SIZE=cf.BUFFER_SIZE
    )

    dataset = cd.get_dataset()

    transformer = Transformer(
        vocab_size_enc=cd.INPUT_VOCAB_SIZE,
        vocab_size_dec=cd.OUTPUT_VOCAB_SIZE,
        d_model=cf.D_MODEL,
        nb_layers=cf.NB_LAYERS,
        FFN_units=cf.FFN_UNITS,
        nb_proj=cf.NB_PROJ,
        dropout_rate=cf.DROPOUT_RATE
    )

    leaning_rate = CustomSchedule(cf.D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        leaning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    ckpt = tf.train.Checkpoint(
        transformer=transformer,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, cf.CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored!!")

    to_pickle('dataset', dataset)
    to_pickle('ckpt_manager', ckpt_manager)
    to_pickle('transformer', transformer)
    to_pickle('optimizer', optimizer)


    return dataset, ckpt_manager, transformer, optimizer




if __name__ == "__main__":
    
    dataset, ckpt_manager, transformer, optimizer = train_model_step_1()