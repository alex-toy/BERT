import app.config as cf
import os
from app.infrastructure.CleanData import CleanData
from app.model.CustomSchedule import CustomSchedule
from app.model.Transformer import Transformer
from app.application.train_model import train_model
from app.application.pickling import to_pickle, from_pickle

import tensorflow as tf
from joblib import dump, load


def train_model_step_2(dataset, ckpt_manager, transformer, optimizer) :

    dataset = from_pickle('dataset')
    ckpt_manager = from_pickle('ckpt_manager')
    transformer = from_pickle('transformer')
    optimizer = from_pickle('optimizer')
    
    transformer = train_model(dataset, ckpt_manager, transformer, optimizer)
    dump(transformer, os.path.join(cf.OUTPUTS_MODELS_DIR, 'transformer.joblib'))




if __name__ == "__main__":
    
    dataset, ckpt_manager, transformer, optimizer = None, None, None, None
    train_model_step_2(dataset, ckpt_manager, transformer, optimizer)