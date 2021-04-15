import os
import app.config as cf
from joblib import dump, load

import tensorflow as tf

from app.model.Transformer import Transformer

def evaluate(inp_sentence):
    INPUT_VOCAB_SIZE = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'INPUT_VOCAB_SIZE.joblib'))
    OUTPUT_VOCAB_SIZE = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'OUTPUT_VOCAB_SIZE.joblib'))

    input_tokenizer = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'input_tokenizer.joblib'))


    inp_sentence = [INPUT_VOCAB_SIZE-2] + input_tokenizer.encode(inp_sentence) + [INPUT_VOCAB_SIZE-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
    
    output = tf.expand_dims([OUTPUT_VOCAB_SIZE-2], axis=0)

    transformer = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'transformer.joblib'))
    
    for _ in range(cf.MAX_LENGTH):
        predictions = transformer(enc_input, output, False)
        
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == OUTPUT_VOCAB_SIZE-1:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)

