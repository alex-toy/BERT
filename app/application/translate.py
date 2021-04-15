import app.config as cf
from app.application.evaluate import evaluate
from joblib import dump, load
import os


def translate(sentence):
    output = evaluate(sentence).numpy()
    
    output_tokenizer = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'output_tokenizer.joblib'))
    OUTPUT_VOCAB_SIZE = load(os.path.join(cf.OUTPUTS_MODELS_DIR, 'OUTPUT_VOCAB_SIZE.joblib')) 
    
    predicted_sentence = output_tokenizer.decode(
        [i for i in output if i < OUTPUT_VOCAB_SIZE-2]
    )
    
    return predicted_sentence
    
