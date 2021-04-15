import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

europarl_en_path = "data/es_en/europarl-v7.es-en.en"
europarl_es_path = "data/es_en/europarl-v7.es-en.es"
#europarl_en_path = "data/es_en_test/europarl-v7.es-en.en"
#europarl_es_path = "data/es_en_test/europarl-v7.es-en.es"
nb_prefix_en_path = "data/nonbreaking_prefix.en"
nb_prefix_es_path = "data/nonbreaking_prefix.es"

OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
INPUTS_FILE = os.path.join(OUTPUTS_DIR, 'inputs.csv')
OUTPUTS_FILE = os.path.join(OUTPUTS_DIR, 'outputs.csv')

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoint'))

OUTPUTS_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))

MAX_LENGTH=20
BATCH_SIZE = 64
BUFFER_SIZE = 20000
D_MODEL = 128 # 512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT_RATE = 0.1 # 0.1
EPOCHS = 1