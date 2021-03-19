from pygoose import *
from helpers import get_rec_sys_dir

project = kg.Project(get_rec_sys_dir())

EMBEDDING_DIM = 300
FASTTEXT_EXECUTABLE = 'fasttext'
PRETRAINED_MODEL_FILE = '/home/denys/kaggle-quora-question-pairs/data/aux/fasttext/wiki.en.bin'
VOCAB_FILE = project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords.vocab'
OUTPUT_FILE = project.aux_dir + 'fasttext_vocab.vec'
EMBEDDING_MATRIX = 'fasttext_vocab_embedding_matrix_v2.pickle'

MAX_VOCAB_SIZE = 226268
MAX_SEQUENCE_LENGTH = 30
MAX_SEQUENCE_LENGTH_TITLES = 15

TOKENIZER = './models/tokenizer'
TOKENIZER_DESCR = './models/tokenizer_descr'
TOKENIZER_TITLES = './models/tokenizer_titles'
ENCODER_LVL2 = './models/encoder_lvl2'
ENCODER_LVL1 = './models/encoder_lvl1'
FASTTEXT_VOCAB = 'fasttext_vocab.vec'

RANDOM_SEED = 42
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
nb_epoch = 5
batch_size = 30
output_1_dim = 14
output_2_dim = 166

MODEL = get_rec_sys_dir() + '/CNN_lvl2_model/model_final_v2.json'
WEIGHTS = 'CNN_lvl2_model/weights_final_v2'


log_dir = './keras_logs_cnn_lstm'
model_dir = './models_cnn_lstm'
train_log_dir = '%s/train' % log_dir
val_log_dir = '%s/val' % log_dir
