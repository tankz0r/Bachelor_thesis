from pygoose import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from load import _tokenizer, _embedding_model
from config import *


def to_sequence(tokens, seq_len):
    texts = join_pairs(tokens)
    sequences = _tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequences, maxlen=seq_len)
    return sequence_pad


def create_embedding_matrix():
    num_words = min(MAX_VOCAB_SIZE, len(_tokenizer.word_index))
    embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
    for word, index in progressbar(_tokenizer.word_index.items()):
        if word in _embedding_model.vocab:
            embedding_matrix[index] = _embedding_model[word]
    kg.io.save(embedding_matrix, project.aux_dir + 'fasttext_vocab_embedding_matrix.pickle')


def join_pairs(tokens):
    return [' '.join(pair) for pair in tokens]


def sequence_for_train(field, seq_len):
    tokens_train = kg.io.load(
        project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords_%s_train.pickle' % field)
    tokens_test = kg.io.load(
        project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords_%s_test.pickle' % field)
    sequences_train = to_sequence(tokens_train, seq_len)
    sequences_test = to_sequence(tokens_test, seq_len)
    kg.io.save(sequences_train, project.preprocessed_data_dir + 'sequences_fasttext_%s_train.pickle' % field)
    kg.io.save(sequences_test, project.preprocessed_data_dir + 'sequences_fasttext_%s_test.pickle' % field)

