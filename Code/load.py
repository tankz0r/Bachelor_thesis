from gensim.models.wrappers.fasttext import FastText
import helpers as helpers
from pygoose import kg
from config import *
from keras.models import model_from_json
from keras.optimizers import Adam


_embedding_model = None
_tokenizer = None
_embedding_matrix = None
_encoder_lvl2 = None
_model = None


def get_embedding_matrix():
    global _embedding_matrix
    if not _embedding_matrix:
        try:
            _embedding_matrix = kg.io.load(project.aux_dir + EMBEDDING_MATRIX)
        except IOError:
            _embedding_matrix = None
    print("get_embedding_matrix")
    return _embedding_matrix


def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        try:
            _tokenizer = kg.io.load(TOKENIZER)
        except IOError:
            _tokenizer = None
    return _tokenizer


def get_embedding_model():
    global _embedding_model
    if not _embedding_model:
        try:
            _embedding_model = FastText.load_word2vec_format(project.aux_dir + FASTTEXT_VOCAB)
        except IOError:
            _embedding_model = None
    return _embedding_model


def get_encoder_lvl2():
    global _encoder_lvl2
    if not _encoder_lvl2:
        try:
            _encoder_lvl2 = kg.io.load(ENCODER_LVL2)
        except IOError:
            _encoder_lvl2 = None
    return _encoder_lvl2


def save_model(model, model_path=MODEL, weights_path=WEIGHTS):
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    weights = model.get_weights()
    helpers.save_file(weights, weights_path)


def load_model(model_path=MODEL, weights_path=WEIGHTS):
    global _model
    print("load model")
    if _model is None:
        try:
            json_file = open(model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            _model = model_from_json(loaded_model_json)
            weights = helpers.get_file(weights_path)
            _model.set_weights(weights)
            _model.compile(optimizer="adam",
                            loss='categorical_crossentropy',
                            metrics=['top_k_categorical_accuracy'])
            print("Loaded model from disk")
        except IOError:
            _model = None
    return _model


get_embedding_matrix()
