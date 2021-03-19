from preprocessing import get_tokens_lowercase_spellcheck_remove_stopwords
from sequences_fasttext import to_sequence
from load import _encoder_lvl2, _model
from helpers import *
from config import *


def predict(text):
    title = get_tokens_lowercase_spellcheck_remove_stopwords(text['title'])
    description = get_tokens_lowercase_spellcheck_remove_stopwords(text['description'])
    title_sequence = to_sequence([title], MAX_SEQUENCE_LENGTH_TITLES)
    description_sequence = to_sequence([description], MAX_SEQUENCE_LENGTH)
    classes = _encoder_lvl2.classes_
    predicted = _model.predict([description_sequence, title_sequence])
    predicted = get_prediction_with_precision(classes, predicted, 3, True)
    return predicted[0]
