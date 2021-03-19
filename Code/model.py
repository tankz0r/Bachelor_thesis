from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from config import *
from load import _embedding_matrix


def create_model(sequence_length_descr=MAX_SEQUENCE_LENGTH,
                 sequence_length_titles=MAX_SEQUENCE_LENGTH_TITLES,
                 vocabulary_size=_embedding_matrix.shape[0],
                 embedding_dim=_embedding_matrix.shape[-1]):
    inputs_descr_1 = Input(shape=(sequence_length_descr,), dtype='int32')
    inputs_titles_1 = Input(shape=(sequence_length_titles,), dtype='int32')

    embedding_descr = Embedding(
        vocabulary_size,
        embedding_dim,
        weights=[_embedding_matrix],
        input_length=sequence_length_descr,
        trainable=False,
    )

    embedding_titles = Embedding(
        vocabulary_size,
        embedding_dim,
        weights=[_embedding_matrix],
        input_length=sequence_length_titles,
        trainable=False,
    )

    def conv_part(embedding_1, sequence_length):
        reshape_1 = Reshape((sequence_length, embedding_dim, 1))(embedding_1)

        conv_1_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)
        conv_1_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)
        conv_1_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)

        maxpool_1_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_0)
        maxpool_1_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_1)
        maxpool_1_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_2)

        merged_tensor_1 = merge([maxpool_1_0, maxpool_1_1, maxpool_1_2], mode='concat', concat_axis=1)
        flatten_1 = Flatten()(merged_tensor_1)
        return flatten_1

    embedding_descr_1 = embedding_descr(inputs_descr_1)
    embedding_titles_1 = embedding_titles(inputs_titles_1)

    flatten_descr_2 = conv_part(embedding_descr_1, sequence_length_descr)
    flatten_titles_2 = conv_part(embedding_titles_1, sequence_length_titles)
    flatten_2 = concatenate([flatten_descr_2, flatten_titles_2])

    dropout_2 = Dropout(drop)(flatten_2)

    output_2 = Dense(output_dim=output_2_dim, activation='softmax')(dropout_2)

    model_2 = Model(input=[inputs_descr_1, inputs_titles_1], output=output_2)

    return model_2