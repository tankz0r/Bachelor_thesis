from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import load_model
from keras.utils.vis_utils import  plot_model
from keras.callbacks import TensorBoard
from utils import find_last_checkpoint
from custom_metrics import batch_pairwise_metrics
from custom_callbacks import FilterTensorBoard, BatchTimer
from load import get_encoder_lvl2, save_model, load_model, _model, _encoder_lvl2
from preprocessing import get_features_for_train
from embeddings_fasttext import create_vocabulary
from sequences_fasttext import sequence_for_train
from helpers import exists
from config import *
from model import *
from pygoose import kg
import os
import datetime


tensorboard = TensorBoard(log_dir='./logs', write_graph=True)


def load_training_samples():
    X_train_descriptions = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_descriptions_train.pickle')
    X_test_descriptions = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_descriptions_test.pickle')
    X_train_titles = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_titles_train.pickle')
    X_test_titles = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_titles_test.pickle')
    y_train_lvl2 = kg.io.load(project.features_dir + 'y_train.pickle')
    y_test_lvl2 = kg.io.load(project.features_dir + 'y_test.pickle')
    encoder_lvl2 = LabelEncoder()
    encoder_lvl2.fit(np.concatenate((y_train_lvl2, y_test_lvl2), axis=0))
    kg.io.save(encoder_lvl2, ENCODER_LVL2)
    encoded_y_train_lvl2 = encoder_lvl2.transform(y_train_lvl2)
    encoded_y_test_lvl2 = encoder_lvl2.transform(y_test_lvl2)
    y_train_encoded_lvl2 = np_utils.to_categorical(encoded_y_train_lvl2)
    y_test_encoded_lvl2 = np_utils.to_categorical(encoded_y_test_lvl2)
    return X_train_descriptions, X_test_descriptions, X_train_titles, X_test_titles, \
           y_train_encoded_lvl2, y_test_encoded_lvl2


def workflow():
    if not exists(project.preprocessed_data_dir + 'tokens_lowercase_spellcheck_no_stopwords_titles_train.pickle'):
        get_features_for_train()
    if not exists(project.aux_dir + 'fasttext_vocab.vec'):
        create_vocabulary()
    if not exists(project.preprocessed_data_dir + 'sequences_fasttext_descriptions_train.pickle'):
        sequence_for_train('titles', MAX_SEQUENCE_LENGTH_TITLES)
    if not exists(project.preprocessed_data_dir + 'sequences_fasttext_titles_train.pickle'):
        sequence_for_train('descriptions', MAX_SEQUENCE_LENGTH)
    if not exists(model_dir):
        print('Creating model directory: %s' % model_dir)
        os.mkdir(model_dir)


def main_func():
    workflow()
    np.random.seed(7)
    log_metrics = ['categorical_accuracy',
                   'categorical_crossentropy',
                   'top_k_categorical_accuracy',
                   batch_pairwise_metrics]
    #load model if exists
    last_epoch, model_checkpoint_path = find_last_checkpoint(model_dir)
    initial_epoch = 0
    if model_checkpoint_path is not None:
        print('Loading epoch {0:d} from {1:s}'.format(last_epoch, model_checkpoint_path))
        _cust_objects = {'batch_pairwise_metrics': batch_pairwise_metrics}
        model = load_model(model_checkpoint_path, custom_objects=_cust_objects)
        initial_epoch = last_epoch + 1
    else:
        print('Building new model')
        model = create_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=log_metrics)

    print(model.summary())
    #plot_model(model_2, to_file= get_rec_sys_dir() + '/CNN.png', show_shapes=True)

    model_saver = ModelCheckpoint('CNN-text-classification-keras/logs/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    train_tboard_logger = FilterTensorBoard(log_dir=train_log_dir,
                                            histogram_freq=2,
                                            write_graph=True,
                                            write_images=False,
                                            log_regex=r'^(?!val).*')

    val_tboard_logger = FilterTensorBoard(log_dir=val_log_dir,
                                          histogram_freq=0,
                                          write_graph=False,
                                          write_images=False,
                                          log_regex=r"^val")
    timer = BatchTimer()

    _callbacks = [model_saver, timer, train_tboard_logger, val_tboard_logger]

    if initial_epoch < nb_epoch:
        training_start_time = datetime.datetime.now()
        print('{0}: Starting training'.format(training_start_time))
        X_train_descriptions, X_test_descriptions, X_train_titles, \
        X_test_titles, y_train_encoded_lvl2, y_test_encoded_lvl2 = load_training_samples()
        model.fit([X_train_descriptions, X_train_titles],
                  y_train_encoded_lvl2,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=1, callbacks=_callbacks,
                  validation_data=([X_test_descriptions, X_test_titles], y_test_encoded_lvl2))
    save_model(model)
    print('{0}: Finished'.format(datetime.datetime.now()))


def evaluate():
    start_time = datetime.datetime.now()
    X_test_descriptions = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_descriptions_test.pickle')
    X_test_titles = kg.io.load(project.preprocessed_data_dir + 'sequences_fasttext_titles_test.pickle')
    y_test_lvl2 = kg.io.load(project.features_dir + 'y_test.pickle')
    get_encoder_lvl2()
    encoded_y_test_lvl2 = _encoder_lvl2.transform(y_test_lvl2)
    y_test_encoded_lvl2 = np_utils.to_categorical(encoded_y_test_lvl2)
    load_model()
    scores = _model.evaluate([X_test_descriptions, X_test_titles],
                             y_test_encoded_lvl2,
                             batch_size=batch_size,
                             verbose=1)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('\n')
    print('Elapsed Time: %s' % str(elapsed_time))
    print("Loss: %1.4f. Accuracy: %.2f%%" % (scores[0], scores[1] * 100))


