import pickle
import os


def get_rec_sys_dir():
    return os.path.abspath(os.path.dirname(__file__))


def save_csv_to_file(path, data):
    data.to_csv(path, sep=',', index=False, encoding='utf-8')


def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


def get_prediction_with_precision(classes, predict_proba, cat_num=3, with_proba=False):
    predictions_proba = predict_proba
    return [
        [
            (
                classes[pos], proba[pos]
            )
            for pos, proba in sorted(
            enumerate(predictions),
            key=lambda arg: arg[1], reverse=True
        )[:cat_num]
        ]
        if with_proba is False
        else
        [
            (
                classes[pos]
            )
            for pos, proba in sorted(
            enumerate(predictions),
            key=lambda arg: arg[1], reverse=True
        )[:cat_num]
        ]
        for i, predictions in enumerate(predictions_proba)
    ]


def transform_prediction(predicted):
    predicted = [item for sublist in predicted for item in sublist]
    return predicted


def top_k_accuracy(y_test, predict_proba, classes, k):
    predictions = get_prediction_with_precision(classes, predict_proba, k, True)
    answer = [1 if y_test[i] in predictions[i] else 0 for i in range(len(predictions))]
    return answer


