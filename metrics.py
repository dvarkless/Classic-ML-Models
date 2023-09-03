import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def get_probabilities_copy(ans_arr: np.ndarray, num_classes) -> np.ndarray:
    """
        Function converts answers to probabilities
        (starting from 1)
        Example:
            >> self.num_classes = 3
            >> a = [1, 2, 3]
            >> self.get_probabilities(a)
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    buff_arr = np.zeros(shape=(ans_arr.shape[0], round(num_classes)-1))
    for i, val in enumerate(ans_arr):
        buff_arr[i, :] = _ans_as_probs_copy(val, num_classes)

    return buff_arr


def _ans_as_probs_copy(answer, num_classes):
    """
        Function converts answers to probabilities
    """
    out = np.zeros(round(num_classes)-1)
    out[int(answer) - 1] = 1

    return out


def show_metrics_matrix(preds, answer):
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ConfusionMatrixDisplay.from_predictions(
        answer,
        preds,
        labels,
        normalize="true",
    )
    plt.show()


def return_precision(preds, answer):
    return precision_score(answer, preds, average='macro', zero_division=1)


def return_recall(preds, answer):
    return recall_score(answer, preds, average='macro', zero_division=1)


def return_f1(preds, answer):
    return f1_score(answer, preds, average='macro', zero_division=1)


def return_roc_auc_ovo(preds, answer):
    num_classes = answer.max()
    preds = get_probabilities_copy(preds, num_classes)
    answer = get_probabilities_copy(answer, num_classes)
    return roc_auc_score(answer, preds, average='macro', multi_class='ovo')


def return_roc_auc_ovr(preds, answer):
    num_classes = answer.max()
    preds = get_probabilities_copy(preds, num_classes)
    answer = get_probabilities_copy(answer, num_classes)

    return roc_auc_score(answer, preds, average='macro', multi_class='ovr')


def return_accuracy(preds, answer):
    return accuracy_score(answer, preds)


def predictions_mean(preds, _):
    return preds.mean()


def predictions_std(preds, _):
    return preds.std()


def print_accuracy(preds, answer):
    print(accuracy_score(answer, preds))
