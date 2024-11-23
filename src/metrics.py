import numpy as np


def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of the model

    Args:
        y_true (np.ndarray): True labels of shape (n_videos,)
        y_pred (np.ndarray): Predicted labels of shape (n_videos,)

    Returns:
        float: Accuracy
    """
    assert len(y_true) == len(y_pred), 'Length of y_true and y_pred should be same'

    return np.sum(y_true == y_pred) / len(y_true)

def calc_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calculate the confusion matrix

    Args:
        y_true (np.ndarray): True labels of shape (n_videos,)
        y_pred (np.ndarray): Predicted labels of shape (n_videos,)
        num_classes (int): Number of classes

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    assert len(y_true) == len(y_pred), 'Length of y_true and y_pred should be same'

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

    return confusion_matrix