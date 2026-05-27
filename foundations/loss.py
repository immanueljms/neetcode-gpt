import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        term = y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        ans = -1*(sum(term))/n
        return round(ans,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        summer=0
        for i in range(len(y_true)):
            summer+= sum(y_true[i]*np.log(y_pred[i]+1e-7))
        summer = -1*summer/(len(y_true))    
        return round(summer , 4)    



