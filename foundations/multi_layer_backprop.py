import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z_1 = W1 @ x + b1

        a_1 = Relu(z_1)

        z_2 = W2 @ a_1 + b2

        loss = np.mean(np.square(z_2 - y_true))

        dz2 = 2 * (z_2 - y_true) / (len(y_true))

        dw2 = (dz2.T) * a_1

        db2 = dz2

        da1 = dz2 * W2

        dz1 = da1 * np.where(z_1 > 0.0, 1.0, 0.0)

        dw1 = (dz1.T) * x

        db1 = dz1
        dw1 = np.round(dw1, 4)
        dw1[dw1 == -0.0] = 0.0
        ans = {
            'loss': round(loss, 4),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1[0], 4).tolist(),
            'dW2': np.round([dw2], 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }

        return ans


def Relu(z):

    for i in range(len(z)):

        if z[i] < 0.0:
            z[i] = 0.0

    return z


