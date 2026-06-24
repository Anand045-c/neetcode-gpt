import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x_arr = np.array(x)          
        W1_arr = np.array(W1)        
        b1_arr = np.array(b1)        
        W2_arr = np.array(W2)        
        b2_arr = np.array(b2)        
        y_true_arr = np.array(y_true)
        z1 = np.dot(x_arr, W1_arr.T) + b1_arr
        x2 = np.maximum(0, z1)       
        

        y_pred = np.dot(W2_arr, x2) + b2_arr 
        
        n = y_pred.size
        loss = np.mean((y_pred - y_true_arr) ** 2)
        

        dL_dypred = (2.0 / n) * (y_pred - y_true_arr)
        db2 = dL_dypred
        
        dW2 = np.outer(dL_dypred, x2)
        
        dx2 = np.dot(dL_dypred, W2_arr).flatten()
        
        dz1 = dx2 * (z1 > 0) # Shape: (2,)
        
        db1 = dz1
        
        dW1 = np.outer( dz1, x_arr)
        
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }