""" Various common loss functions """

import numpy as np

def compute_loss_mse(y, tx, w):
    """
    Calculate the MSE loss
    """
    n, d = tx.shape
    e = y - tx.dot(w)

    return e.dot(e)/(2*n)

def compute_loss_mae(y, tx, w):
    """
    Calculate MAE loss.
    """
    n, d = tx.shape
    e = y - tx.dot(w)

    return sum(abs(e))/(n)
