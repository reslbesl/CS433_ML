# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

#MES
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse.
    """
    return 1/(2*len(y))*np.sum((y-np.dot(tx,w))**2)
    raise NotImplementedError

def compute_lossMAE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MAE
    # ***************************************************
    return 1/len(y)*np.sum(np.abs(y-np.dot(tx,w)))