# -*- coding: utf-8 -*-
"""Various cost functiosn for project1"""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss under weights vector w."""
    e = y - tx.dot(w)

    return e.dot(e)/(2*len(e))


def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    e = y - tx.dot(w)

    return -tx.T.dot(e) / len(e)


def compute_loss_lasso(y, tx, w, lambda_):
    """Calculate the Lasso loss under weights vector w."""
    e = y - tx.dot(w)

    return e.dot(e)/(2 * len(e)) + lambda_ * sum(abs(w))


def compute_gradient_lasso(y, tx, w, lambda_):
    """Compute gradient under Lasso regression."""
    e = y - tx.dot(w)
    subgrad = lambda_ * np.sign(w)

    return -tx.T.dot(e)/len(e) + subgrad