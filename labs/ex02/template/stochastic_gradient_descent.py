# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

import numpy as np
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from costs import *


def compute_stoch_gradient_mse(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    n, d = tx.shape

    e = y - tx.dot(w)

    return -tx.T.dot(e) / n


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def stochastic_gradient_descent_mse(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    n,d = tx.shape

    ws = [initial_w]
    initial_loss = compute_loss_mse(y, tx, initial_w)
    losses = [initial_loss]

    w = initial_w
    n_iter = 0

    for batch_y, batch_tx in batch_iter(y, tx, batch_size, max_iters):
        # Compute gradient for current batch
        gradient = compute_stoch_gradient_mse(batch_y, batch_tx, w)

        # Update model parameters
        w = w - gamma * gradient

        # Compute new loss
        loss = compute_loss_mse(y, tx, initial_w)

        # Store w and loss
        ws.append(w)
        losses.append(loss)

        print("Stochastic GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        n_iter += 1

    return losses, ws