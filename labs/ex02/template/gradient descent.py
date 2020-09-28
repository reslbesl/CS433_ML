"""Gradient Descent"""
import numpy as np
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from costs import *

def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    n, d = tx.shape

    e = y - tx.dot(w)

    return -tx.T.dot(e) / n


def compute_subgradient_mae(y, tx, w):
    common=np.sign(y-np.dot(tx,w))
    return -1/len(y)*np.array([np.sum(common), np.sum(tx[:,1]*common)])


def gradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for a two-parameter model."""

    ws = [initial_w]
    initial_loss = compute_loss_mae(y, tx, initial_w)
    losses = [initial_loss]

    w = initial_w

    for n_iter in range(max_iters):
        # Compute subgradient
        gradient = compute_subgradient_mae(y, tx, w)

        # Update model parameters
        w = w - gamma * gradient

        # Compute new loss
        loss = compute_loss_mae(y, tx, w)

        # Store w and loss
        ws.append(w)
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def gradient_descent_mse(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for a two-parameter model."""

    ws = [initial_w]
    initial_loss = compute_loss_mse(y, tx, initial_w)
    losses = [initial_loss]

    w = initial_w

    for n_iter in range(max_iters):
        # Compute gradient
        gradient = compute_gradient_mse(y, tx, w)

        # Update model parameters
        w = w - gamma * gradient

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        # Store w and loss
        ws.append(w)
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws