# -*- coding: utf-8 -*-
"""Implementations of main functions required for project1"""

import numpy as np

from data_utils import *
from costs import *


def least_squares(y, tx):
    """
    Least-squares regression using normal equations

    Takes as input a dataset (y, tx) and finds the weights vector w
    that is the solution to the least-squares problem (X^T * X) * w = X^T * y

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records

    :return w: np.array: (d, ): array containing the model weights w that minimise the MSE loss
    :return loss: float: mean-squared error under w
    """
    # Compute Gram Matrix
    gram = tx.T.dot(tx)

    # Solve the linear system from normal equations
    w = np.linalg.solve(gram, tx.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Normal equations using L2 regularization

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param lambda_: float: penalty parameter
    """
    assert lambda_ > 0, "Penalty factor must be positive."

    if len(tx.shape) > 1:
        num_samples, num_dims = tx.shape
    else:
        num_samples, num_dims = len(y), 1

    # Compute Gram matrix
    gram = tx.T.dot(tx)

    # Compute identity dxd matrix
    eye = np.identity(num_dims)

    # Compute lambda prime as lamda*2N
    plambda = lambda_ * 2 * num_samples

    # Solve the linear system from normal equation under L2 regularization
    w = np.linalg.solve((gram + plambda * eye), tx.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial modeo parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_mse(y, tx, w)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):

            # Compute gradient for current batch
            grad = compute_gradient_mse(batch_y, batch_tx, w)

            # Update model parameters
            w = w - gamma * grad

            # Compute new loss
            loss = compute_loss_mse(y, tx, w)

            print("Stochastic GD({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss


# Variants
def lasso_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Lasso regression using subgradient method

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_lasso(y, tx, w, lambda_)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

    return w, loss


def least_squares_SGD_robbinson(y, tx, initial_w, max_iters, r_gamma=.7):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1 and decreasing step-size.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert .5 < r_gamma < 1, 'Parameter r must be in (0.5, 1)'

    w = initial_w
    loss = compute_loss_mse(y, tx, w)

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):

            # Compute gradient for current batch
            grad = compute_gradient_mse(batch_y, batch_tx, w)

            # Update step size
            gamma = 1/((n_iter + 1)**r_gamma)

            # Update model parameters
            w = w - gamma * grad

            # Compute new loss
            loss = compute_loss_mse(y, tx, w)

            print("Stochastic GD({bi}/{ti}): loss={l}, gradient={grad}, gamma={gam}".format(bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad), gam=gamma))

    return w, loss



