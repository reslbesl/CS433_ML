# -*- coding: utf-8 -*-
"""Implementations of main functions required for project1"""

import numpy as np

from data_utils import *
from costs import *


def least_squares(y, x):
    """
    Least-squares regression using normal equations

    Takes as input a dataset (y, x) and finds the weights vector w
    that is the solution to the least-squares problem (X^T * X) * w = X^T * y

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param x: np.array: (n, d): array containing the (normalised) indepent variable values of n records

    :return w: np.array: (d, ): array containing the model weights w that minimise the MSE loss
    :return loss: float: mean-squared error under w
    """
    # Compute Gram Matrix
    gram = x.T.dot(x)

    # Solve the linear system from normal equations
    w = np.linalg.solve(gram, x.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, x, w)

    return w, loss


def ridge_regression(y, x, lambda_):
    """
    Normal equations using L2 regularization

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param x: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param lambda_: float: penalty parameter
    
    
    :return: (w, loss)
    """
    assert lambda_ > 0, "Penalty factor must be positive."

    if len(x.shape) > 1:
        num_samples, num_dims = x.shape
    else:
        num_samples, num_dims = len(y), 1

    # Compute Gram matrix
    gram = x.T.dot(x)

    # Compute identity dxd matrix
    eye = np.identity(num_dims)

    # Compute lambda prime as lamda*2N
    plambda = lambda_ * 2 * num_samples

    # Solve the linear system from normal equation under L2 regularization
    w = np.linalg.solve((gram + plambda * eye), x.T.dot(y))

    # Compute loss
    loss = compute_loss_mse(y, x, w)

    return w, loss


def least_squares_GD(y, x, initial_w, max_iters, gamma, threshold=1e-9, verbose=False):
    """
    Linear regression using gradient descent

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param x: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert gamma > 0, "Step size gamma must be positive."

    # Init
    w = initial_w
    loss = compute_loss_mse(y, x, w)
    losses = [loss]

    if verbose:
        print("Gradient Descent: Initial loss={l}".format(l=loss))

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_mse(y, x, w)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, x, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

        # Check termination conditions
        if np.isnan(loss):
            print('Divergence warning: Terminate because loss is NaN.')
            break

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print('Loss convergence: Terminate because loss did not change by more than threshold.')
            break

    return w, loss


def least_squares_SGD(y, x, initial_w, max_iters, gamma, threshold=1e-9, verbose=False):
    """
    Linear regression using stochastic gradient descent with default mini-batch size 1.

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param x: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size

    :return: (w, loss)
    """
    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, x, w)
    losses = [loss]

    for batch_y, batch_tx, n_iter in batch_iter(y, x, batch_size=1, num_batches=max_iters):

        # Compute gradient for current batch
        grad = compute_gradient_mse(batch_y, batch_tx, w)

        # Update model parameters
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, x, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Stochastic GD({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

        # Check termination conditions
        if np.isnan(loss):
            print('Divergence warning: Terminate because loss is NaN.')
            break

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print('Loss convergence: Terminate because loss did not change by more than threshold.')
            break

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold=1e-9, verbose=False):
    """

    :param y: np.array: (n, ): array containing the binary class labels of n records. Class labels must be encoded as {0, 1}!
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records. Must include a constant offset variable as first feature!
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size
    :param threshold: float: defines termination condition based on delta in loss from step k to k+1 being smaller
    :param verbose: bool: whether to print out additional info
    
    :return: (w, loss)
    """
    # Check correct class label encodings
    labels = set(y)
    assert len(labels) == 2, "More than two classes detected. Function implements binary classification only."
    assert len(labels.difference({0, 1})) == 0, "Class labels must be encoded as {0, 1}"

    # Init
    w = initial_w
    loss = compute_loss_logreg(y, tx, w)
    losses = [loss]

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_logreg(y, tx, w)

        # Update model parameters
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_logreg(y, tx, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Gradient Descent ({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

        # Check termination conditions
        if np.isnan(loss):
            print('Divergence warning: Terminate because loss is NaN.')
            break

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print('Loss convergence:Terminate because loss did not change by more than threshold.')
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8, verbose=False):
    """

    :param y: np.array: (n, ): array containing the binary class labels of n records. Class labels must be encoded as {0, 1}!
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records. Must include a constant offset variable as first feature!
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size
    :param threshold: float: defines termination condition based on delta in loss from step k to k+1 being smaller
    :param verbose: bool: whether to print out additional info
    :param lambda_: float: penalty parameter
    
    :return: (w, loss)
    """
    
    w = initial_w
    loss = compute_loss_logreg(y, tx, w)
    losses = [loss]

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_logreg_regl2(y, tx, w, lambda_)

        # Update model parameters
        w -= gamma * grad

        # Compute new (non-penalised) loss
        loss = compute_loss_logreg(y, tx, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Gradient Descent ({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

        # Check termination conditions
        if np.isnan(loss):
            if verbose:
                print('Divergence warning: Terminate because loss is NaN.')
            # Will return loss and weights of last step
            break

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            if verbose:
                print('Loss convergence:Terminate because loss did not change by more than threshold.')
            break

    return w, loss


