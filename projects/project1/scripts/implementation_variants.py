import numpy as np

from costs import *
from data_utils import *


# Variants
def lasso_GD(y, tx, initial_w, max_iters, gamma, lambda_, threshold=1e-9, verbose=False):
    """
    Lasso regression using subgradient method

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records
    :param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size
    :param lambda_: float: regularisation parameter

    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, w)
    losses = [loss]

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_lasso(y, tx, w, lambda_)

        # Update parameters according to gradient
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Gradient Descent({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

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


def logistic_regression_SGD(y, tx, w_initial, max_iters, gamma, batch_size=10, threshold=1e-9, verbose=False):
    """

    :param y: np.array: (n, ): array containing the binary class labels of n records. Class labels must be encoded as {0, 1}!
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records. Must include a constant offset variable as first feature!
    ::param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size
    :param threshold:
    :param verbose:
    :return:
    """
    # Check correct class label encodings
    labels = set(y)
    assert len(labels) == 2, f"{len(labels)} classes detected. Function implements binary classification only."
    assert len(labels.difference({0, 1})) == 0, "Class labels must be encoded as {0, 1}"

    # Init
    w = w_initial
    loss = compute_loss_logreg_mean(y, tx, w)
    losses = [loss]

    for batch_y, batch_x, n_iter in batch_iter(y, tx, batch_size, num_batches=max_iters):
        # Compute gradient
        grad = compute_gradient_logreg_mean(batch_y, batch_x, w)

        # Update model parameters
        w = w - gamma * grad

        # Compute new loss
        loss = compute_loss_logreg_mean(batch_y, batch_x, w)
        losses.append(loss)

        if verbose:
            if n_iter % 100 == 0:
                print("Stochastic Gradient Descent ({bi}/{ti}): loss={l}, gradient={g}".format(bi=n_iter, ti=max_iters - 1, l=loss, g=np.linalg.norm(grad)))

        # Check terminating conditions
        if np.isnan(loss):
            print('Divergence warning: Terminate because loss is NaN.')
            # Will return loss and weights of last step
            break

    return w, loss


def logistic_regression_mean(y, tx, initial_w, max_iters, gamma, threshold=1e-9, verbose=False):
    """

    :param y: np.array: (n, ): array containing the binary class labels of n records. Class labels must be encoded as {0, 1}!
    :param tx: np.array: (n, d): array containing the (normalised) independent variable values of n records. Must include a constant offset variable as first feature!
    ::param initial_w: np.array: (d, ): array containing the initial model parameter values
    :param max_iters: int: scalar value indicating the maximum number of iterations to run
    :param gamma: float: gradient step-size
    :param threshold: float: defines termination condition based on delta in loss from step k to k+1 being smaller
    :param verbose: bool: whether to print out additional info
    :return:
    """
    # Check correct class label encodings
    labels = set(y)
    assert len(labels) == 2, "More than two classes detected. Function implements binary classification only."
    assert len(labels.difference({0, 1})) == 0, "Class labels must be encoded as {0, 1}"

    # Init
    w = initial_w
    loss = compute_loss_logreg_mean(y, tx, w)
    losses = [loss]

    for n_iter in range(max_iters):
        # Compute gradient
        grad = compute_gradient_logreg_mean(y, tx, w)

        # Update model parameters
        w -= gamma * grad

        # Compute new loss
        loss = compute_loss_logreg_mean(y, tx, w)
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


def least_squares_SGD_robbinson(y, tx, initial_w, max_iters, r_gamma=.7, verbose=False):
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
            if verbose:
                print("Stochastic GD({bi}/{ti}): loss={l}, gradient={grad}, gamma={gam}".format(bi=n_iter, ti=max_iters - 1, l=loss, grad=np.linalg.norm(grad), gam=gamma))

    return w, loss
