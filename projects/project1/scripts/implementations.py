import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return: (w, loss)
    """

    assert gamma > 0, "Step size gamma must be positive."

    w = initial_w
    loss = compute_loss_mse(y, tx, initial_w)

    for n_iter in range(max_iters):
        # Compute gradient
        g = compute_gradient_mse(y, tx, w)

        # Update parameters according to gradient
        w = w  - gamma * g

        # Compute new loss
        loss = compute_loss_mse(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss


def compute_loss_mse(y, tx, w):
    """
    Calculate the MSE loss under weights vector w
    """
    n, d = tx.shape
    e = y - tx.dot(w)

    return e.dot(e)/(2*n)


def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    n,d = tx.shape

    e = y - tx.dot(w)

    return -tx.T.dot(e) / n
