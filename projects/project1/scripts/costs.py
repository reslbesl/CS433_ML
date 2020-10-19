# -*- coding: utf-8 -*-
"""Various cost functiosn for project1"""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss under weights vector w."""
    e = y - tx.dot(w)

    return e.dot(e.T)/(2*len(e)) 


def compute_gradient_mse(y, tx, w):
    """Compute the gradient under MSE loss."""
    e = y - tx.dot(w)

    return -tx.T.dot(e) / len(e)


def compute_loss_ridge(y, tx, w, lambda_):
    loss = compute_loss_mse(y, tx, w)
    penal_loss = loss + lambda_ * w.dot(w)

    return penal_loss


def compute_loss_lasso(y, tx, w, lambda_):
    """Calculate the Lasso loss under weights vector w."""
    e = y - tx.dot(w)

    return e.dot(e)/(2 * len(e)) + lambda_ * sum(abs(w))


def compute_gradient_lasso(y, tx, w, lambda_):
    """Compute gradient under Lasso regression."""
    e = y - tx.dot(w)
    subgrad = lambda_ * np.sign(w)

    return -tx.T.dot(e)/len(e) + subgrad


def compute_loss_logreg(y, tx, w):
    """Compute the loss under a logistic regression model (negative log likelihood) with class labels {0, 1}."""
    assert len(set(y).difference({0., 1.})) == 0, "Class labels must be encoded as {0, 1}"

    z = tx.dot(w)

    return np.sum(np.log(1 + np.exp(z)) - y * z)


def compute_loss_logreg_mean(y, tx, w):
    loss = compute_loss_logreg(y, tx, w)

    return loss/len(y)


def compute_gradient_logreg(y, tx, w):
    """Compute the gradient of the negative log-likelihood under a logistic regression model  with class labels {0, 1}."""
    assert len(set(y).difference({0., 1.})) == 0, "Class labels must be encoded as {0, 1}"

    s = sigmoid(tx.dot(w)) - y
    grad = tx.T.dot(s)

    return grad


def compute_gradient_logreg_mean(y, tx, w):
    grad = compute_gradient_logreg(y, tx, w)

    return grad/len(y)



def compute_hessian_logreg(tx, w):
    """Compute the Hessian of the negative log-likelihood under a logistic regression model."""
    t = tx.dot(w)
    s = np.diag(sigmoid(t)*(1 - sigmoid(t)))

    return tx.T.dot(s).dot(tx)


def compute_loss_logreg_reg(y, tx, w, lambda_):
    """Compute the loss of the negative log-likelihood under a logistics regression model under L2 regularisation"""
    loss = compute_loss_logreg(y, tx, w)
    penal_loss = loss + lambda_ / 2 * np.linalg.norm(w)

    return penal_loss


def compute_gradient_logreg_reg(y, tx, w, lambda_):
    """Compute the gradient of the negative log-likelihood under a logistics regression model under L2 regularisation"""
    grad = compute_gradient_logreg(y, tx, w)
    penal_grad = grad + lambda_ * w

    return penal_grad


def sigmoid(t):
    """apply the sigmoid function on t."""

    return np.exp(t)/(1 + np.exp(t))
