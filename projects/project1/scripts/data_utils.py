# -*- coding: utf-8 -*-
""" Helper functions for data splitting and test data generation"""

import numpy as np

SEED = 42

def train_eval_split(y, tx, split_ratio, seed=SEED):
    """
    Split a dataset into train and evaluation sets based on the split ratio.

    If split_ratio is 0.8 you will have 80% of your data set dedicated to training
    and the rest dedicated to testing

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param split_ratio: float: scalar in (0, 1] that defines what fraction of records will be assigned to the training set
    :param seed: int: seed for pseudo-random number generator

    :return (tx_train, y_train): tuple: training data
    :return (tx_test, y_test): tuple: test data
    """
    # set seed for reproducibility
    np.random.seed(seed)

    num_samples = len(y)
    num_train = int(np.ceil(num_samples * split_ratio))

    shuffle_indices = np.random.permutation(num_samples)
    shuffled_y = y[shuffle_indices]
    shuffled_x = tx[shuffle_indices]

    return (shuffled_x[:num_train], shuffled_y[:num_train]), (shuffled_x[num_train:], shuffled_y[num_train:])


def k_fold_iter(y, tx, k_fold, seed=SEED):
    """
    Generate a k-fold iterator for dataset (tx, y)

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives a k-fold train-test split of the data.

    Example use:
    for (x_train y_train), (x_test, y_test) in k_fold_iter(y, tx, 4):
        <DO_SOMETHING>

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param k_fold: int: number of dataset folds to be returned
    :param seed: int: seed for pseudo-random number generator
    """
    np.random.seed(seed)

    num_samples = len(y)
    interval = int(num_samples / k_fold)

    indices = np.random.permutation(num_samples)

    for k in range(k_fold):
        test_idx = indices[k * interval: (k + 1) * interval]
        train_idx = np.ones(num_samples).astype(bool)
        train_idx[test_idx] = False

        yield (tx[train_idx], y[train_idx]), (tx[test_idx], y[test_idx])


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True, seed=SEED):
    """
    Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    :param y: np.array: (n, ): array containing the target variable values of n record
    :param tx: np.array: (n, d): array containing the (normalised) indepent variable values of n records
    :param batch_size: int: size of each batch
    :param: num_batches: int: number of batches
    :param shuffle: bool: Whether to shuffle data to avoid ordering in original data
    :param seed: int: seed for pseudo-random number generator
    """
    # set seed for reproducibility
    np.random.seed(seed)

    num_samples = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, num_samples)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index], batch_num