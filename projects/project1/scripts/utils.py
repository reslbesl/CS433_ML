# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

from os import path

from data_utils import standardise, standardise_to_fixed

SEED = 42
LABELS = {'b': 0, 's': 1}
PRI_JET_NUM_IDX = 22


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), input_data (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = LABELS['b']
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, thresh=0):
    """
    Generates class predictions given weights, and a test data matrix.
    Applies a threshold function g(xT.dot(w)) to the dot product between feature and weights vector to decide which class data point x belongs to.
    *Threshold value needs to be adjusted depending on the training function used to generate weights*

    For linear regression: Use thresh=0
    For logistic regression: Use thresh=0.5
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= thresh)] = LABELS['b']
    y_pred[np.where(y_pred > thresh)] = LABELS['s']
    
    return y_pred


def get_accuracy(y_pred, y_true):
    """ Calculate accuracy of predictions y_pred in predicting true labels y_true """

    return sum(y_pred == y_true)/len(y_true)


def eval_model(y, x, w, thresh=0):
    y_pred = predict_labels(w, x, thresh)
    acc = get_accuracy(y_pred, y)

    return acc


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to AICrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    # Re-code to match labels on submission platform
    y_pred[y_pred == LABELS['b']] = -1
    y_pred[y_pred == LABELS['s']] = 1

    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def load_data(data_path, seed=SEED):
    # Load train data
    y, x, ids = load_csv_data(path.join(data_path, 'train.csv'))

    # Split into train and evaluation set
    (x_train, y_train), (x_eval, y_eval) = train_eval_split(y, x, split_ratio=.7, seed=seed)

    # Load test data
    y_test, x_test, ids_test = load_csv_data(path.join(data_path, 'test.csv'))

    # Standardise to training set mean
    x_train, mu, sigma = standardise(x_train)
    x_eval = standardise_to_fixed(x_eval, mu, sigma)
    x_test = standardise_to_fixed(x_test, mu, sigma)

    # Generate x tilde
    tx_train = np.c_[np.ones(len(y_train)), x_train]
    tx_eval = np.c_[np.ones(len(y_eval)), x_eval]
    tx_test = np.c_[np.ones(len(y_test)), x_test]

    return (y_train, x_train, tx_train), (y_eval, x_eval, tx_eval), (y_test, x_test, tx_test, ids_test)



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


def get_pos_rates(y_pred, y_true):
    """ Calculate true and false positive rates of predicted labels y_pred in predicting true labels y_true """
    tp = sum(y_pred[y_true == 1] == 1)
    fp = sum(y_true == 1) - tp

    return tp, fp


def get_neg_rates(y_pred, y_true):
    """ Calculate true and false negative rates of predicted labels y_pred in predicting true labels y_true """
    tn = sum(y_pred[y_true == -1] == -1)
    fn = sum(y_true == -1) - tn

    return tn, fn


def get_f1_score(y_pred, y_true):
    """ Calculate F1 score of predicted labels y_pred in predicting true labels y_true """
    tp, fp = get_pos_rates(y_pred, y_true)
    tn, fn = get_neg_rates(y_pred, y_true)

    return tp/(tp + (fp + fn)/2.)



